"""Exact alias spans and frozen local object observations, not dense segmentation."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


IMPLEMENTATION = "geometry-grounding-observation-diagnostic-v1-20261002"
REPO = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


@dataclass(frozen=True)
class GroundingConfig:
    maximum_prompt_tokens: int = 192
    box_threshold: float = .25
    windows_per_image: int = 2

    def signature(self):
        return {**asdict(self), "repo": REPO, "revision": REVISION,
            "alias_measurement": "arithmetic mean of sigmoid logits over exact phrase token spans",
            "queries": "all20 aliases, round-robin class order, no truncated or selected phrases",
            "absence": "no detected box supplies no negative evidence",
            "supervision": "extra frozen detection pretraining, not matched standalone DINO.text",
            "scope": "two image-only windows per fixed sample, not full-image dataset mIoU"}


def caption_spans(aliases, indices):
    text, spans = "", []
    for index in indices:
        phrase = aliases[index].lower().strip()
        if not phrase or "." in phrase:
            raise ValueError("Empty phrase or sentence separator inside an alias.")
        start = len(text)
        text += phrase
        spans.append((start, len(text)))
        text += ". "
    return text.strip(), spans


def prompt_chunks(tokenizer, aliases, parents, maximum_tokens=192):
    if len(aliases) != len(parents) or not aliases or maximum_tokens < 4:
        raise ValueError("Invalid alias bank or token budget.")
    groups = [[i for i, parent in enumerate(parents) if parent == c] for c in sorted(set(parents))]
    order = [group[k] for k in range(max(map(len, groups))) for group in groups if k < len(group)]
    chunks, current = [], []
    for index in order:
        caption, _ = caption_spans(aliases, [*current, index])
        size = len(tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"])
        if size > maximum_tokens and current:
            chunks.append(current)
            current = []
            caption, _ = caption_spans(aliases, [index])
            size = len(tokenizer(caption, add_special_tokens=True, truncation=False)["input_ids"])
        if size > maximum_tokens:
            raise ValueError("An entire alias exceeds the frozen prompt budget.")
        current.append(index)
    if current:
        chunks.append(current)
    output = []
    for indices in chunks:
        caption, spans = caption_spans(aliases, indices)
        encoded = tokenizer(caption, return_offsets_mapping=True, truncation=False)
        offsets = encoded["offset_mapping"]
        masks = np.array([[end > begin and end > start and begin < finish
                           for begin, end in offsets] for start, finish in spans], dtype=bool)
        if not masks.any(1).all() or len(encoded["input_ids"]) > maximum_tokens:
            raise RuntimeError("Missing phrase tokens or unexpectedly truncated caption.")
        output.append({"indices": indices, "caption": caption, "token_masks": masks,
                       "input_ids": encoded["input_ids"]})
    flattened = [index for chunk in output for index in chunk["indices"]]
    if sorted(flattened) != list(range(len(aliases))):
        raise RuntimeError("Alias coverage changed while packing captions.")
    return output


def fractional_box_support(boxes, side=32):
    """Area intersection with each patch; a box is never treated as a true mask."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or side < 1 or not torch.isfinite(boxes).all():
        raise ValueError("Invalid normalized xyxy boxes.")
    boxes = boxes.clamp(0, 1)
    if bool((boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError("Inverted boxes.")
    y, x = torch.meshgrid(torch.arange(side, device=boxes.device),
                          torch.arange(side, device=boxes.device), indexing="ij")
    x, y = x.flatten()/side, y.flatten()/side
    width = (torch.minimum(boxes[:, 2, None], x[None]+1/side)
             - torch.maximum(boxes[:, 0, None], x[None])).clamp_min(0)
    height = (torch.minimum(boxes[:, 3, None], y[None]+1/side)
              - torch.maximum(boxes[:, 1, None], y[None])).clamp_min(0)
    return (width*height*side**2).clamp(0, 1)


def box_observation(boxes, alias_scores, alias_indices, parents, classes, side=32, threshold=.25):
    if alias_scores.shape != (len(boxes), len(alias_indices)):
        raise ValueError("Box and alias-score dimensions differ.")
    if not 0 <= threshold <= 1 or not torch.isfinite(alias_scores).all():
        raise ValueError("Invalid box-confidence scores.")
    evidence = torch.zeros(side*side, classes, device=boxes.device)
    if len(boxes) == 0:
        return evidence
    support = fractional_box_support(boxes, side)
    parent = torch.as_tensor(parents, device=boxes.device)[alias_indices]
    for c in range(classes):
        if not bool((parent == c).any()):
            continue
        scores = alias_scores[:, parent == c].amax(-1)
        keep = scores > threshold
        if bool(keep.any()):
            evidence[:, c] = (support[keep].T*scores[keep]).amax(-1)
    return evidence


class FrozenGroundingObserver:
    def __init__(self, source, aliases, parents, device="cuda", config=GroundingConfig()):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.config, self.device = config, device
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            source, local_files_only=True, trust_remote_code=False, use_safetensors=True).to(device)
        self.model.eval().requires_grad_(False)
        if config.maximum_prompt_tokens > self.model.config.max_text_len:
            raise ValueError("Prompt budget exceeds pretrained maximum length.")
        self.aliases, self.parents = tuple(aliases), tuple(parents)
        self.chunks = prompt_chunks(self.processor.tokenizer, self.aliases, self.parents,
                                    config.maximum_prompt_tokens)

    @torch.inference_mode()
    def observe(self, rgb):
        from PIL import Image
        if rgb.shape != (1, 3, 512, 512) or not bool(torch.isfinite(rgb).all()):
            raise ValueError("Expected the exact original512 RGB window.")
        image = Image.fromarray((rgb[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8))
        rows = []
        for chunk in self.chunks:
            inputs = self.processor(images=image, text=chunk["caption"], return_tensors="pt",
                                    truncation=False).to(self.device)
            if inputs.input_ids[0].tolist() != chunk["input_ids"]:
                raise RuntimeError("Processor changed alias token sequence.")
            outputs = self.model(**inputs)
            probability = outputs.logits[0, :, :len(chunk["input_ids"])].float().sigmoid()
            masks = torch.as_tensor(chunk["token_masks"], device=self.device, dtype=torch.float32)
            scores = probability @ (masks/masks.sum(1, keepdim=True)).T
            max_scores = torch.stack([probability[:, mask.bool()].amax(-1) for mask in masks], -1)
            center, size = outputs.pred_boxes[0, :, :2], outputs.pred_boxes[0, :, 2:]
            boxes = torch.cat((center-size/2, center+size/2), -1).float().clamp(0, 1)
            if not torch.isfinite(scores).all() or not torch.isfinite(boxes).all():
                raise RuntimeError("Detector returned nonfinite measurements.")
            rows.append({"boxes": boxes.cpu().numpy(), "alias_scores": scores.cpu().numpy(),
                         "alias_max_scores": max_scores.cpu().numpy(),
                         "alias_indices": np.asarray(chunk["indices"], np.int64)})
        return rows
