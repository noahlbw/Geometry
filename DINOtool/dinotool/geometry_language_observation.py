"""Position-specific frozen language observation of actual Geometry supports."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .gear_ov import _crop_at


IMPLEMENTATION = "geometry-position-language-observation-v1-20261002"
REPOSITORY = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
METHODS = ("Geometry", "LanguageContext", "LanguageFoveal", "LanguageGrounded", "GroundedConsensus")


@dataclass(frozen=True)
class LanguageObservationConfig:
    probes_per_predicted_class: int = 4
    support_mass: float = .95
    minimum_crop_pixels: int = 64
    maximum_crop_pixels: int = 512
    minimum_encoder_pixels: int = 224 * 224
    maximum_encoder_pixels: int = 512 * 512
    option_orders: tuple = ("forward", "reverse")

    def signature(self):
        return {"language_observation": asdict(self), "implementation": IMPLEMENTATION,
                "geometry": "unchanged cached Geometry; image-valid relation defines actual foveal extent",
                "probe_sampling": "even confidence-rank positions, including extrema, within original predicted classes",
                "text": "same20 aliases/class listed together; no alias selection or target descriptions",
                "observation": "real RGB with a hollow target marker, context and Geometry foveal crop",
                "scoring": "next-token letter compatibility; restricted distribution is not calibrated correctness",
                "primary_rule": "all four view/order argmax choices agree on a non-abstaining class; otherwise exact Geometry",
                "scope": "selected original patch-center diagnostic, not a full dense segmentation model"}


def probe_indices(scores, valid, per_class=4):
    if (scores.ndim != 2 or valid.shape != (len(scores),) or valid.dtype != torch.bool
            or per_class < 1 or not bool(torch.isfinite(scores).all()) or scores.shape[1] < 2):
        raise ValueError("Require finite [N,C] scores and image-only validity.")
    top = scores.topk(2, dim=-1)
    prediction, margin = top.indices[:, 0], top.values[:, 0] - top.values[:, 1]
    chosen = []
    for category in range(scores.shape[1]):
        candidates = ((prediction == category) & valid).nonzero().flatten()
        if not len(candidates):
            continue
        order = torch.argsort(margin[candidates], stable=True)
        positions = torch.linspace(0, len(candidates)-1, min(per_class, len(candidates)), device=scores.device).round().long()
        chosen.extend(candidates[order[positions]].tolist())
    return torch.tensor(sorted(set(chosen)), dtype=torch.long, device=scores.device)


def foveal_extent(relation, valid, query, shape, config=LanguageObservationConfig(), patch_size=16):
    h, w = shape
    if (relation.shape != (h*w, h*w) or valid.shape != (h*w,) or valid.dtype != torch.bool
            or not 0 <= query < h*w or not bool(valid[query]) or not 0 < config.support_mass <= 1
            or not bool(torch.isfinite(relation).all()) or bool((relation < 0).any())):
        raise ValueError("Invalid Geometry support or query.")
    row = relation[query].float().masked_fill(~valid, 0)
    if float(row.sum()) <= 0:
        raise ValueError("Query has no usable support.")
    ids = torch.arange(h*w, device=relation.device)
    y, x = query // w, query % w
    radius = torch.maximum((ids // w-y).abs(), (ids % w-x).abs())
    order = radius.argsort(stable=True)
    cumulative = (row[order]/row.sum()).cumsum(0)
    index = int(torch.searchsorted(cumulative, torch.tensor(config.support_mass, device=row.device)).clamp_max(len(order)-1))
    pixels = (2*int(radius[order[index]])+1)*patch_size
    extent = max(config.minimum_crop_pixels, min(config.maximum_crop_pixels, pixels))
    return math.ceil(extent/patch_size)*patch_size


def option_prompt(names, aliases, parents, order):
    if sorted(order) != list(range(len(names))) or len(names) >= 25 or len(aliases) != len(parents):
        raise ValueError("Invalid class/alias order.")
    counts = [sum(parent == category for parent in parents) for category in range(len(names))]
    if any(count != 20 for count in counts):
        raise ValueError("The diagnostic must retain the same20 aliases per class.")
    lines = []
    for option, category in enumerate(order):
        words = [alias for alias, parent in zip(aliases, parents) if parent == category]
        lines.append(f"{chr(65+option)}. {names[category]} (vocabulary: {', '.join(words)})")
    return ("This is a remote-sensing image. Identify the physical surface or object exactly at "
            "the center of the hollow red target marker. The marker is an annotation, not part "
            "of the scene. Judge that location, not a nearby object or the whole scene. "
            "Select the best matching category. If the location is not visually identifiable, "
            "select Z. Answer with exactly one uppercase letter, without explanation.\n"
            + "\n".join(lines) + "\nZ. Cannot determine from visible content.")


def mark_image(rgb, point):
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Expected [3,H,W] RGB.")
    array = (rgb.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    x, y = point
    if not 0 <= x < image.width or not 0 <= y < image.height:
        raise ValueError("Target marker lies outside the image.")
    # The target center and its immediate pixels remain visible.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.line((x+5*dx, y+5*dy, x+11*dx, y+11*dy), fill=(255, 0, 0), width=2)
    return image


def consensus_predictions(original, choices, abstention):
    if choices.ndim != 3 or choices.shape[:2] != (2, 2) or choices.shape[-1] != len(original):
        raise ValueError("Require [two views,two orders,queries] choices.")
    first = choices[0, 0]
    accepted = (choices == first).all(0).all(0) & (first != abstention)
    return torch.where(accepted, first, original), accepted


class FrozenLanguageObserver:
    def __init__(self, source, names, aliases, parents, device="cuda", config=LanguageObservationConfig()):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        source = Path(source)
        self.manifest = json.loads((source/"language_source_manifest.json").read_text())
        if (self.manifest["repository"] != REPOSITORY or self.manifest["revision"] != REVISION
                or not self.manifest["frozen"]):
            raise ValueError("Require the pinned frozen public observation model.")
        self.names, self.aliases, self.parents = tuple(names), tuple(aliases), tuple(parents)
        self.config, self.device = config, torch.device(device)
        self.orders = (tuple(range(len(names))), tuple(reversed(range(len(names)))))
        self.prompts = [option_prompt(names, aliases, parents, order) for order in self.orders]
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True,
            trust_remote_code=False, min_pixels=config.minimum_encoder_pixels, max_pixels=config.maximum_encoder_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source,
            local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to(self.device).eval()
        self.model.requires_grad_(False)
        letters = [chr(65+i) for i in range(len(names))] + ["Z"]
        ids = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        if any(len(value) != 1 for value in ids) or len(set(value[0] for value in ids)) != len(ids):
            raise ValueError("Forced option letters must be distinct single tokens.")
        self.option_ids = torch.tensor([value[0] for value in ids], device=self.device)

    @torch.inference_mode()
    def score(self, image, prompt, order):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        selected = logits[self.option_ids]
        restored = torch.empty_like(selected)
        restored[torch.tensor(order, device=self.device)] = selected[:-1]
        restored[-1] = selected[-1]
        allowed_mass = float((selected.logsumexp(0)-logits.logsumexp(0)).exp())
        return restored, allowed_mass, int(inputs.input_ids.shape[-1])

    @torch.inference_mode()
    def __call__(self, rgb, relation, valid, local):
        started = time.perf_counter()
        if rgb.shape != (1, 3, 512, 512) or local.shape != (1024, len(self.names)):
            raise ValueError("Require an original512 window and unchanged patch scores.")
        indices = probe_indices(local, valid, self.config.probes_per_predicted_class)
        if not len(indices):
            raise ValueError("No image-valid queries.")
        observations, records = [], []
        for query in indices.tolist():
            y, x = (query//32)*16+8, (query%32)*16+8
            extent = foveal_extent(relation, valid, query, (32, 32), self.config)
            crop = _crop_at(rgb[0], y-extent//2, x-extent//2, extent)[0]
            images = (mark_image(rgb[0], (x, y)), mark_image(crop, (extent//2, extent//2)))
            views, masses, lengths = [], [], []
            for image in images:
                ordered = []
                for prompt, order in zip(self.prompts, self.orders):
                    scores, mass, length = self.score(image, prompt, order)
                    ordered.append(scores)
                    masses.append(mass)
                    lengths.append(length)
                views.append(torch.stack(ordered))
            observations.append(torch.stack(views))
            records.append({"query": query, "crop_extent": extent, "allowed_token_mass": masses, "input_tokens": lengths})
        logits = torch.stack(observations, dim=2)
        log_probability = logits.log_softmax(-1)
        combined = log_probability.mean((0, 1))
        choices = logits.argmax(-1)
        original = local[indices].argmax(-1)
        consensus, accepted = consensus_predictions(original, choices, len(self.names))
        predictions = {"Geometry": original,
            "LanguageContext": log_probability[0].mean(0).argmax(-1),
            "LanguageFoveal": log_probability[1].mean(0).argmax(-1),
            "LanguageGrounded": combined.argmax(-1), "GroundedConsensus": consensus}
        diagnostics = {"queries": int(len(indices)), "accepted": int(accepted.sum()),
                       "view_order_agreement": float((choices == choices[0, 0]).all(0).all(0).float().mean()),
                       "abstention_fraction": float((combined.argmax(-1) == len(self.names)).float().mean()),
                       "mean_crop_extent": float(np.mean([row["crop_extent"] for row in records])),
                       "mean_allowed_token_mass": float(np.mean([mass for row in records for mass in row["allowed_token_mass"]])),
                       "wall_seconds": time.perf_counter()-started}
        return {"indices": indices, "predictions": predictions, "option_logits": logits,
                "records": records, "accepted": accepted, "diagnostics": diagnostics}
