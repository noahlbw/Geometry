"""Frozen SAM2 pixel supports for unchanged cached GroundingDINO observations.

SAM2 and Grounded-SAM are borrowed sources, not a new segmentation invention.
The experiment isolates spatial support without changing phrase confidences.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .grounded_local_observer import fractional_box_support


IMPLEMENTATION = "geometry-grounded-mask-likelihood-v1-20261002"
REPO = "facebook/sam2.1-hiera-tiny"
REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
PRIMARY = "Geometry_MaskLikelihood"
METHODS = ("Geometry", "BoxLocalLikelihood", "Geometry_LocalLikelihood",
           "MaskLocalLikelihood", "MeanProb_MaskLikelihood", "ShuffledMaskLikelihood", PRIMARY)


@dataclass(frozen=True)
class MaskObservationConfig:
    batch_size: int = 16
    confidence: float = .25

    def signature(self):
        return {**asdict(self), "repo": REPO, "revision": REVISION,
            "mask": "single frozen SAM2 mask; sigmoid of interpolated logits; mean per original16px patch",
            "support": "fractional original box support times SAM2 patch foreground probability",
            "neutrality": "constant original box support retains original box, not an arbitrary salient mask",
            "scores": "original phrase means and all20 aliases unchanged; no NMS or query cap",
            "supervision": "additional pretrained detection and segmentation supervision",
            "source_rerun": "no Geometry or detector rerun; SAM2 sees original image-only windows"}


def masked_box_support(boxes, mask_probability, valid):
    patches = len(valid)
    side = int(patches**.5)
    if (side*side != patches or valid.dtype != torch.bool or not bool(valid.any())
            or mask_probability.shape != (len(boxes), patches)
            or not torch.isfinite(mask_probability).all()
            or bool(((mask_probability < 0) | (mask_probability > 1)).any())):
        raise ValueError("Invalid observed mask probability or valid grid.")
    box_support = fractional_box_support(boxes, side)
    result = box_support*mask_probability
    constant = (box_support[:, valid] == box_support[:, valid][:, :1]).all(-1)
    result[constant] = box_support[constant]
    return result


class FrozenMaskObserver:
    def __init__(self, source, device="cuda", config=MaskObservationConfig()):
        from transformers import Sam2Model, Sam2Processor
        if config != MaskObservationConfig():
            raise ValueError("The complete mask-observation rule is frozen.")
        self.device, self.config = device, config
        self.processor = Sam2Processor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        self.model = Sam2Model.from_pretrained(source, local_files_only=True,
            trust_remote_code=False, use_safetensors=True).to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def observe(self, rgb, rows, valid):
        from PIL import Image
        if rgb.shape != (1, 3, 512, 512) or not bool(torch.isfinite(rgb).all()):
            raise ValueError("Original512 RGB window required.")
        image = Image.fromarray((rgb[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8))
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embeddings = self.model.get_image_embeddings(image_inputs["pixel_values"])
        masks, query_count, neutral_count = [], 0, 0
        for row in rows:
            boxes = torch.as_tensor(row["boxes"], dtype=torch.float32, device=self.device)
            scores = torch.as_tensor(row["alias_scores"], dtype=torch.float32, device=self.device)
            box_support = fractional_box_support(boxes)
            active = (scores > self.config.confidence).any(-1)
            constant = (box_support[:, valid] == box_support[:, valid][:, :1]).all(-1)
            indices = (active & ~constant).nonzero().flatten()
            probability = torch.zeros_like(box_support)
            for start in range(0, len(indices), self.config.batch_size):
                current = indices[start:start+self.config.batch_size]
                inputs = self.processor(original_sizes=[[512, 512]],
                    input_boxes=[(boxes[current]*512).cpu().tolist()], return_tensors="pt").to(self.device)
                outputs = self.model(image_embeddings=embeddings, input_boxes=inputs["input_boxes"],
                                     multimask_output=False)
                logits = self.processor.post_process_masks(outputs.pred_masks.float(), [[512, 512]],
                                                           binarize=False)[0]
                if logits.shape != (len(current), 1, 512, 512) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid SAM2 observation shape/value.")
                probability[current] = F.avg_pool2d(logits.sigmoid(), 16).reshape(len(current), 1024)
                query_count += len(current)
            masks.append(masked_box_support(boxes, probability, valid).cpu().numpy())
            neutral_count += int((active & constant).sum())
        return masks, {"mask_queries": query_count, "constant_box_queries": neutral_count,
                       "target_masks_loaded": False}
