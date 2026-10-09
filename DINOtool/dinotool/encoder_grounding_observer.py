"""Native GroundingDINO encoder-position observations before object-query decoding.

Detection alignment is borrowed; it is not a pretrained pixel classifier or a
new contribution by itself. This tests a different observation, not a new gate.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .grounded_local_observer import FrozenGroundingObserver, REPO, REVISION
from .geometry_semantic_innovation import anchored_innovation


IMPLEMENTATION = "geometry-encoder-position-grounding-v1-20261002"
PRIMARY = "Geometry_EncoderCoupling"
METHODS = ("Geometry", "EncoderGrounding", "MeanLogit_Encoder", "MeanProb_Encoder",
           "Shuffled_EncoderCoupling", PRIMARY)


@dataclass(frozen=True)
class EncoderObservationConfig:
    side: int = 32
    temperature: float = .07
    maximum_prompt_tokens: int = 192

    def signature(self):
        return {**asdict(self), "repo": REPO, "revision": REVISION,
            "measurement": "native enc_outputs_class; phrase mean sigmoid over exact original token spans",
            "alignment": "levelwise validity-weighted bilinear probability interpolation, then equal-level pooling",
            "class": "uniform arithmetic mean of all20 alias probabilities, no threshold or deletion",
            "units": "b=.07*log(class probability); class softmax is normalized native probability",
            "unknown": "all-class zero or unobserved numeric support falls back to Geometry, not a confidence gate",
            "coupling": "fixed existing unit-weight Geometry fidelity reconstruction; standard quadratic solver",
            "supervision": "extra frozen detection pretraining; encoder location logits are NOT semantic-mask supervision"}


def align_encoder_aliases(logits, token_masks, spatial_shapes, valid, side=32):
    if (logits.ndim != 2 or token_masks.ndim != 2 or token_masks.shape[1] != logits.shape[1]
            or token_masks.dtype != torch.bool or not bool(token_masks.any(-1).all())
            or valid.shape != logits.shape[:1] or valid.dtype != torch.bool
            or not spatial_shapes or side < 1
            or any(h < 1 or w < 1 for h, w in spatial_shapes)
            or sum(h*w for h, w in spatial_shapes) != len(logits)):
        raise ValueError("Invalid encoder token spans, level geometry or validity.")
    used = token_masks.any(0)
    if not torch.isfinite(logits[valid][:, used]).all():
        raise ValueError("Nonfinite used native encoder-position logits.")
    scores = torch.where(valid[:, None] & used[None], logits.float(), 0.).sigmoid()
    masks = token_masks.float()
    aliases = scores @ (masks/masks.sum(-1, keepdim=True)).T
    aliases = aliases.masked_fill(~valid[:, None], 0.)
    numerator = torch.zeros(len(masks), side, side, device=logits.device)
    denominator = torch.zeros(1, side, side, device=logits.device)
    start = 0
    for height, width in spatial_shapes:
        stop = start+height*width
        field = aliases[start:stop].T.reshape(1, len(masks), height, width)
        support = valid[start:stop].float().reshape(1, 1, height, width)
        numerator += F.interpolate(field, (side, side), mode="bilinear", align_corners=False)[0]
        denominator += F.interpolate(support, (side, side), mode="bilinear", align_corners=False)[0]
        start = stop
    observed = denominator[0] > 0
    output = (numerator/denominator.clamp_min(torch.finfo(torch.float32).tiny)).flatten(1).T
    output[~observed.flatten()] = 0.
    if not torch.isfinite(output).all() or bool(((output < 0) | (output > 1+2e-7)).any()):
        raise RuntimeError("Invalid aligned native alias probabilities.")
    return output.clamp(0, 1), observed.flatten()


def native_class_scores(alias_probability, parents, local, observed, config=EncoderObservationConfig()):
    if config != EncoderObservationConfig():
        raise ValueError("Native probability units are fixed.")
    classes = local.shape[-1]
    if (alias_probability.shape != (len(local), len(parents)) or observed.shape != local.shape[:1]
            or observed.dtype != torch.bool or not torch.isfinite(alias_probability).all()
            or bool(((alias_probability < 0) | (alias_probability > 1)).any())
            or set(parents) != set(range(classes)) or not torch.isfinite(local).all()):
        raise ValueError("Invalid native class observation.")
    parent = torch.as_tensor(parents, dtype=torch.long, device=local.device)
    probability = torch.stack([alias_probability[:, parent == c].mean(-1) for c in range(classes)], -1)
    has_information = observed & (probability.sum(-1) > 0)
    scores = config.temperature*probability.clamp_min(torch.finfo(torch.float32).tiny).log()
    scores = torch.where(has_information[:, None], scores, local)
    return scores, has_information


def couple_encoder_scores(local, observation, relation, valid, informative):
    if informative.shape != valid.shape or informative.dtype != torch.bool:
        raise ValueError("Expected an image-only numeric observation support.")
    result, diagnostics = anchored_innovation(local, observation, relation, valid)
    result = torch.where(informative[..., None], result, local)
    return result, diagnostics


class FrozenEncoderGroundingObserver(FrozenGroundingObserver):
    @torch.inference_mode()
    def observe_encoder(self, rgb):
        from PIL import Image
        if rgb.shape != (1, 3, 512, 512) or not bool(torch.isfinite(rgb).all()):
            raise ValueError("Expected unchanged original512 RGB window.")
        image = Image.fromarray((rgb[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8))
        captured = {}

        def capture(module, positional, keywords):
            captured["shapes"] = [tuple(shape) for shape in keywords["spatial_shapes_list"]]
            captured["valid"] = ~keywords["vision_attention_mask"][0]

        hook = self.model.model.encoder.register_forward_pre_hook(capture, with_kwargs=True)
        output = torch.zeros(1024, len(self.aliases), device=self.device)
        common_observed = torch.ones(1024, dtype=torch.bool, device=self.device)
        seen, levels = [], []
        try:
            for chunk in self.chunks:
                captured.clear()
                inputs = self.processor(images=image, text=chunk["caption"], return_tensors="pt",
                                        truncation=False).to(self.device)
                if inputs.input_ids[0].tolist() != chunk["input_ids"]:
                    raise RuntimeError("Processor changed the original exact alias token sequence.")
                measurement = self.model(**inputs)
                logits = measurement.enc_outputs_class[0, :, :len(chunk["input_ids"])]
                valid = captured["valid"] & torch.isfinite(measurement.enc_outputs_coord_logits[0]).all(-1)
                masks = torch.as_tensor(chunk["token_masks"], dtype=torch.bool, device=self.device)
                probabilities, observed = align_encoder_aliases(logits, masks, captured["shapes"], valid)
                indices = torch.as_tensor(chunk["indices"], dtype=torch.long, device=self.device)
                output[:, indices] = probabilities
                common_observed &= observed
                seen.extend(chunk["indices"])
                levels.append({"spatial_shapes": captured["shapes"], "native_positions": len(logits),
                               "valid_positions": int(valid.sum()), "aligned_observed_patches": int(observed.sum())})
        finally:
            hook.remove()
        if sorted(seen) != list(range(len(self.aliases))):
            raise RuntimeError("Incomplete or duplicated all-alias measurement.")
        return output, common_observed, {"levels": levels, "target_masks_loaded": False,
            "decoder_outputs_used_for_classification": False, "decoder_was_computed": True}
