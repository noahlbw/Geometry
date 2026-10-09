"""VIP paper Eqs. 3-6 on canonical-replacement multiclass probability maps."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PaperDistillationConfig:
    affinity_power: float = 2.0
    random_walk_steps: int = 2
    activation_threshold: float = 0.4
    text_cosine_threshold: float = 0.7
    alias_chunk: int = 32


def canonical_indices(names, aliases, parents: Tensor) -> Tensor:
    indices = []
    for index, name in enumerate(names):
        found = [i for i, word in enumerate(aliases) if word == name and int(parents[i]) == index]
        if len(found) != 1:
            raise ValueError(f"Exactly one canonical query required: {name}")
        indices.append(found[0])
    return torch.tensor(indices, device=parents.device, dtype=torch.long)


class BackboneAttentionCapture:
    """Observe actual normalized attention inputs without replacing a forward."""

    def __init__(self, model, patch_count=441):
        self.blocks = list(model.blocks)
        self.patch_count = patch_count
        self.handles = []
        self.reset()

    def reset(self):
        self.aggregate = None
        self.count = 0

    def _observe(self, attention, args, kwargs):
        tokens = args[0]
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError("VIP selection observes one crop at a time.")
        rope = kwargs.get("rope", args[2] if len(args) > 2 else None)
        qkv = attention.qkv(tokens)
        batch, length, _ = qkv.shape
        dim = attention.qkv.in_features
        qkv = qkv.reshape(batch, length, 3, attention.num_heads, dim // attention.num_heads)
        query, key, _ = (value.transpose(1, 2) for value in qkv.unbind(2))
        if rope is not None:
            query, key = attention.apply_rope(query, key, rope)
        prefix = length - self.patch_count
        if prefix < 0:
            raise ValueError("Attention has fewer tokens than the VIP patch grid.")
        # Normalize over ALL tokens before extracting the patch submatrix.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            affinity = ((query.float() @ key.float().transpose(-1, -2)) * attention.scale).softmax(-1)
            affinity = affinity[..., prefix:, prefix:].mean(1)[0]
        if not torch.isfinite(affinity).all():
            raise FloatingPointError("Nonfinite original backbone attention.")
        self.aggregate = affinity if self.aggregate is None else self.aggregate + affinity
        self.count += 1

    def __enter__(self):
        try:
            for block in self.blocks:
                self.handles.append(block.attn.register_forward_pre_hook(self._observe, with_kwargs=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def mean(self):
        if self.count != len(self.blocks) or self.aggregate is None:
            raise RuntimeError(f"Captured {self.count}/{len(self.blocks)} backbone attention layers.")
        return self.aggregate / self.count


def random_walk(attention: Tensor, valid: Tensor, config: PaperDistillationConfig) -> Tensor:
    if attention.shape != (valid.numel(), valid.numel()) or not valid.any():
        raise ValueError("A nonempty valid square patch affinity is required.")
    relation = attention.float().clamp_min(0).pow(config.affinity_power)
    relation = relation.masked_fill(~valid[None], 0)
    sums = relation.sum(-1, keepdim=True)
    if (sums[valid] <= 0).any() or not torch.isfinite(relation).all():
        raise FloatingPointError("Invalid valid-key random-walk normalization.")
    relation = relation / sums.clamp_min(torch.finfo(torch.float32).tiny)
    walk = relation
    for _ in range(config.random_walk_steps - 1):
        walk = walk @ relation
    return walk


def replacement_probabilities(alias_logits: Tensor, parents: Tensor, canonical: Tensor,
                              indices: Tensor) -> Tensor:
    """[patch, alias] -> [candidate, patch, class], replacing only the parent."""
    baseline = alias_logits[:, canonical]
    logits = baseline[None].expand(indices.numel(), -1, -1).clone()
    logits.scatter_(2, parents[indices, None, None].expand(-1, logits.shape[1], 1),
                    alias_logits[:, indices].T[..., None])
    return logits.softmax(-1)


class PaperAliasAccumulator:
    def __init__(self, alias_count, device):
        self.vg_sum = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.sc_sum = torch.zeros_like(self.vg_sum)
        self.observed_images = torch.zeros(alias_count, device=device, dtype=torch.long)
        self.image = torch.zeros((4, alias_count), device=device, dtype=torch.float64)

    def update_crop(self, logits, parents, canonical, attention, valid, config):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite original VIP alias logits.")
        walk = random_walk(attention, valid, config)
        for start in range(0, logits.shape[1], config.alias_chunk):
            indices = torch.arange(start, min(start + config.alias_chunk, logits.shape[1]), device=logits.device)
            probability = replacement_probabilities(logits.float(), parents, canonical, indices)
            own = probability.gather(2, parents[indices, None, None].expand(-1, logits.shape[0], 1))[..., 0].T
            propagated = walk @ own
            high = (own >= config.activation_threshold) & valid[:, None]
            entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1).T
            self.image[0, indices] += (own * propagated * high).sum(0).double()
            self.image[1, indices] += ((own + propagated - own * propagated) * high).sum(0).double()
            self.image[2, indices] += (entropy * high).sum(0).double()
            self.image[3, indices] += high.sum(0).double()

    def finalize_image(self):
        present = self.image[3] > 0
        self.vg_sum[present] += self.image[0, present] / self.image[1, present]
        self.sc_sum[present] += self.image[2, present] / self.image[3, present]
        self.observed_images += present.long()
        self.image.zero_()

    def state(self):
        if self.image.any():
            raise RuntimeError("Finalize the image before exporting global scores.")
        return {name: getattr(self, name).cpu().tolist() for name in ("vg_sum", "sc_sum", "observed_images")}


def select_aliases(state, queries, config):
    parents = queries.parents.cpu()
    canonical = canonical_indices(queries.class_names, queries.aliases, parents)
    features = F.normalize(queries.features.float().mean(1), dim=-1).cpu()
    cosine = (features * features[canonical[parents]]).sum(-1)
    return select_from_scores(state, queries.class_names, queries.aliases, parents, cosine, config)


def select_from_scores(state, names, aliases, parents, cosine, config):
    parents, cosine = torch.as_tensor(parents), torch.as_tensor(cosine)
    canonical = canonical_indices(names, aliases, parents)
    count = torch.tensor(state["observed_images"])
    vg = torch.tensor(state["vg_sum"], dtype=torch.float64) / count.clamp_min(1)
    sc = torch.tensor(state["sc_sum"], dtype=torch.float64) / count.clamp_min(1)
    anchor = canonical[parents]
    keep = (count > 0) & (count[anchor] > 0) & (cosine >= config.text_cosine_threshold)
    keep &= (vg > vg[anchor]) & (sc < sc[anchor])
    keep[canonical] = True
    entries = []
    for index, word in enumerate(aliases):
        reason = ("canonical" if index == int(anchor[index]) else "text_prefilter" if cosine[index] < config.text_cosine_threshold
                  else "no_high_activation_support" if count[index] == 0 or count[anchor[index]] == 0
                  else "retained" if keep[index] else "VG_or_SC_not_better_than_canonical")
        entries.append(dict(index=index, alias=word, parent=int(parents[index]), canonical=index == int(anchor[index]),
                            cosine_to_canonical=float(cosine[index]), vg=float(vg[index]) if count[index] else None,
                            sc=float(sc[index]) if count[index] else None, observed_images=int(count[index]),
                            retained=bool(keep[index]), reason=reason))
    groups = [[word for index, word in enumerate(aliases) if parents[index] == c and keep[index]]
              for c in range(len(names))]
    return dict(keep=keep.tolist(), selected_alias_groups=groups, selected_counts=list(map(len, groups)), aliases=entries)


@torch.inference_mode()
def source_logits(patch, queries, settings):
    """Preserve the pinned adapter's autocast, template mean and logit scaling."""
    with torch.autocast(device_type=patch.device.type, dtype=torch.bfloat16, enabled=patch.device.type == "cuda"):
        similarities = torch.einsum("bnd,mtd->bnmt", patch, queries.features.float()).mean(-1)[0]
        patch_mean = F.normalize(patch.mean(1), dim=-1)
        text_mean = F.normalize(queries.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0] / settings.tem).float()
        logits = (similarities * settings.logit_scale).T.reshape(-1, 21, 21).float()
    if not torch.isfinite(logits).all() or not torch.isfinite(salience).all():
        raise FloatingPointError("Nonfinite pinned VIP features/logits; no fallback substituted.")
    return logits, salience


def aggregate_logits(logits, salience, parents, class_count, settings, keep=None):
    if keep is None:
        keep = torch.ones_like(parents, dtype=torch.bool)
    scores = []
    for c in range(class_count):
        members = (parents == c) & keep
        if not members.any():
            raise ValueError("Cannot remove every query from a class.")
        weights = salience[members].softmax(0)
        scaled = logits[members] * (weights / weights.mean())[:, None, None]
        scores.append(torch.logsumexp(settings.tau * scaled, dim=0) / settings.tau)
    return F.interpolate(torch.stack(scores)[None], size=(336, 336), mode="bilinear", align_corners=False)[0].float()


def image_crops(rgb, settings):
    original = tuple(rgb.shape[-2:])
    ratio = settings.resize_long_edge / max(original)
    height, width = [int(size * ratio + .5) for size in original]
    array = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255)
    crops = []
    for iy in range(max(height - settings.slide_crop + settings.slide_stride - 1, 0) // settings.slide_stride + 1):
        for ix in range(max(width - settings.slide_crop + settings.slide_stride - 1, 0) // settings.slide_stride + 1):
            bottom = min(iy * settings.slide_stride + settings.slide_crop, height)
            right = min(ix * settings.slide_stride + settings.slide_crop, width)
            top, left = max(bottom - settings.slide_crop, 0), max(right - settings.slide_crop, 0)
            crop = image[:, top:bottom, left:right]
            crop = F.pad(crop, (0, settings.slide_crop - crop.shape[-1], 0, settings.slide_crop - crop.shape[-2]))
            crops.append((crop, (top, left, bottom, right)))
    return original, (height, width), crops


def valid_patches(bounds, device):
    top, left, bottom, right = bounds
    centers = torch.arange(21, device=device) * 16 + 8
    return ((centers[:, None] < bottom - top) & (centers[None, :] < right - left)).flatten()


def cached_prediction(arrays, key, parents, names, settings, keep, device):
    height, width = map(int, arrays["resized_shape"])
    original = tuple(map(int, arrays["original_shape"]))
    count = torch.zeros((1, height, width), device=device)
    total = torch.zeros((len(names), height, width), device=device)
    for index, (top, left, bottom, right) in enumerate(arrays["bounds"]):
        logits = torch.from_numpy(arrays[key + "_logits"][index]).to(device)
        salience = torch.from_numpy(arrays[key + "_salience"][index]).to(device)
        window = aggregate_logits(logits, salience, parents, len(names), settings, keep)
        total[:, top:bottom, left:right] += window[:, :bottom - top, :right - left]
        count[:, top:bottom, left:right] += 1
    logits = F.interpolate((total / count)[None], size=original, mode="bilinear", align_corners=False)[0]
    probability = logits.softmax(0)
    prediction = probability.argmax(0)
    if settings.background:
        prediction = prediction.masked_fill(probability.amax(0) < settings.prob_thd, settings.bg_idx)
    return prediction.to(torch.uint8).cpu().numpy()
