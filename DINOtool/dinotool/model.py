from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import CheckpointConfig
from .parallel_readout import ParallelReadoutConfig, parallel_vision_head_readouts
from .prompts import ClassSpec, REMOTE_SENSING_TEMPLATES, clean_phrase, prompts_for_class


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SATELLITE_MEAN = (0.430, 0.411, 0.296)
SATELLITE_STD = (0.213, 0.156, 0.143)


class DINOTextSegmenter:
    """Official DINOv3 dino.txt model with optional SAT-493M structure features."""

    patch_size = 16

    def __init__(
        self,
        checkpoints: CheckpointConfig,
        device: str = "cuda",
        amp: bool = True,
        use_satellite: bool = False,
    ) -> None:
        missing = [path for path in checkpoints.required(use_satellite) if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Required DINOv3 files are missing:\n{formatted}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        self.checkpoints = checkpoints
        self.device = torch.device(device)
        self.use_amp = amp and self.device.type == "cuda"
        _configure_local_hub_cache(checkpoints)
        self.model, tokenizer = torch.hub.load(
            str(checkpoints.dinov3_repo),
            "dinov3_vitl16_dinotxt_tet1280d20h24l",
            source="local",
            weights=str(checkpoints.dinotxt_weights),
            backbone_weights=str(checkpoints.lvd_weights),
            bpe_path_or_url=str(checkpoints.bpe_path),
        )
        self.model.to(self.device).eval().requires_grad_(False)
        self.tokenize = tokenizer.tokenize
        self._visual_tuning_indices: tuple[int, ...] = ()
        self.satellite = None
        if use_satellite:
            self.satellite = torch.hub.load(
                str(checkpoints.dinov3_repo),
                "dinov3_vitl16",
                source="local",
                weights=str(checkpoints.sat_weights),
            ).to(self.device).eval().requires_grad_(False)

        self._imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._imagenet_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._satellite_mean = torch.tensor(SATELLITE_MEAN, device=self.device).view(1, 3, 1, 1)
        self._satellite_std = torch.tensor(SATELLITE_STD, device=self.device).view(1, 3, 1, 1)

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        if not classes:
            raise ValueError("At least one class is required.")
        if batch_size < 1:
            raise ValueError("Text batch_size must be positive.")

        # Flatten all prompt variants before tokenization.  The previous
        # implementation launched one tokenizer/model loop per class, which
        # made dynamic-vocabulary requests pay a Python and kernel-launch cost
        # for every concept.  A single stream keeps the same per-class mean
        # while allowing the text tower to process all concepts in batches.
        prompts_by_class = [prompts_for_class(spec) for spec in classes]
        flat_prompts = [prompt for prompts in prompts_by_class for prompt in prompts]
        prompt_features: list[Tensor] = []
        for start in range(0, len(flat_prompts), batch_size):
            tokens = self.tokenize(flat_prompts[start : start + batch_size]).to(self.device, non_blocking=True)
            with torch.inference_mode(), self._autocast():
                encoded = self.model.encode_text(tokens, normalize=False)
            patch_aligned = encoded[:, encoded.shape[-1] // 2 :].float()
            prompt_features.append(F.normalize(patch_aligned, dim=-1))

        flat_features = torch.cat(prompt_features, dim=0)
        class_features: list[Tensor] = []
        offset = 0
        for prompts in prompts_by_class:
            count = len(prompts)
            averaged = flat_features[offset : offset + count].mean(dim=0)
            class_features.append(F.normalize(averaged, dim=0))
            offset += count
        return torch.stack(class_features, dim=0)

    def encode_text_aliases(
        self,
        aliases_by_class: Sequence[Sequence[str]],
        *,
        templates: Sequence[str] = REMOTE_SENSING_TEMPLATES,
        batch_size: int = 64,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode every alias independently while averaging prompt templates.

        The first alias of each class is canonical.  Returning the parent map
        keeps image-conditioned alias filtering separate from semantic class
        aggregation, as required by VIP-style vocabularies.
        """
        if not aliases_by_class or any(not aliases for aliases in aliases_by_class):
            raise ValueError("Every semantic class must provide at least one alias.")
        if not templates or batch_size < 1:
            raise ValueError("Text templates and batch_size must be non-empty/positive.")
        cleaned = [tuple(dict.fromkeys(clean_phrase(alias) for alias in aliases))
                   for aliases in aliases_by_class]
        flat_aliases = [alias for aliases in cleaned for alias in aliases]
        prompts = [template.format(label=alias) for alias in flat_aliases for template in templates]
        encoded_prompts: list[Tensor] = []
        for start in range(0, len(prompts), batch_size):
            tokens = self.tokenize(prompts[start:start + batch_size]).to(self.device, non_blocking=True)
            with torch.inference_mode(), self._autocast():
                encoded = self.model.encode_text(tokens, normalize=False)
            patch_aligned = encoded[:, encoded.shape[-1] // 2:].float()
            encoded_prompts.append(F.normalize(patch_aligned, dim=-1))
        prompt_features = torch.cat(encoded_prompts, dim=0)
        per_alias = prompt_features.reshape(len(flat_aliases), len(templates), -1).mean(dim=1)
        per_alias = F.normalize(per_alias, dim=-1)
        parents = torch.tensor(
            [class_index for class_index, aliases in enumerate(cleaned) for _ in aliases],
            device=self.device,
            dtype=torch.long,
        )
        canonical = torch.tensor(
            [alias_index == 0 for aliases in cleaned for alias_index in range(len(aliases))],
            device=self.device,
            dtype=torch.bool,
        )
        return per_alias, parents, canonical

    def encode_image(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        """Return dino.txt patch features and the normalized vision-head CLS token."""
        return self._encode_image(rgb, inference=True)

    def encode_image_with_structure(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return text-aligned patches, raw backbone patches, and the CLS anchor.

        DINO.text already computes both patch streams in one vision forward.
        Exposing the raw stream avoids a second backbone pass for structure-aware
        inference while keeping ``encode_image`` backward compatible.
        """
        return self._encode_image_with_structure(rgb, inference=True)

    def encode_image_parallel_readouts(
        self,
        rgb: Tensor,
        configs: Mapping[str, ParallelReadoutConfig],
    ) -> dict[str, Tensor]:
        """Return multiple text-aligned patch maps from one shared backbone pass."""
        self._validate_rgb(rgb)
        if not configs:
            raise ValueError("At least one named ParallelReadoutConfig is required.")
        normalized = (rgb.to(self.device, non_blocking=True) - self._imagenet_mean) / self._imagenet_std
        with torch.inference_mode(), self._autocast():
            visual = self.model.visual_model
            class_token, backbone_patches, register_tokens = visual.get_backbone_features(normalized)
            tokens = torch.cat((class_token.unsqueeze(1), register_tokens, backbone_patches), dim=1)
            outputs = parallel_vision_head_readouts(
                visual.head,
                tokens,
                backbone_patches,
                rgb.shape[-2] // self.patch_size,
                rgb.shape[-1] // self.patch_size,
                configs,
            )
        prefix = 1 + register_tokens.shape[1]
        height = rgb.shape[-2] // self.patch_size
        width = rgb.shape[-1] // self.patch_size
        return {
            name: F.normalize(
                encoded[:, prefix:].float().transpose(1, 2).reshape(rgb.shape[0], -1, height, width),
                dim=1,
            )
            for name, encoded in outputs.items()
        }

    def encode_image_for_adapter(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        """Return frozen dino.txt features that can be consumed by a trainable adapter.

        ``torch.inference_mode`` tensors cannot be saved by convolution backward
        passes.  The backbone remains frozen, but this path uses ``no_grad`` so
        a downstream lightweight decoder can be optimized safely.
        """
        return self._encode_image(rgb, inference=False)

    def enable_visual_finetuning(self, last_blocks: int) -> None:
        """Expose only the final DINOv3 vision blocks for external-data adaptation.

        The text tower and the DINO text-alignment head remain frozen.  This is
        deliberately narrow: it lets the backbone adapt to RGB remote-sensing
        texture without turning an open-vocabulary decoder into a fixed-class
        segmenter.
        """
        blocks = self.model.visual_model.backbone.blocks
        if last_blocks < 0 or last_blocks > len(blocks):
            raise ValueError(f"last_blocks must be in [0, {len(blocks)}].")
        self._visual_tuning_indices = tuple(range(len(blocks) - last_blocks, len(blocks)))
        for index in self._visual_tuning_indices:
            for parameter in blocks[index].parameters():
                parameter.requires_grad_(True)
        self.set_visual_tuning_training(True)

    def set_visual_tuning_training(self, training: bool) -> None:
        """Keep frozen DINO modules in eval mode while toggling tuned blocks."""
        self.model.eval()
        blocks = self.model.visual_model.backbone.blocks
        for index in self._visual_tuning_indices:
            blocks[index].train(training)

    def visual_tuning_parameters(self) -> list[Tensor]:
        blocks = self.model.visual_model.backbone.blocks
        return [
            parameter
            for index in self._visual_tuning_indices
            for parameter in blocks[index].parameters()
            if parameter.requires_grad
        ]

    def visual_tuning_state_dict(self) -> dict[str, dict[str, Tensor]]:
        """Return only externally tuned visual blocks, not the full foundation model."""
        blocks = self.model.visual_model.backbone.blocks
        return {
            str(index): {name: value.detach().cpu() for name, value in blocks[index].state_dict().items()}
            for index in self._visual_tuning_indices
        }

    def load_visual_tuning_state_dict(self, state: dict[str, dict[str, Tensor]] | None) -> None:
        if state is None:
            return
        if not isinstance(state, dict):
            raise ValueError("Visual tuning state must be a mapping of DINO block indices to state dictionaries.")
        blocks = self.model.visual_model.backbone.blocks
        indices: list[int] = []
        for index_text, block_state in state.items():
            try:
                index = int(index_text)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid DINO visual block index: {index_text!r}") from error
            if index < 0 or index >= len(blocks) or not isinstance(block_state, dict):
                raise ValueError(f"Invalid visual tuning state for DINO block {index_text!r}.")
            blocks[index].load_state_dict(block_state, strict=True)
            indices.append(index)
        self._visual_tuning_indices = tuple(sorted(indices))
        self.set_visual_tuning_training(False)

    @property
    def visual_tuning_manifest(self) -> dict[str, object]:
        return {
            "tuned_backbone_blocks": list(self._visual_tuning_indices),
            "text_tower_frozen": True,
            "vision_head_frozen": True,
        }

    def encode_image_for_cost_aggregation(
        self,
        rgb: Tensor,
        *,
        train_visual_backbone: bool,
    ) -> tuple[Tensor, Tensor]:
        """Encode frozen features or propagate gradients into configured final vision blocks."""
        if train_visual_backbone and not self._visual_tuning_indices:
            raise RuntimeError("No DINO visual blocks are enabled for fine-tuning.")
        return self._encode_image(
            rgb,
            inference=False,
            allow_visual_grad=train_visual_backbone,
        )

    def encode_image_multiscale_for_mask(
        self,
        rgb: Tensor,
        *,
        layers: Sequence[int] = (5, 11, 17, 23),
        train_visual_backbone: bool = False,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        """Return text-aligned DINO patch maps from several backbone depths.

        The official dino.txt vision head is applied independently to each
        intermediate backbone output, so every returned map remains in the
        same 1024-D space as the frozen text prototypes. This gives a dense
        decoder both mid-level structure and final semantic features without
        replacing the dino.txt alignment head.
        """
        if train_visual_backbone and not self._visual_tuning_indices:
            raise RuntimeError("No DINO visual blocks are enabled for fine-tuning.")
        return self._encode_image_multiscale(
            rgb,
            layers=layers,
            inference=False,
            allow_visual_grad=train_visual_backbone,
        )

    def _encode_image(
        self,
        rgb: Tensor,
        *,
        inference: bool,
        allow_visual_grad: bool = False,
    ) -> tuple[Tensor, Tensor]:
        patches, _, anchor = self._encode_image_with_structure(
            rgb,
            inference=inference,
            allow_visual_grad=allow_visual_grad,
        )
        return patches, anchor

    def _encode_image_with_structure(
        self,
        rgb: Tensor,
        *,
        inference: bool,
        allow_visual_grad: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_rgb(rgb)
        normalized = (rgb.to(self.device, non_blocking=True) - self._imagenet_mean) / self._imagenet_std
        gradient_context = nullcontext() if allow_visual_grad else (torch.inference_mode() if inference else torch.no_grad())
        with gradient_context, self._autocast():
            image_features, patch_tokens, backbone_patch_tokens = self.model.encode_image_with_patch_tokens(
                normalized,
                normalize=False,
            )
        grid_h = rgb.shape[-2] // self.patch_size
        grid_w = rgb.shape[-1] // self.patch_size
        patches = patch_tokens.float().transpose(1, 2).reshape(rgb.shape[0], -1, grid_h, grid_w)
        patches = F.normalize(patches, dim=1)
        structure = backbone_patch_tokens.float().transpose(1, 2).reshape(rgb.shape[0], -1, grid_h, grid_w)
        structure = F.normalize(structure, dim=1)
        # DINO.txt concatenates its vision-head CLS and pooled-patch features.
        cls_features = image_features[:, : patch_tokens.shape[-1]].float()
        anchor = F.normalize(cls_features, dim=-1)
        return patches, structure, anchor

    def _encode_image_multiscale(
        self,
        rgb: Tensor,
        *,
        layers: Sequence[int],
        inference: bool,
        allow_visual_grad: bool = False,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        self._validate_rgb(rgb)
        selected_layers = tuple(int(layer) for layer in layers)
        if not selected_layers:
            raise ValueError("At least one DINO intermediate layer is required.")
        block_count = len(self.model.visual_model.backbone.blocks)
        if len(set(selected_layers)) != len(selected_layers) or any(
            layer < 0 or layer >= block_count for layer in selected_layers
        ):
            raise ValueError(f"DINO intermediate layers must be unique indices in [0, {block_count}).")

        normalized = (rgb.to(self.device, non_blocking=True) - self._imagenet_mean) / self._imagenet_std
        gradient_context = nullcontext() if allow_visual_grad else (torch.inference_mode() if inference else torch.no_grad())
        with gradient_context, self._autocast():
            outputs = self.model.visual_model.backbone.get_intermediate_layers(
                normalized,
                n=selected_layers,
                reshape=False,
                return_class_token=True,
                return_extra_tokens=True,
            )
            projected_layers: list[Tensor] = []
            anchor: Tensor | None = None
            for patch_tokens, class_token, extra_tokens in outputs:
                tokens = torch.cat((class_token.unsqueeze(1), extra_tokens, patch_tokens), dim=1)
                projected = self.model.visual_model.head(tokens)
                projected_patch = projected[:, 1 + extra_tokens.shape[1] :]
                grid_h = rgb.shape[-2] // self.patch_size
                grid_w = rgb.shape[-1] // self.patch_size
                patch_map = projected_patch.float().transpose(1, 2).reshape(rgb.shape[0], -1, grid_h, grid_w)
                projected_layers.append(F.normalize(patch_map, dim=1))
                anchor = projected[:, 0].float()
        if anchor is None:
            raise RuntimeError("DINO did not return an intermediate class token.")
        return tuple(projected_layers), F.normalize(anchor, dim=-1)

    def encode_satellite_structure(self, rgb: Tensor) -> Tensor:
        return self._encode_satellite_structure(rgb, inference=True)

    def encode_satellite_structure_for_adapter(self, rgb: Tensor) -> Tensor:
        """Return frozen SAT features suitable as inputs to a trainable adapter."""
        return self._encode_satellite_structure(rgb, inference=False)

    def _encode_satellite_structure(self, rgb: Tensor, *, inference: bool) -> Tensor:
        if self.satellite is None:
            raise RuntimeError("SAT-493M structure encoder was not loaded.")
        self._validate_rgb(rgb)
        normalized = (rgb.to(self.device, non_blocking=True) - self._satellite_mean) / self._satellite_std
        gradient_context = torch.inference_mode() if inference else torch.no_grad()
        with gradient_context, self._autocast():
            features = self.satellite.get_intermediate_layers(normalized, n=1)[0]
        if isinstance(features, tuple):
            features = features[0]
        grid_h = rgb.shape[-2] // self.patch_size
        grid_w = rgb.shape[-1] // self.patch_size
        structure = features.float().transpose(1, 2).reshape(rgb.shape[0], -1, grid_h, grid_w)
        return F.normalize(structure, dim=1)

    @property
    def text_feature_dim(self) -> int:
        """Dimension of dino.txt patch features and text prototypes."""
        return int(self.model.visual_model.backbone.embed_dim)

    @staticmethod
    def similarity_logits(patch_features: Tensor, text_features: Tensor) -> Tensor:
        return torch.einsum("bdhw,cd->bchw", patch_features, text_features)

    def _validate_rgb(self, rgb: Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB input must have shape [B, 3, H, W].")
        if rgb.shape[-2] % self.patch_size or rgb.shape[-1] % self.patch_size:
            raise ValueError(f"RGB dimensions must be divisible by patch size {self.patch_size}.")

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def checkpoint_manifest(checkpoints: CheckpointConfig) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for name, path in (
        ("dinotxt", checkpoints.dinotxt_weights),
        ("lvd1689m", checkpoints.lvd_weights),
        ("sat493m", checkpoints.sat_weights),
        ("bpe", checkpoints.bpe_path),
    ):
        if path.exists():
            stat = path.stat()
            manifest[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return manifest


def _configure_local_hub_cache(checkpoints: CheckpointConfig) -> None:
    """Keep torch.hub's file-URL cache on the data volume without duplicating weights."""
    hub_dir = checkpoints.checkpoint_dir / ".torch-hub"
    checkpoint_cache = hub_dir / "checkpoints"
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    for source in (checkpoints.dinotxt_weights, checkpoints.lvd_weights, checkpoints.sat_weights):
        if not source.exists():
            continue
        cached = checkpoint_cache / source.name
        if cached.exists() or cached.is_symlink():
            continue
        try:
            os.symlink(source, cached)
        except OSError:
            # Some filesystems disallow symlinks. torch.hub will fall back to a data-volume copy.
            pass
