"""STRIDE-OV: structure-interface constrained open-vocabulary propagation.

The module is deliberately training-free. It consumes the two patch streams
already produced by one frozen DINO.text vision forward, builds the ordinary
TLP proposal, and only constrains supported class-contrast flow across DINO
structure interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import CheckpointConfig, TLPConfig
from .contrast_flow import ContrastFlowConfig, ContrastFlowDiagnostics, solve_contrast_flow
from .interface_budget import BudgetDiagnostics, InterfaceBudgetConfig, build_interface_budgets
from .model import DINOTextSegmenter
from .prompts import ClassSpec
from .structure_interfaces import StructureInterfaceConfig, build_structure_interfaces
from .tlp import TLPDiagnostics, TLPState, apply_tlp_state, build_tlp_state


@dataclass(frozen=True)
class StrideOVConfig:
    """All fixed, dataset-independent settings used by STRIDE-OV."""

    tlp: TLPConfig = field(default_factory=TLPConfig)
    structure: StructureInterfaceConfig = field(default_factory=StructureInterfaceConfig)
    budget: InterfaceBudgetConfig = field(default_factory=InterfaceBudgetConfig)
    flow: ContrastFlowConfig = field(default_factory=ContrastFlowConfig)

    def validate(self) -> None:
        self.tlp.validate()
        self.structure.validate()
        self.budget.validate()
        self.flow.validate()


@dataclass(frozen=True)
class StrideOVImageDiagnostics:
    """Per-image diagnostics; batch images never share graph state."""

    atoms: int
    interfaces: int
    eligible_interfaces: int
    tlp: TLPDiagnostics
    budget: BudgetDiagnostics
    flow: ContrastFlowDiagnostics


@dataclass(frozen=True)
class StrideOVResult:
    """Patch-grid scores from the raw, TLP, and constrained stages."""

    logits: Tensor
    initial_logits: Tensor
    proposal_logits: Tensor
    anchors: Tensor | None
    diagnostics: tuple[StrideOVImageDiagnostics, ...]


def stride_ov_from_features(
    text_patch_features: Tensor,
    raw_patch_features: Tensor,
    rgb: Tensor,
    text_features: Tensor,
    config: StrideOVConfig = StrideOVConfig(),
    *,
    valid_mask: Tensor | None = None,
    anchors: Tensor | None = None,
) -> StrideOVResult:
    """Run STRIDE-OV from one shared DINO image encoding.

    ``valid_mask`` is defined on the patch grid. Edges touching padded patches
    are removed from both the TLP proposal and the structure partition.
    """
    config.validate()
    _validate_inputs(text_patch_features, raw_patch_features, rgb, text_features, anchors)
    batch, _, height, width = text_patch_features.shape
    masks = _normalize_valid_mask(valid_mask, batch, height, width, text_patch_features.device)

    visual = F.normalize(text_patch_features.float(), dim=1)
    text = F.normalize(text_features.to(device=visual.device, dtype=torch.float32), dim=-1)
    initial_logits = torch.einsum("bdhw,cd->bchw", visual, text)
    grid_rgb = F.interpolate(
        rgb.to(device=visual.device, dtype=torch.float32),
        size=(height, width),
        mode="area",
    )

    proposal_pieces: list[Tensor] = []
    output_pieces: list[Tensor] = []
    diagnostics: list[StrideOVImageDiagnostics] = []
    for index in range(batch):
        initial = initial_logits[index : index + 1]
        state = build_tlp_state(initial, grid_rgb[index : index + 1], text, config.tlp)
        state = _restrict_state_to_valid_mask(state, masks[index])
        proposal, tlp_diagnostics = apply_tlp_state(initial, state)
        partition = build_structure_interfaces(
            raw_patch_features[index],
            config.structure,
            masks[index],
        )
        budgets, budget_diagnostics = build_interface_budgets(
            initial,
            proposal,
            state,
            partition,
            config.budget,
        )
        output, flow_diagnostics = solve_contrast_flow(
            initial,
            proposal,
            state,
            partition,
            budgets,
            config.flow,
        )
        proposal_pieces.append(proposal)
        output_pieces.append(output)
        diagnostics.append(
            StrideOVImageDiagnostics(
                atoms=partition.atom_count,
                interfaces=len(partition.interfaces),
                eligible_interfaces=sum(item.eligible for item in partition.interfaces),
                tlp=tlp_diagnostics,
                budget=budget_diagnostics,
                flow=flow_diagnostics,
            )
        )

    return StrideOVResult(
        logits=torch.cat(output_pieces, dim=0),
        initial_logits=initial_logits,
        proposal_logits=torch.cat(proposal_pieces, dim=0),
        anchors=anchors,
        diagnostics=tuple(diagnostics),
    )


class StrideOVDINOTextSegmenter:
    """Frozen DINO.text wrapper exposing STRIDE-OV as a callable segmenter."""

    method_name = "STRIDE-OV"
    implementation_note = (
        "Training-free DINO.text OVSS with TLP proposals and supported "
        "class-contrast capacity constraints on raw-DINO structure interfaces."
    )

    def __init__(
        self,
        checkpoints: CheckpointConfig | None = None,
        *,
        backbone: DINOTextSegmenter | None = None,
        config: StrideOVConfig = StrideOVConfig(),
        device: str = "cuda",
        amp: bool = True,
    ) -> None:
        if backbone is None:
            if checkpoints is None:
                raise ValueError("Either checkpoints or an initialized DINOTextSegmenter is required.")
            backbone = DINOTextSegmenter(checkpoints, device=device, amp=amp, use_satellite=False)
        elif checkpoints is not None:
            raise ValueError("Pass checkpoints or backbone, not both.")
        config.validate()
        self.backbone = backbone
        self.config = config

    @property
    def checkpoints(self) -> CheckpointConfig:
        return self.backbone.checkpoints

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    @property
    def text_feature_dim(self) -> int:
        return self.backbone.text_feature_dim

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        return self.backbone.encode_text(classes, batch_size=batch_size)

    def segment(
        self,
        rgb: Tensor,
        classes: Sequence[ClassSpec],
        *,
        valid_mask: Tensor | None = None,
        text_batch_size: int = 64,
    ) -> StrideOVResult:
        text_features = self.encode_text(classes, batch_size=text_batch_size)
        return self.segment_with_text_features(rgb, text_features, valid_mask=valid_mask)

    def segment_with_text_features(
        self,
        rgb: Tensor,
        text_features: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> StrideOVResult:
        patches, raw_patches, anchors = self.backbone.encode_image_with_structure(rgb)
        return stride_ov_from_features(
            patches,
            raw_patches,
            rgb,
            text_features,
            self.config,
            valid_mask=valid_mask,
            anchors=anchors,
        )


def _validate_inputs(
    text_patch_features: Tensor,
    raw_patch_features: Tensor,
    rgb: Tensor,
    text_features: Tensor,
    anchors: Tensor | None,
) -> None:
    if text_patch_features.ndim != 4 or raw_patch_features.ndim != 4:
        raise ValueError("Patch features must be BCHW tensors.")
    if text_patch_features.shape[0] != raw_patch_features.shape[0] or (
        text_patch_features.shape[-2:] != raw_patch_features.shape[-2:]
    ):
        raise ValueError("Text-aligned and raw patch maps must share batch and grid dimensions.")
    if rgb.ndim != 4 or rgb.shape[:2] != (text_patch_features.shape[0], 3):
        raise ValueError("RGB input must have shape [B,3,H,W] with the same batch size.")
    if text_features.ndim != 2 or text_features.shape[0] < 1:
        raise ValueError("text_features must have shape [C,D] with at least one class.")
    if text_features.shape[1] != text_patch_features.shape[1]:
        raise ValueError("Text prototypes and text-aligned patches must share feature dimension.")
    if anchors is not None and (anchors.ndim != 2 or anchors.shape[0] != text_patch_features.shape[0]):
        raise ValueError("anchors must have shape [B,D].")


def _normalize_valid_mask(
    valid_mask: Tensor | None,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    if valid_mask is None:
        return torch.ones((batch, height, width), device=device, dtype=torch.bool)
    if valid_mask.ndim == 2 and batch == 1:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape != (batch, height, width):
        raise ValueError("valid_mask must have shape [B,H,W] on the patch grid.")
    return valid_mask.to(device=device, dtype=torch.bool)


def _restrict_state_to_valid_mask(state: TLPState, valid_mask: Tensor) -> TLPState:
    horizontal_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    vertical_valid = valid_mask[:-1, :] & valid_mask[1:, :]
    return TLPState(
        fidelity=state.fidelity,
        horizontal=state.horizontal * horizontal_valid[None, None],
        vertical=state.vertical * vertical_valid[None, None],
        smoothing_strength=state.smoothing_strength,
        cg_max_iterations=state.cg_max_iterations,
        cg_tolerance=state.cg_tolerance,
    )
