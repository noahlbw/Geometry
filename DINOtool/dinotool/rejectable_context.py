"""Training-free, proposal-specific acceptance of bounded context evidence."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .contextual_phrase_readout import ContextualPhraseSegmenter, masked_geometry, sample_overview_aliases
from .grounded_context import context_box, normalized_logmeanexp
from .hypothesis_readout import uniform_subset_scores
from .tcpr import _tokens_to_map


METHODS = (
    "G_all20_uniform", "BoundedUnion", "ConfidenceGate",
    "GAR_ClosedForm", "GAR_GCR",
)


@dataclass(frozen=True)
class RejectableContextConfig:
    context_extent: int = 1024
    visual_radius: int = 2
    graph_strength: float = 1.0
    solver_steps: int = 16
    posterior_temperature: float = 0.07

    def validate(self) -> None:
        if self.context_extent < 512 or self.context_extent % 16:
            raise ValueError("Context extent must be patch aligned and at least 512.")
        if self.visual_radius < 1 or self.solver_steps < 1:
            raise ValueError("Visual radius and solver steps must be positive.")
        if self.graph_strength < 0 or self.posterior_temperature <= 0:
            raise ValueError("Graph strength and posterior temperature are invalid.")


def local_visual_graph(geometry: Tensor, valid: Tensor, grid_height: int,
                       grid_width: int, radius: int) -> Tensor:
    """Symmetric, self-excluding DINO affinity with bounded spatial support."""
    batch, patches, keys = geometry.shape
    if patches != keys or patches != grid_height * grid_width or valid.shape != (batch, patches):
        raise ValueError("Geometry and valid mask do not match the patch grid.")
    if radius < 1:
        raise ValueError("Visual radius must be positive.")
    index = torch.arange(patches, device=geometry.device)
    row, col = index // grid_width, index % grid_width
    nearby = ((row[:, None] - row[None, :]).abs() <= radius)
    nearby &= ((col[:, None] - col[None, :]).abs() <= radius)
    nearby &= index[:, None] != index[None, :]
    allowed = nearby[None] & valid[:, :, None] & valid[:, None, :]
    graph = (0.5 * (geometry + geometry.transpose(-1, -2))).clamp_min(0)
    graph = graph * allowed
    return graph / graph.sum(-1).amax(-1)[:, None, None].clamp_min(1e-8)


def visual_witness(local_posterior: Tensor, graph: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Leave-one-out neighborhood consensus and its class-vector variance."""
    degree = graph.sum(-1, keepdim=True)
    mean = (graph @ local_posterior) / degree.clamp_min(1e-8)
    witness = torch.where(degree > 0, mean, local_posterior)
    second = (graph @ local_posterior.square()) / degree.clamp_min(1e-8)
    disagreement = (second - mean.square()).clamp_min(0).sum(-1)
    witness = torch.where(valid[..., None], witness, local_posterior)
    disagreement = torch.where(valid, disagreement, torch.zeros_like(disagreement))
    return witness, disagreement


def proposal_acceptance(local_posterior: Tensor, bounded_posterior: Tensor,
                        witness: Tensor, disagreement: Tensor,
                        valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Project an independently witnessed class-vector revision onto [0, 1]."""
    revision = bounded_posterior - local_posterior
    support = (revision * (witness - local_posterior)).sum(-1)
    denominator = revision.square().sum(-1) + disagreement + 1e-8
    acceptance = (support / denominator).clamp(0, 1)
    acceptance = torch.where(valid & (support > 0), acceptance, torch.zeros_like(acceptance))
    return acceptance, support, revision


def reconstruct_acceptance(local_posterior: Tensor, witness: Tensor,
                           revision: Tensor, disagreement: Tensor, graph: Tensor,
                           valid: Tensor, initial: Tensor, support: Tensor, *,
                           strength: float, steps: int) -> Tensor:
    """Projected descent for a convex graph energy on accepted revisions."""
    if strength < 0 or steps < 1:
        raise ValueError("Invalid graph reconstruction configuration.")
    magnitude = revision.square().sum(-1)
    degree = graph.sum(-1)
    alignment = (revision @ revision.transpose(-1, -2)).abs()
    diagonal = magnitude + disagreement + strength * degree * magnitude
    lipschitz = (diagonal + strength * (graph * alignment).sum(-1)).amax().clamp_min(1e-8)
    eligible = valid & (support > 0)
    acceptance = initial
    for _ in range(steps):
        neighboring_revision = graph @ (acceptance[..., None] * revision)
        gradient = -support + diagonal * acceptance
        gradient -= strength * (revision * neighboring_revision).sum(-1)
        next_acceptance = (acceptance - gradient / lipschitz).clamp(0, 1)
        acceptance = torch.where(eligible, next_acceptance, torch.zeros_like(next_acceptance))
    return acceptance


def accepted_logits(local_posterior: Tensor, revision: Tensor,
                    acceptance: Tensor, temperature: float) -> Tensor:
    posterior = local_posterior + acceptance[..., None] * revision
    return temperature * posterior.clamp_min(1e-12).log()


class RejectableContextSegmenter(ContextualPhraseSegmenter):
    """A bounded-context proposal accepted by local Geometry evidence."""

    needs_whole = False

    def __init__(self, backbone, config, tcpr_config,
                 rejectable_config=RejectableContextConfig()):
        super().__init__(backbone, config, tcpr_config)
        rejectable_config.validate()
        self.rejectable_config = rejectable_config

    @torch.inference_mode()
    def read_grounded(self, prepared, whole_map: Tensor | None, bounded_map: Tensor,
                      bank, *, height: int, width: int, top: int, left: int,
                      box: tuple[int, int, int], valid_mask: Tensor):
        del whole_map
        valid = valid_mask.reshape(1, -1).bool()
        text = F.normalize(bank.features.float(), dim=-1)
        local = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
        by, bx, side = box
        bounded = sample_overview_aliases(
            bounded_map, image_height=side, image_width=side,
            tile_top=top - by, tile_left=left - bx,
            grid_height=prepared.grid_height, grid_width=prepared.grid_width,
            patch_size=self.patch_size,
        )
        geometry = masked_geometry(prepared.geometry_patch_conditional.float(), valid, 1e-6)
        subset = torch.ones(text.shape[0], dtype=torch.bool, device=text.device)
        tau = self.config.alias_temperature

        def class_readout(aliases: Tensor) -> Tensor:
            return uniform_subset_scores(aliases, bank.parent_indices,
                                         bank.class_count, subset, tau)

        local_logits = class_readout(local)
        bounded_logits = class_readout(normalized_logmeanexp(
            torch.stack((local, geometry @ bounded), dim=-1), tau))
        temperature = self.rejectable_config.posterior_temperature
        local_p = (local_logits / temperature).softmax(-1)
        bounded_p = (bounded_logits / temperature).softmax(-1)
        graph = local_visual_graph(
            geometry, valid, prepared.grid_height, prepared.grid_width,
            self.rejectable_config.visual_radius,
        )
        witness, disagreement = visual_witness(local_p, graph, valid)
        initial, support, revision = proposal_acceptance(
            local_p, bounded_p, witness, disagreement, valid,
        )
        accepted = reconstruct_acceptance(
            local_p, witness, revision, disagreement, graph, valid, initial,
            support, strength=self.rejectable_config.graph_strength,
            steps=self.rejectable_config.solver_steps,
        )
        confidence_gate = ((bounded_p.amax(-1) > local_p.amax(-1)) & valid).to(local_p.dtype)
        fields = {
            "G_all20_uniform": local_logits,
            "BoundedUnion": bounded_logits,
            "ConfidenceGate": accepted_logits(local_p, revision, confidence_gate, temperature),
            "GAR_ClosedForm": accepted_logits(local_p, revision, initial, temperature),
            "GAR_GCR": accepted_logits(local_p, revision, accepted, temperature),
        }
        shape = prepared.grid_height, prepared.grid_width
        output = {
            name: _tokens_to_map(torch.where(valid[..., None], value, local_logits), *shape)
            for name, value in fields.items()
        }
        active = valid.sum().clamp_min(1)
        diagnostics = {
            "proposal_changed_fraction": float(((local_p.argmax(-1) != bounded_p.argmax(-1)) & valid).sum() / active),
            "gar_eligible_fraction": float(((support > 0) & valid).sum() / active),
            "gar_mean_acceptance": float((initial * valid).sum() / active),
            "gcr_mean_acceptance": float((accepted * valid).sum() / active),
            "mean_witness_disagreement": float((disagreement * valid).sum() / active),
            "mean_revision_norm": float(((revision.square().sum(-1).sqrt()) * valid).sum() / active),
        }
        return output, diagnostics
