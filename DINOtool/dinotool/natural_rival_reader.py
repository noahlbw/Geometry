"""Query-block execution of the unchanged retained alias-admission equations."""
import torch

from .calibrated_competitive_alias import profiled_logits
from .fine_alias_view import CONFIG, physical_margins
from .matched_contribution_alias import stencil_margins
from .native_alias_noise import hard_pair_observation, signed_potential
from .native_query_alias import native_risk
from .rival_fine_full import PRIMARY
from .stratified_soft_alias import crop_stencil


@torch.inference_mode()
def retained_scores_chunked(local, operator, broad, wide, wide_count, fine, fine_count,
                            coordinates, fine_coordinates, valid, members, canonical,
                            parents, image_size, query_chunk=CONFIG.query_chunk):
    if query_chunk < 1:
        raise ValueError('Positive execution chunk required.')
    pair = torch.empty(len(valid), members.numel(), len(members), device=broad.device)
    potential = torch.empty(len(valid), len(members), dtype=torch.float64, device=broad.device)
    normal_error, gauge_error = 0., 0.
    for start in range(0, len(valid), query_chunk):
        sl = slice(start, start+query_chunk)
        full = torch.zeros(len(valid[sl]), members.numel(), len(members), device=broad.device)
        for crop in wide:
            ids, coeff = crop_stencil(crop, wide_count, coordinates[sl], image_size)
            full += stencil_margins(profiled_logits(crop, members), ids, coeff, CONFIG.beta)
        fine_margin = physical_margins(fine, fine_count, fine_coordinates[sl], members, valid[sl], CONFIG)
        current = native_risk(full, fine_margin, fine_margin, parents, canonical, valid[sl], CONFIG)
        directed = hard_pair_observation(wide, wide_count, coordinates[sl], image_size, members,
                                        current, valid[sl], CONFIG.beta, CONFIG.query_chunk)
        potential[sl], consistency = signed_potential(directed, valid[sl])
        pair[sl] = current
        normal_error = max(normal_error, consistency['normal_equation_max_error'])
        gauge_error = max(gauge_error, consistency['gauge_max_error'])
        del full, fine_margin, current, directed
    baseline = local.double()+operator.double() @ (broad.double()-local.double())
    values = dict(Geometry=local.double(), NoAdmission_Exact=baseline,
                  **{PRIMARY: baseline+operator.double() @ potential})
    if bool((pair[:, canonical] != 0).any()) or not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Nonfinite scores or changed canonical protection.')
    grouped = pair[:, members]
    rivals = ~torch.eye(len(members), device=pair.device, dtype=torch.bool)
    retained = (grouped[valid] == 0).sum(2)[:, rivals]
    stats = dict(retained_count_mean=float(retained.double().mean()) if retained.numel() else 20.,
                 deleted_alias_rival_fraction=float((retained < members.shape[1]).double().mean()) if retained.numel() else 0.,
                 canonical_risk_max=float(pair[:, canonical].abs().max()),
                 mean_absolute_admission_potential=float(potential.abs().mean()),
                 normal_equation_max_error=normal_error, gauge_max_error=gauge_error)
    return values, pair, stats
