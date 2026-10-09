"""The retained RivalFineHard rule with separate global and tile coordinates."""
import torch

from .calibrated_competitive_alias import profiled_logits
from .fine_alias_view import CONFIG, physical_margins
from .matched_contribution_alias import stencil_margins
from .native_alias_noise import hard_pair_observation, signed_potential
from .native_query_alias import native_risk
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-physical8-rival-admission-coupled-v1-20261003'
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact')
PRIMARY = METHODS[-1]


def tile_reference_coordinates(coordinates, top, left):
    return coordinates - coordinates.new_tensor((top, left))


@torch.inference_mode()
def retained_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                    coordinates, fine_coordinates, valid, members, canonical,
                    parents, image_size):
    full = torch.zeros(len(valid), members.numel(), len(members), device=broad.device)
    for crop in wide:
        ids, coeff = crop_stencil(crop, wide_count, coordinates, image_size)
        full += stencil_margins(profiled_logits(crop, members), ids, coeff, CONFIG.beta)
    fine_margin = physical_margins(fine, fine_count, fine_coordinates, members, valid, CONFIG)
    pair = native_risk(full, fine_margin, fine_margin, parents, canonical, valid, CONFIG)
    directed = hard_pair_observation(wide, wide_count, coordinates, image_size, members,
                                    pair, valid, CONFIG.beta, CONFIG.query_chunk)
    potential, consistency = signed_potential(directed, valid)
    baseline = local.double() + operator.double() @ (broad.double() - local.double())
    values = {'Geometry': local.double(), 'NoAdmission_Exact': baseline,
              PRIMARY: baseline + operator.double() @ potential}
    if bool((pair[:, canonical] != 0).any()) or not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Nonfinite scores or unprotected canonical alias.')
    grouped = pair[:, members]
    rivals = ~torch.eye(len(members), device=pair.device, dtype=torch.bool)
    retained = (grouped[valid] == 0).sum(2)[:, rivals]
    stats = {'retained_count_mean': float(retained.double().mean()) if retained.numel() else 20.,
             'deleted_alias_rival_fraction': float((retained < members.shape[1]).double().mean()) if retained.numel() else 0.,
             'canonical_risk_max': float(pair[:, canonical].abs().max()),
             'mean_absolute_admission_potential': float(potential.abs().mean()),
             **consistency}
    return values, pair, stats
