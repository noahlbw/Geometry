"""Physical8 alias/rival contradiction replaces the original native16 view source."""
from dataclasses import dataclass

import torch

from .coherent_native_admission import union_risk
from .fine_reference_admission import FineReferenceConfig
from .matched_contribution_alias import stencil_margins
from .native_ownership_reader import query_permutation
from .native_query_alias import CONFIG as NATIVE_CONFIG, native_risk
from .calibrated_competitive_alias import profiled_logits
from .native_alias_noise import hard_pair_observation, signed_potential
from .pair_context_reader import pair_observation
from .rival_preserving_alias import competitive_potential
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'geometry-physical8-alias-view-admission-v1-20261003'
PRIMARY = 'FineAliasViewJoint_Exact'
SHUFFLES = tuple('FineAliasViewShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = {'Geometry': 'Geometry', 'NoAdmission_Exact': 'NoAdmission_Exact',
    'FineNative16_Exact': 'FineReferenceOnly_Exact',
    'PreviousFineComplete_Exact': 'FineReferenceJoint_Exact',
    'PreviousFineCosine_Exact': 'FineReferenceCosine_Exact'}
OBSERVER_SHUFFLES = tuple('ObserverFinePairRandom'+str(i) for i in range(3))
OBSERVER_METHODS = ('ObserverAll20', 'ObserverFinePairSoft', 'ObserverFinePairHard',
    'ObserverNativePairHard', *OBSERVER_SHUFFLES)
METHODS = (*REPLAY, 'FineClassOnly_Exact', PRIMARY, 'FineAliasViewOnly_Exact',
    'FineAliasViewClassMean_Exact', 'FineAliasViewSpatialShuffle_Exact',
    'FineAliasViewHardDelete_Exact', *SHUFFLES, *OBSERVER_METHODS)
GATE_CONTROLS = ('FineNative16_Exact', 'PreviousFineComplete_Exact', 'PreviousFineCosine_Exact',
    'FineAliasViewOnly_Exact', 'FineAliasViewClassMean_Exact',
    'FineAliasViewSpatialShuffle_Exact', 'FineAliasViewHardDelete_Exact', *SHUFFLES)
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_wrong_parent_alias_increment_pp': .1, 'minimum_wrong_parent_domain_wins': 5,
    'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1,
    'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'no_automatic_full_rollout': True, 'no_post_result_control_promotion': True}


@dataclass(frozen=True)
class FineAliasViewConfig(FineReferenceConfig):
    epsilon: float = NATIVE_CONFIG.epsilon


CONFIG = FineAliasViewConfig()


def physical_margins(crops, count, coordinates, members, valid, config=CONFIG):
    config.validate()
    margins = torch.zeros(len(valid), members.numel(), len(members), device=coordinates.device)
    for crop in crops:
        from .stratified_soft_alias import crop_stencil

        ids, coeff = crop_stencil(crop, count, coordinates, (512, 512))
        margins += stencil_margins(profiled_logits(crop, members), ids, coeff, config.beta)
    if not bool(torch.isfinite(margins).all()):
        raise RuntimeError('Nonfinite physical8 alias/rival margins.')
    return margins.masked_fill(~valid[:, None, None], 0.)


def view_sources(broad, fine, parents, canonical, valid, members, class_risk, config=CONFIG):
    pair = native_risk(broad, fine, fine, parents, canonical, valid, config)
    view = pair.double().amax(-1)
    if class_risk.shape != view.shape:
        raise ValueError('Matching frozen fine class-reference risk required.')
    sources = {'FineClassOnly_Exact': class_risk, PRIMARY: union_risk(class_risk, view),
        'FineAliasViewOnly_Exact': view}
    generator = torch.Generator().manual_seed(config.random_seed)
    spectra = {}
    for name in SHUFFLES:
        changed = view.clone()
        for group in members:
            ids = group[~torch.isin(group, canonical)]
            permutation = torch.randperm(len(ids), generator=generator).to(view.device)
            changed[:, ids] = view[:, ids[permutation]]
        spectra[name] = float((changed[:, members].sort(-1).values-view[:, members].sort(-1).values).abs().max())
        sources[name] = union_risk(class_risk, changed)
    shared = view.clone()
    for group in members:
        ids = group[~torch.isin(group, canonical)]
        if len(ids):
            shared[:, ids] = view[:, ids].mean(-1, keepdim=True)
    shared[:, canonical] = 0.
    sources['FineAliasViewClassMean_Exact'] = union_risk(class_risk, shared)
    index = query_permutation(valid, config.random_seed)
    sources['FineAliasViewSpatialShuffle_Exact'] = union_risk(class_risk, view[index])
    spatial_error = float((view[valid].sort(0).values-view[index][valid].sort(0).values).abs().max()) if bool(valid.any()) else 0.
    return sources, {'pair': pair, 'view': view, 'alias_spectrum_errors': spectra,
        'spatial_view_spectrum_error': spatial_error,
        'class_mean_view_budget_error': float((shared[:, members].sum(-1)-view[:, members].sum(-1)).abs().max()),
        'canonical_view_risk_max': float(view[:, canonical].abs().max()),
        'unknown_zero_fraction': float((view[valid] == 0).double().mean()) if bool(valid.any()) else 1.}


def independent_observation(broad, crops, count, coordinates, image_size, members,
                            pair, valid, canonical, native_pair, config=CONFIG):
    """Apply the same pair admission without the Geometry anchor or reconstruction."""
    args = (crops, count, coordinates, image_size, members)
    directed = pair_observation(*args, pair, valid, config.beta, config.query_chunk)[2]
    potential, _, consistency = competitive_potential(directed, valid)
    values = {'ObserverAll20': broad.double(), 'ObserverFinePairSoft': broad.double()+potential}
    stats = {'ObserverFinePairSoft': {**consistency, 'count_max_error': 0,
        'canonical_risk_max': float(pair[:, canonical].abs().max())}}
    risks = {'ObserverFinePairHard': pair, 'ObserverNativePairHard': native_pair}
    grouped = pair[:, members]
    for name, permutation in zip(OBSERVER_SHUFFLES,
            alias_permutations(members, canonical, config.random_seed)):
        changed = torch.empty_like(pair)
        changed[:, members] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
        risks[name] = changed
    for name, risk in risks.items():
        directed = hard_pair_observation(*args, risk, valid, config.beta, config.query_chunk)
        potential, consistency = signed_potential(directed, valid)
        reference = native_pair if name == 'ObserverNativePairHard' else pair
        error = int(((risk[:, members] > 0).sum(2)-(reference[:, members] > 0).sum(2)).abs().max())
        stats[name] = {**consistency, 'count_max_error': error,
            'canonical_risk_max': float(risk[:, canonical].abs().max()),
            'deleted_alias_rival_fraction': float((risk[valid] > 0).double().mean())}
        values[name] = broad.double()+potential
    return values, stats
