"""Frozen vocabulary-stress contract for native-query contextual admission."""
import torch

from .calibrated_competitive_alias import profiled_logits
from .native_query_alias import CONFIG
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-native-query-alias-noise-study-v1-20261003'
PRIMARY = 'NativeAlias_Exact'
ALIAS_SHUFFLES = tuple('AliasShuffle'+str(i)+'_Exact' for i in range(3))
RANDOM_DELETIONS = tuple('RandomDelete'+str(i)+'_Exact' for i in range(3))
SPATIAL_SHUFFLES = tuple('NoiseDirectionalShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = ('Geometry', 'NoAdmission_Exact', PRIMARY, 'NativeAlias_MeanLogit', 'MeanLogit_Original',
    'TextOnly_Exact', 'HardDelete_Exact', *RANDOM_DELETIONS, *ALIAS_SHUFFLES,
    'NoiseDirectionalMean_Exact', *SPATIAL_SHUFFLES)
SCENARIOS = ('clean', 'wrong_parent', 'paraphrase')
STYLE_FILES = {'loveda': 'gar_llm_raw20_loveda_v1.json', 'udd5': 'gar_llm_raw20_udd5_v1.json',
               'oem': 'gar_llm_raw20_oem_v1.json'}
GATE_CONTROLS = ('NoAdmission_Exact', 'TextOnly_Exact', 'HardDelete_Exact', *RANDOM_DELETIONS,
    *ALIAS_SHUFFLES, 'NativeAlias_MeanLogit', 'MeanLogit_Original', 'NoiseDirectionalMean_Exact', *SPATIAL_SHUFFLES)
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_wrong_parent_domain_wins': 5, 'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1, 'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'style_diagnostic_only': True, 'earlier_failed_gates_unchanged': True, 'no_automatic_full_rollout': True}


def signed_potential(directed, valid):
    if (directed.ndim != 3 or directed.shape[-1] != directed.shape[-2]
            or valid.shape != directed.shape[:1] or not bool(torch.isfinite(directed).all())
            or bool((directed[~valid] != 0).any()) or bool((directed.diagonal(dim1=-2, dim2=-1) != 0).any())):
        raise ValueError('Finite directed comparisons with zero self and invalid actions required.')
    margins = directed.double()-directed.double().transpose(-1, -2)
    potential = margins.mean(-1)
    cycle = margins-(potential[:, :, None]-potential[:, None])
    error = float(cycle.sum(-1).abs().max())
    gauge = float(potential.sum(-1).abs().max())
    if max(error, gauge) > 1e-10:
        raise RuntimeError('Classical competitive consistency identity failed.')
    return potential, {'normal_equation_max_error': error, 'gauge_max_error': gauge}


def hard_pair_observation(crops, count, coordinates, image_size, members, risk, valid, beta=1., chunk=128):
    """Same profiled observer; delete risk-positive aliases and normalize retained count."""
    if (risk.shape != (len(valid), members.numel(), len(members)) or beta <= 0
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Finite bounded alias/rival risks required.')
    output = torch.zeros((len(valid), len(members), len(members)), dtype=torch.float64, device=risk.device)
    k = members.shape[-1]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        keep = risk[sl, members] == 0
        remaining = keep.sum(2)
        if bool((remaining == 0).any()):
            raise ValueError('Hard aggregation needs a surviving alias in every pair.')
        for crop in crops:
            ids, coeff = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = (beta*profiled_logits(crop, members)[ids]).double()
            full = evidence.logsumexp(-1)
            changed = evidence[..., None].masked_fill(~keep[:, None], -torch.inf).logsumexp(3)
            delta = (changed-full[..., None]+(k/remaining.double()).log()[:, None])/beta
            delta = delta.masked_fill(keep.all(2)[:, None], 0.)
            output[sl] += (delta*coeff.double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    return output
