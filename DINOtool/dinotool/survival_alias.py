"""Same native risk with survivor-normalized soft contextual aggregation."""
import numpy as np
import torch

from .calibrated_competitive_alias import profiled_logits
from .native_alias_noise import METHODS as LEGACY, GATE as PREVIOUS_GATE, PRIMARY as SOURCE
from .native_alias_noise import CONFIG, SCENARIOS, STYLE_FILES, ALIAS_SHUFFLES as OLD_SHUFFLES
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-native-soft-survival-alias-v1-20261003'
PRIMARY = 'SurvivalAlias_Exact'
SHUFFLES = tuple('SurvivalAliasShuffle'+str(i)+'_Exact' for i in range(3))
SPATIAL = tuple('SurvivalSpatialShuffle'+str(i)+'_Exact' for i in range(3))
EXTRA = (PRIMARY, 'SurvivalText_Exact', *SHUFFLES, 'SurvivalMeanLogit', 'SurvivalPairMean_Exact', *SPATIAL)
METHODS = (*LEGACY, *EXTRA)
GATE_CONTROLS = (*PREVIOUS_GATE['wrong_parent_mean_above_controls'], 'SurvivalText_Exact',
                 *SHUFFLES, 'SurvivalMeanLogit', 'SurvivalPairMean_Exact', *SPATIAL)
GATE = {**PREVIOUS_GATE, 'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
        'clean_mean_above_previous_soft': True, 'wrong_parent_mean_above_previous_soft': True,
        'same_source_and_old_predictions_required': True}


def soft_survival_observation(crops, count, coordinates, image_size, members, risk, valid, beta=1., chunk=128):
    if (risk.shape != (len(valid), members.numel(), len(members)) or beta <= 0 or chunk < 1
            or not bool(torch.isfinite(risk).all()) or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Finite bounded alias/rival risks and positive beta/chunk required.')
    output = torch.zeros((len(valid), len(members), len(members)), dtype=torch.float64, device=risk.device)
    k = members.shape[-1]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        weights = 1-risk[sl, members].double()
        mass = weights.sum(2)
        if bool((mass <= 0).any()):
            raise ValueError('Every pair needs positive surviving alias mass.')
        for crop in crops:
            ids, coeff = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = (beta*profiled_logits(crop, members)[ids]).double()
            full = evidence.logsumexp(-1)
            changed = (evidence[..., None]+weights.log()[:, None]).logsumexp(3)
            delta = (changed-full[..., None]+(k/mass).log()[:, None])/beta
            delta = delta.masked_fill((weights == 1).all(2)[:, None], 0.)
            output[sl] += (delta*coeff.double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    return output


def signed_allocation_controls(directed, valid, seed=CONFIG.random_seed):
    values = np.asarray(directed, np.float64)
    valid = np.asarray(valid, bool)
    if (values.ndim != 3 or values.shape[-1] != values.shape[-2] or valid.shape != values.shape[:1]
            or not np.isfinite(values).all() or np.any(values[~valid] != 0)
            or np.any(np.diagonal(values, axis1=1, axis2=2) != 0)):
        raise ValueError('Finite zero-self/invalid signed actions required.')
    known = np.flatnonzero(valid)
    mean = np.zeros_like(values)
    if len(known):
        mean[known] = values[known].mean(0)
    outputs = {'SurvivalPairMean_Exact': mean}
    for i, name in enumerate(SPATIAL):
        generator, shuffled = np.random.default_rng(seed+i), np.zeros_like(values)
        for c in range(values.shape[-1]):
            for d in range(values.shape[-1]):
                shuffled[known, c, d] = values[generator.permutation(known), c, d]
        outputs[name] = shuffled
    return outputs
