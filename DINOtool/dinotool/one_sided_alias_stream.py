"""Execution-only acceleration of the retained one-sided fixed-slot reader."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .one_sided_alias_audit import (PRIMARY as REFERENCE, OBSERVATION_MEAN,
    PROTOCOL as SOURCE_PROTOCOL, scores as reference_scores)
from .rival_alias_fast import risk_statistics, sampled_cached
from .supported_positive_alias import predict_image as source_predict


IMPLEMENTATION = 'one-sided-fixed-slot-stream-execution-v1-20261008'
PRIMARY = 'OneSide_Stream'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled',
           REFERENCE, PRIMARY, OBSERVATION_MEAN)
DIAGNOSTICS = ('score_max_abs_error', 'score_bitwise_mismatch',
               'positive_directed_delta_max', 'mass_log_fallbacks')
PROTOCOL = {**SOURCE_PROTOCOL, 'execution': IMPLEMENTATION,
    'semantic_rule_changed': False, 'query_chunk': 1024, 'rival_chunk': 16}


@torch.inference_mode()
def fixed_slot_stream(observations, members, risk, valid, *, query_chunk=1024, rival_chunk=16):
    """Cache crop softmax before stencil gathering; bound temporary rival width.

    Keep FP64, crop order, fixed slots, padding and exact log-domain fallback.
    Only the kernel batching changes; compare score tolerance and actual pixels.
    """
    if (not observations or min(query_chunk, rival_chunk) < 1
            or risk.shape != (len(valid), members.numel(), len(members))
            or valid.dtype != torch.bool or not bool(torch.isfinite(risk).all())
            or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Matching finite bounded risks and positive block sizes required.')
    classes = len(members)
    output = torch.zeros((len(valid), classes, classes), dtype=torch.float64, device=risk.device)
    # Original implementation repeats this on four gathered stencil points in every block.
    prepared = [(crop, crop.evidence.double(), crop.evidence.double().softmax(-1))
                for crop in observations]
    fallbacks = 0
    for start in range(0, len(valid), query_chunk):
        sl = slice(start, start+query_chunk)
        for rival in range(0, classes, rival_chunk):
            rs = slice(rival, min(rival+rival_chunk, classes))
            allocation = 1-risk[sl, :, rs][:, members].double()
            if bool((allocation.sum(2) == 0).any()):
                raise ValueError('Every pair requires surviving alias mass.')
            untouched = (allocation == 1).all(2)[:, None]
            for crop, evidence, probability in prepared:
                responsibility = probability[crop.indices[sl]]
                mass = torch.einsum('qsck,qckd->qscd', responsibility, allocation)
                underflow = mass == 0
                change = mass.clamp_min(torch.finfo(torch.float64).tiny).log()
                if bool(underflow.any()):
                    sampled = evidence[crop.indices[sl]]
                    stable = (sampled[..., None]+allocation.log()[:, None]).logsumexp(3)
                    stable -= sampled.logsumexp(-1)[..., None]
                    change = torch.where(underflow, stable, change)
                    fallbacks += int(underflow.sum())
                change.masked_fill_(untouched, 0.)
                output[sl, :, rs] += (change*crop.coefficients[sl].double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    maximum = float(output.max())
    if maximum > 1e-12 or not bool(torch.isfinite(output).all()):
        raise RuntimeError('Nonfinite or positive fixed-slot action.')
    return output, dict(positive_directed_delta_max=maximum, mass_log_fallbacks=float(fallbacks))


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared stream execution methods required.')
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    innovation = .5*(fine_field.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = baseline+operator.double()@innovation
    values = {OBSERVATION_MEAN: positive} if OBSERVATION_MEAN in methods else {}
    stats = dict.fromkeys((*DIAGNOSTICS, 'canonical_risk_max', 'mean_absolute_admission_potential',
        'normal_equation_max_error', 'gauge_max_error', 'deleted_alias_rival_fraction'), 0.)
    stats['retained_count_mean'] = float(members.shape[1])
    if set(methods) == {OBSERVATION_MEAN}:
        return values, stats
    wm = sampled_cached(wide, coordinates)
    fm = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
    directed, diagnostics = fixed_slot_stream(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    result = positive+.5*(operator.double()@potential)
    if PRIMARY in methods:
        values[PRIMARY] = result
    if REFERENCE in methods:
        previous, _ = reference_scores(local, operator, broad, fine_field, wide, fine,
            coordinates, relation, valid, members, canonical, parents, methods=(REFERENCE,))
        values.update(previous)
        error = float((result-previous[REFERENCE]).abs().max())
        stats.update(score_max_abs_error=error,
                     score_bitwise_mismatch=float(not torch.equal(result, previous[REFERENCE])))
        if error > 1e-10:
            raise RuntimeError(f'Execution changes patch scores: {error}')
    stats.update({**risk_statistics(risk, potential, valid, members, canonical),
                  **diagnostics, **consistency})
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite streamed score.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
