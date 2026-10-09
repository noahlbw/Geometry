"""Semantic eligibility changes reference evidence, not unconditional class scores."""
from dataclasses import dataclass

import torch

from .coherent_native_admission import native_risk, union_risk
from .native_class_ownership import CONFIG as NATIVE_CONFIG, family_class_field, ownership_risk
from .semantic_role_admission import CONFIG as SOURCE_CONFIG, METHODS as LEGACY
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-semantic-admissible-reference-v1-20261003'
PRIMARY = 'SemanticReferenceJoint_Exact'
SHUFFLES = tuple('SemanticReferenceShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'SemanticReferenceOnly_Exact', 'SemanticReferenceNoView_Exact',
    'SemanticReferenceAttachmentOnly_Exact', 'SemanticReferenceCosine_Exact',
    'SemanticReferenceBinary_Exact', 'SemanticReferenceMeanLogit', *SHUFFLES)
METHODS = (*LEGACY, *NEW_METHODS)
GATE_CONTROLS = ('CoherentNativeJoint_Exact', 'CoherentNoHoldout_Exact',
    'ClassAttachmentJoint_Exact', 'ClassAttachmentTextOnly_Exact',
    'SemanticRoleJoint_Exact', 'SemanticRoleVisualSupported_Exact',
    'SemanticReferenceOnly_Exact', 'SemanticReferenceNoView_Exact',
    'SemanticReferenceAttachmentOnly_Exact', 'SemanticReferenceCosine_Exact',
    'SemanticReferenceBinary_Exact', 'SemanticReferenceMeanLogit', *SHUFFLES)
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_wrong_parent_domain_wins': 5, 'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1,
    'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'earlier_failed_gates_unchanged': True, 'no_post_result_control_promotion': True,
    'full_image_implementation_must_be_frozen_before_rollout': True}


@dataclass(frozen=True)
class SemanticReferenceConfig:
    beta: float = SOURCE_CONFIG.beta
    query_chunk: int = SOURCE_CONFIG.query_chunk
    random_seed: int = SOURCE_CONFIG.random_seed
    family_chunk: int = NATIVE_CONFIG.family_chunk


CONFIG = SemanticReferenceConfig()


def weighted_family_field(crops, count, coordinates, members, excluded, valid, admissible, config=CONFIG):
    if (admissible.shape != (members.numel(),) or not bool(torch.isfinite(admissible).all())
            or bool(((admissible < 0) | (admissible > 1)).any())
            or excluded.ndim != 2 or excluded.shape[1] != members.numel() or excluded.dtype != torch.bool
            or valid.dtype != torch.bool or coordinates.shape != (len(valid), 2)
            or config.beta <= 0 or config.family_chunk < 1):
        raise ValueError('Require bounded frozen admissibility, family masks and valid queries.')
    if bool((admissible == 1).all()):
        return family_class_field(crops, count, coordinates, members, excluded, valid, config)
    keep = ~excluded[:, members]
    remaining = keep.sum(-1)
    prior_known = remaining > 0
    semantic = admissible.double()[members][None]*keep
    mass = semantic.sum(-1)
    known = mass > 0
    log_weight = semantic.log()
    field = torch.zeros(len(valid), len(excluded), len(members), dtype=torch.float64, device=coordinates.device)
    coverage = torch.zeros(len(valid), device=coordinates.device)
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, (512, 512))
        coverage += coeff.sum(-1)
        raw, salience = crop.alias_logits[:, members].double(), crop.salience[members].double()
        if not bool(torch.isfinite(raw).all() and torch.isfinite(salience).all()):
            raise ValueError('Finite original native observations required.')
        for start in range(0, len(excluded), config.family_chunk):
            sl = slice(start, start+config.family_chunk)
            masked = salience[None].masked_fill(~keep[sl], -torch.inf)
            total = masked.logsumexp(-1, keepdim=True)
            profile = (masked-total.masked_fill(~prior_known[sl, :, None], 0.)).exp()
            # Preserve the original held-family profiled evidence and its units.
            evidence = raw[:, None]*remaining[None, sl, :, None]*profile[None]
            score = (config.beta*evidence+log_weight[None, sl]).logsumexp(-1)/config.beta
            score -= mass[sl].clamp_min(torch.finfo(torch.float64).tiny).log()[None]/config.beta
            score = score.masked_fill(~known[None, sl], 0.)
            field[:, sl] += (score[ids]*coeff.double()[..., None, None]).sum(1)
    field.masked_fill_(~valid[:, None, None], 0.)
    if (not bool(torch.isfinite(field).all()) or bool(valid.any()) and
            float((coverage[valid]-1).abs().max()) > 1e-6):
        raise RuntimeError('Native weighted reference is nonfinite or incomplete.')
    return field, known


def source_controls(crops, count, coordinates, members, excluded, valid, assignments,
                    parents, canonical, semantic, cosine, view_pair, config=CONFIG):
    if semantic.shape != (len(parents), len(members)) or not bool(torch.isfinite(semantic).all()):
        raise ValueError('Frozen semantic attachment scores must match aliases/classes.')
    empty = torch.zeros(1, len(parents), dtype=torch.bool, device=parents.device)
    view = view_pair.double().amax(-1)
    zero_assignments = torch.zeros_like(assignments)
    all_one = torch.ones(len(parents), dtype=torch.float64, device=parents.device)

    def reference_risks(conflict, admissible):
        held, known = weighted_family_field(crops, count, coordinates, members, excluded, valid, admissible, config)
        full, full_known = weighted_family_field(crops, count, coordinates, members, empty, valid, admissible, config)
        class_risk = native_risk(full, full_known, zero_assignments, parents, valid)
        base = union_risk(view, class_risk)
        attach = ownership_risk(held, known, assignments, parents, canonical, conflict, valid).amax(-1)
        return base, attach, class_risk, held, known, full, full_known

    admissible = 1-semantic.double().amax(-1)
    base, extra, class_risk, held, known, full, full_known = reference_risks(semantic, admissible)
    old_full, old_full_known = weighted_family_field(crops, count, coordinates, members, empty, valid, all_one, config)
    old_base = union_risk(view, native_risk(old_full, old_full_known, zero_assignments, parents, valid))
    cosine_extra = ownership_risk(held, known, assignments, parents, canonical, cosine, valid).amax(-1)
    hard_base, hard_extra, *_ = reference_risks(semantic, (semantic.amax(-1) == 0).double())
    output = {PRIMARY: union_risk(base, extra), 'SemanticReferenceOnly_Exact': base,
        'SemanticReferenceNoView_Exact': union_risk(class_risk, extra),
        'SemanticReferenceAttachmentOnly_Exact': extra,
        'SemanticReferenceCosine_Exact': union_risk(base, cosine_extra),
        'SemanticReferenceBinary_Exact': union_risk(hard_base, hard_extra)}
    spectrum_errors = {}
    generator = torch.Generator().manual_seed(config.random_seed)
    for name in SHUFFLES:
        permutation = torch.stack([torch.randperm(members.shape[-1], generator=generator) for _ in members]).to(members.device)
        changed = torch.empty_like(semantic)
        grouped = semantic[members]
        changed[members] = grouped.gather(1, permutation[..., None].expand_as(grouped))
        changed_weights = 1-changed.amax(-1)
        shuffled_base, shuffled_extra, *_ = reference_risks(changed, changed_weights)
        output[name] = union_risk(shuffled_base, shuffled_extra)
        spectrum_errors[name] = float((changed_weights[members].sort(-1).values-
            admissible[members].sort(-1).values).abs().max())
    return output, {'base': base, 'extra': extra, 'old_base': old_base,
        'full_reference': full, 'full_known': full_known, 'held_reference': held,
        'held_known': known, 'admissible': admissible, 'semantic_spectrum_errors': spectrum_errors}
