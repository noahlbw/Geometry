"""Vocabulary-time role priors in local LME; unchanged FamilySUM wide path.

Recorded generated wrappers are not independent local semantic observations.
Without that provenance every existing member remains in its lexical family.
This is static input-role treatment, not pixel-wise semantic reliability.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from . import family_mass_alias as old


IMPLEMENTATION = 'geometry-fixed20-local-role-family-wide-v1-20261009'
CURRENT_BASE = 'Current20_Base'
RAW_BASE = 'Complete20_Base'
UNIFORM = 'Complete20_UniformMass'
COMPLETE_BASE = 'LocalRole_ClassPooled'
PRIMARY = 'LocalRole_Prior'
SHUFFLE = 'LocalRole_AliasShuffle'
VIP = 'VIP_Complete20'
METHODS = (CURRENT_BASE, RAW_BASE, UNIFORM, COMPLETE_BASE, PRIMARY, SHUFFLE)
SEED = 20261012


@dataclass(frozen=True)
class Plan:
    mass: object
    families: object
    members: torch.Tensor
    log_prior: torch.Tensor
    shuffled_log_prior: torch.Tensor
    uniform_classes: tuple
    role_source: str
    zero_local_slots: int

    @property
    def changed_classes(self):
        return self.mass.changed_classes


def role_priors(words, labels, canonical, roots=None):
    """One unit of local mass/class, equal mass for each declared family."""
    if len(words) != 20 or len(labels) != 20 or not 0 <= canonical < 20:
        raise ValueError('Twenty slots and one canonical required.')
    groups = len(set(labels))
    if sorted(set(labels)) != list(range(groups)):
        raise ValueError('Nonempty contiguous families required.')
    if roots is not None and len(roots) != groups:
        raise ValueError('One documented exact root per generated family required.')
    prior = [0.] * 20
    for g in range(groups):
        ids = [i for i, label in enumerate(labels) if label == g]
        if roots is not None:
            ids = [i for i in ids if words[i] == roots[g]]
            if len(ids) != 1:
                raise ValueError('Documented root must occur once in its family.')
        for i in ids:
            prior[i] = 1. / (groups * len(ids))
    if prior[canonical] <= 0 or abs(sum(prior) - 1.) > 1e-12:
        raise ValueError('Canonical support or unit local mass violated.')
    return prior


def prepare_plan(bank, families, roots=None):
    mass = old.prepare_plan(bank, families)
    members = torch.stack(mass.mean.members)
    protected = bank.canonical_mask[members].detach().cpu()
    words = [[bank.alias_names[i] for i in ids] for ids in members.cpu().tolist()]
    priors = []
    for c, row in enumerate(words):
        canonical = protected[c].nonzero().flatten().tolist()
        if len(canonical) != 1:
            raise ValueError('One canonical per class required.')
        priors.append(role_priors(row, families[c], canonical[0], None if roots is None else roots[c]))
    p = torch.tensor(priors, device=members.device, dtype=torch.float32)
    logp = p.log()
    rng = torch.Generator().manual_seed(SEED)
    shuffled = logp.clone()
    for c in range(bank.class_count):
        ids = (~protected[c]).nonzero().flatten()
        order = ids[torch.randperm(len(ids), generator=rng)]
        shuffled[c, ids.to(logp.device)] = logp[c, order.to(logp.device)]
    uniform = tuple(all(abs(v - .05) < 1e-12 for v in row) for row in priors)
    return Plan(mass, families, members, logp, shuffled, uniform,
                'recorded_generation_roots' if roots is not None else 'declared_lexical_members',
                sum(v == 0 for row in priors for v in row))


def apply_recorded_roles(plan, bank, roots):
    if roots is None:
        return plan
    return prepare_plan(bank, plan.families, roots)


def local_scores(cosine, plan, method, temperature=.07):
    """Only [patch,class,20], never [patch,class,alias,rival]."""
    if method not in (PRIMARY, SHUFFLE) or temperature <= 0:
        raise ValueError('Frozen local role route required.')
    values = cosine[:, plan.members] / .07
    logp = plan.log_prior if method == PRIMARY else plan.shuffled_log_prior
    score = (values + logp[None]).logsumexp(-1)
    # Exact old multiply/divide rounding for genuinely uniform classes.
    for c, uniform in enumerate(plan.uniform_classes):
        if uniform:
            score[:, c] = values[:, c].logsumexp(-1) - math.log(20)
    return (.07 * score) / temperature


@torch.inference_mode()
def role_probabilities(source, profile, bank, broad, plan, method, cosine_cache):
    from eval_geometry_vip_reliability import sample_broad
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .development_readout import coupled
    from .inference import hann_blend_window

    h, w = source['size']
    text = F.normalize(bank.features.float(), dim=-1)
    blend = torch.from_numpy(hann_blend_window(512)).to(text.device)
    with DeviceProbabilityAccumulator(bank.class_count, h, w, text.device) as accumulator:
        for number, tile in enumerate(source['local']):
            if number not in cosine_cache:
                cosine_cache[number] = tile['features'][profile.strength].float() @ text.T
            local = local_scores(cosine_cache[number], plan, method, profile.temperature)
            top, left = tile['top'], tile['left']
            wide = sample_broad(broad, top, left, h, w).reshape_as(local)
            logits = coupled(local, wide, tile['operator'], profile.coupling)
            dense = F.interpolate(logits.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                  mode='bilinear', align_corners=False)[0]
            ah, aw = min(512, h-top), min(512, w-left)
            accumulator.add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
        if not bool((accumulator.normalizer > 0).all()):
            raise RuntimeError('Local coverage gap.')
        normalized = accumulator.probabilities / accumulator.normalizer[None]
        result = F.interpolate(normalized[None], source['output_size'], mode='bilinear', align_corners=False)[0]
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite local-role probabilities.')
    return result


@torch.inference_mode()
def predict(image, geometry, vip, banks, queries, plans, dataset, methods=METHODS):
    from eval_development_readout import observations, probabilities
    from .fixed20_task_readout import POLICIES

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared methods required.')
    policy = POLICIES[dataset]
    source = observations(image, geometry, vip, (policy.strength,), wide_policy=policy.wide_policy)
    result = {}
    wanted = tuple(name for name in methods if name != CURRENT_BASE)
    if wanted:
        routes = tuple(dict.fromkeys('base' if name == RAW_BASE else 'uniform' if name == UNIFORM else 'sum'
                                     for name in wanted))
        plan = plans['Complete20']
        fields = old.wide_fields(source, queries['Complete20'], plan.mass, routes, policy.tau, policy.tem)
        cache, cosines = {}, {}
        for name in wanted:
            if name in (PRIMARY, SHUFFLE):
                probability = role_probabilities(source, policy.profile(), banks['Complete20'],
                                                fields['sum'], plan, name, cosines)
            else:
                route = 'base' if name == RAW_BASE else 'uniform' if name == UNIFORM else 'sum'
                probability = probabilities(source, policy.profile(), banks['Complete20'], fields[route], cache)
            result[name] = probability.argmax(0).cpu().numpy()
    if CURRENT_BASE in methods:
        field = old.wide_fields(source, queries['Current20'], plans['Current20'].mass,
                                ('base',), policy.tau, policy.tem)['base']
        probability = probabilities(source, policy.profile(), banks['Current20'], field, {})
        result[CURRENT_BASE] = probability.argmax(0).cpu().numpy()
    return result, dict(geometry_encodings=len(source['local']), wide_encodings=len(source['wide']),
        fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0, alias_slots_per_class=20,
        changed_family_classes=plans['Complete20'].changed_classes,
        zero_local_prior_slots=plans['Complete20'].zero_local_slots,
        maximum_local_alias_elements=1024*plans['Complete20'].members.numel())
