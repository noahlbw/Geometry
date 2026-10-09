"""Frozen redundancy-by-class-mass interaction, not a new reliability signal.

Each declared family receives one unit of exponential evidence. The complete
factorial separates an inherited class offset from the already tested relative
alias weights. Inherited salience, local scores and Geometry stay unchanged.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .family_lme_alias import prepare_plan as mean_plan, reduce_fixed_evidence


IMPLEMENTATION = 'geometry-fixed20-family-mass-interaction-v1-20261009'
CURRENT_BASE = 'Current20_Base'
CURRENT_SUM = 'Current20_FamilySum'
COMPLETE_BASE = 'Complete20_Base'
FAMILY_MEAN = 'Complete20_FamilyLME'
UNIFORM = 'Complete20_UniformMass'
PRIMARY = 'Complete20_FamilySum'
SHUFFLE = 'Complete20_FamilySumShuffle'
VIP = 'VIP_Complete20'
METHODS = (CURRENT_BASE, CURRENT_SUM, COMPLETE_BASE, FAMILY_MEAN,
           UNIFORM, PRIMARY, SHUFFLE)
ROUTES = {CURRENT_BASE: ('Current20', 'base'), CURRENT_SUM: ('Current20', 'sum'),
          COMPLETE_BASE: ('Complete20', 'base'), FAMILY_MEAN: ('Complete20', 'mean'),
          UNIFORM: ('Complete20', 'uniform'), PRIMARY: ('Complete20', 'sum'),
          SHUFFLE: ('Complete20', 'shuffle')}


@dataclass(frozen=True)
class MassPlan:
    mean: object
    log_weights: tuple
    shuffled_log_weights: tuple
    offsets: tuple

    @property
    def changed_classes(self):
        return sum(n < 20 for n in self.mean.family_counts)


def prepare_plan(bank, families):
    old = mean_plan(bank, families)
    offsets = tuple(math.log(n / 20) for n in old.family_counts)
    return MassPlan(old,
        tuple(weight + offset for weight, offset in zip(old.log_weights, offsets)),
        tuple(weight + offset for weight, offset in zip(old.shuffled_log_weights, offsets)),
        offsets)


def reduce_routes(evidence, plan, c, wanted, tau):
    """Only two distinct alias reductions; the other arms are scalar shifts."""
    if tau <= 0 or evidence.shape[0] != 20:
        raise ValueError('Positive tau and twenty fixed evidence rows required.')
    fields = {}
    if any(name in wanted for name in ('base', 'uniform')):
        base = (tau * evidence).logsumexp(0) / tau
        if 'base' in wanted: fields['base'] = base
        if 'uniform' in wanted: fields['uniform'] = base + plan.offsets[c] / tau
    if any(name in wanted for name in ('mean', 'sum')):
        mean = reduce_fixed_evidence(evidence, plan.mean.log_weights[c], tau,
                                     identity=not plan.mean.active[c])
        if 'mean' in wanted: fields['mean'] = mean
        if 'sum' in wanted: fields['sum'] = mean + plan.offsets[c] / tau
    if 'shuffle' in wanted:
        shuffled = reduce_fixed_evidence(evidence, plan.mean.shuffled_log_weights[c], tau,
                                         identity=not plan.mean.active[c])
        fields['shuffle'] = shuffled + plan.offsets[c] / tau
    if set(fields) != set(wanted): raise ValueError('Unknown fixed route.')
    return fields


@torch.inference_mode()
def wide_fields(source, query, plan, wanted, tau, tem):
    from eval_development_readout import wide_scores
    if wanted == ('base',):
        return {'base': wide_scores(source, query, tau, tem)}
    h, w = source['wide_size']
    totals = {name: torch.zeros(len(query.class_names), h, w,
                               device=query.features.device) for name in wanted}
    count = torch.zeros(h, w, device=query.features.device)
    for crop in source['wide']:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            raw = torch.einsum('bnd,mtd->bnmt', crop['features'], query.features.float()).mean(-1)[0]
            visual = F.normalize(crop['features'].mean(1), dim=-1)
            text = F.normalize(query.features.float().mean(1), dim=-1)
            salience = ((visual @ text.T)[0] / tem).float()
            aliases = (raw * 40.).T.reshape(-1, 21, 21)
            fields = {name: [] for name in wanted}
            for c, ids in enumerate(plan.mean.members):
                weights = salience[ids].softmax(0)
                evidence = aliases[ids] * (weights / weights.mean())[:, None, None]
                reduced = reduce_routes(evidence, plan, c, wanted, tau)
                for name in wanted: fields[name].append(reduced[name])
            dense = {name: F.interpolate(torch.stack(rows)[None], (336, 336),
                    mode='bilinear', align_corners=False)[0].float()
                    for name, rows in fields.items()}
        top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
        for name in wanted: totals[name][:, top:top+ah, left:left+aw] += dense[name][:, :ah, :aw]
        count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()): raise RuntimeError('Wide coverage gap.')
    return {name: field / count[None] for name, field in totals.items()}


@torch.inference_mode()
def predict(image, geometry, vip, banks, queries, plans, dataset, methods=METHODS):
    from eval_development_readout import observations, probabilities
    from .fixed20_task_readout import POLICIES
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared fixed methods required.')
    policy = POLICIES[dataset]
    source = observations(image, geometry, vip, (policy.strength,), wide_policy=policy.wide_policy)
    result = {}; residuals = []; mass = []
    for key in dict.fromkeys(ROUTES[name][0] for name in methods):
        names = [name for name in methods if ROUTES[name][0] == key]
        routes = tuple(dict.fromkeys(ROUTES[name][1] for name in names))
        fields = wide_fields(source, queries[key], plans[key], routes, policy.tau, policy.tem)
        cache = {}
        for name in names:
            probability = probabilities(source, policy.profile(), banks[key], fields[ROUTES[name][1]], cache)
            result[name] = probability.argmax(0).cpu().numpy()
        if all(name in fields for name in ('base', 'mean', 'uniform', 'sum')):
            residuals.append(float(((fields['sum']-fields['uniform'])-(fields['mean']-fields['base'])).abs().max()))
            mass.append(float((fields['uniform']-fields['base']).abs().mean()))
    return result, dict(geometry_encodings=len(source['local']), wide_encodings=len(source['wide']),
        fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0,
        alias_slots_per_class=20, changed_family_classes=plans['Complete20'].changed_classes,
        factorial_field_identity_max=max(residuals, default=0.),
        class_mass_offset_abs=sum(mass)/max(len(mass), 1))
