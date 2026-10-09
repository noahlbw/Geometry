"""Wide-only expression-mass correction at fixed inherited alias evidence.

This is language provenance/lexical grouping, not visual reliability estimation.
Inherited salience is unchanged, so no end-to-end duplication invariance is
claimed. Uniform family sizes use the exact inherited reduction.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


IMPLEMENTATION = 'geometry-fixed20-wide-family-exponential-mean-v1-20261009'
CURRENT_BASE = 'Current20_Base'
CURRENT_TREATMENT = 'Current20_FamilyLME'
COMPLETE_BASE = 'Complete20_Base'
PRIMARY = 'Complete20_FamilyLME'
SHUFFLE = 'Complete20_FamilyShuffle'
METHODS = (CURRENT_BASE, CURRENT_TREATMENT, COMPLETE_BASE, PRIMARY, SHUFFLE)
VIP = 'VIP_Complete20'
SEED = 20261009
ROUTES = {CURRENT_BASE: ('Current20', 'base'),
          CURRENT_TREATMENT: ('Current20', 'family'),
          COMPLETE_BASE: ('Complete20', 'base'),
          PRIMARY: ('Complete20', 'family'),
          SHUFFLE: ('Complete20', 'shuffle')}


@dataclass(frozen=True)
class FamilyPlan:
    members: tuple
    log_weights: tuple
    shuffled_log_weights: tuple
    family_counts: tuple
    active: tuple
    changed_classes: int


def expression_log_weights(labels):
    """Mean-one slot weights; each family has total mass 20/G."""
    if len(labels) != 20 or sorted(set(labels)) != list(range(max(labels) + 1)):
        raise ValueError('Twenty slots and contiguous nonempty families required.')
    counts = [labels.count(g) for g in range(max(labels) + 1)]
    if len(set(counts)) == 1:
        return [0.] * 20
    return [math.log(20 / (len(counts) * counts[g])) for g in labels]


def prepare_plan(bank, families):
    if len(families) != bank.class_count:
        raise ValueError('One frozen family partition per class required.')
    members, weights, shuffled, counts, active = [], [], [], [], []
    rng = torch.Generator().manual_seed(SEED)
    for c, labels in enumerate(families):
        ids = (bank.parent_indices == c).nonzero().flatten()
        protected = bank.canonical_mask[ids].detach().cpu()
        if len(ids) != 20 or int(protected.sum()) != 1:
            raise ValueError('Twenty slots with one canonical anchor required.')
        declared = expression_log_weights(labels)
        active.append(any(v != 0 for v in declared))
        values = torch.tensor(declared, dtype=torch.float32,
                              device=ids.device)
        permutation = torch.arange(20)
        unprotected = (~protected).nonzero().flatten()
        permutation[unprotected] = unprotected[torch.randperm(19, generator=rng)]
        control = values[permutation.to(ids.device)]
        if not torch.equal(values[protected.to(ids.device)], control[protected.to(ids.device)]):
            raise RuntimeError('Shuffle changed canonical weight.')
        members.append(ids); weights.append(values); shuffled.append(control)
        counts.append(len(set(labels)))
    return FamilyPlan(tuple(members), tuple(weights), tuple(shuffled), tuple(counts),
                      tuple(active), sum(active))


def reduce_fixed_evidence(evidence, log_weight, tau=1., *, identity=None):
    if tau <= 0 or evidence.shape[0] != 20 or log_weight.shape != (20,):
        raise ValueError('Positive tau and twenty fixed alias evidence rows required.')
    if identity is None: identity = not bool(torch.count_nonzero(log_weight))
    if identity:
        return (tau * evidence).logsumexp(0) / tau
    shape = (20,) + (1,) * (evidence.ndim - 1)
    return (tau * evidence + log_weight.reshape(shape)).logsumexp(0) / tau


@torch.inference_mode()
def wide_fields(source, query, plan, wanted, tau, tem):
    from eval_development_readout import wide_scores
    if wanted == ('base',):
        return {'base': wide_scores(source, query, tau, tem)}
    h, w = source['wide_size']
    totals = {name: torch.zeros(len(query.class_names), h, w, device=query.features.device)
              for name in wanted}
    count = torch.zeros(h, w, device=query.features.device)
    for crop in source['wide']:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            raw = torch.einsum('bnd,mtd->bnmt', crop['features'], query.features.float()).mean(-1)[0]
            visual = F.normalize(crop['features'].mean(1), dim=-1)
            text = F.normalize(query.features.float().mean(1), dim=-1)
            salience = ((visual @ text.T)[0] / tem).float()
            aliases = (raw * 40.).T.reshape(-1, 21, 21)
            fields = {name: [] for name in wanted}
            for c, ids in enumerate(plan.members):
                weights = salience[ids].softmax(0)
                evidence = aliases[ids] * (weights / weights.mean())[:, None, None]
                for name in wanted:
                    if name == 'base': field = (tau * evidence).logsumexp(0) / tau
                    elif name in ('family', 'shuffle'):
                        lw = plan.log_weights[c] if name == 'family' else plan.shuffled_log_weights[c]
                        field = reduce_fixed_evidence(evidence, lw, tau, identity=not plan.active[c])
                    else: raise ValueError('Unknown frozen wide route.')
                    fields[name].append(field)
            dense = {name: F.interpolate(torch.stack(rows)[None], (336, 336),
                                        mode='bilinear', align_corners=False)[0].float()
                     for name, rows in fields.items()}
        top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
        for name in wanted: totals[name][:, top:top+ah, left:left+aw] += dense[name][:, :ah, :aw]
        count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()): raise RuntimeError('Wide coverage gap.')
    return {name: values / count[None] for name, values in totals.items()}


@torch.inference_mode()
def predict(image, geometry, vip, banks, queries, plans, dataset, methods=METHODS):
    from eval_development_readout import observations, probabilities
    from .fixed20_task_readout import POLICIES
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen methods required.')
    policy = POLICIES[dataset]
    source = observations(image, geometry, vip, (policy.strength,), wide_policy=policy.wide_policy)
    result = {}; deltas = []
    for key in dict.fromkeys(ROUTES[name][0] for name in methods):
        names = [name for name in methods if ROUTES[name][0] == key]
        routes = tuple(dict.fromkeys(ROUTES[name][1] for name in names))
        fields = wide_fields(source, queries[key], plans[key], routes, policy.tau, policy.tem)
        cache = {}
        for name in names:
            probability = probabilities(source, policy.profile(), banks[key], fields[ROUTES[name][1]], cache)
            result[name] = probability.argmax(0).cpu().numpy()
        if 'base' in fields and 'family' in fields:
            deltas.append(float((fields['family'] - fields['base']).abs().mean()))
    return result, dict(geometry_encodings=len(source['local']), wide_encodings=len(source['wide']),
                       fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0,
                       alias_slots_per_class=20, wide_treatment_abs=sum(deltas)/max(len(deltas), 1),
                       changed_family_classes=plans['Complete20'].changed_classes)
