"""Sparse task-semantic constraints on positive wide-alias excess.

The image's unchanged coupled top-two classes choose a declared semantic
contract. Unknown contracts abstain. No semantic correctness is inferred from
visual agreement, and no aliases are removed or survivor masses redistributed.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


IMPLEMENTATION = 'geometry-semantic-contract-positive-excess-v1-20261009'
BASE = 'Same20_NoContract'
PRIMARY = 'Same20_ContractExcess'
METHODS = (BASE, PRIMARY)


@dataclass(frozen=True)
class Plan:
    members: torch.Tensor
    canonical: tuple
    edges: tuple
    log_weights: torch.Tensor
    log_complements: torch.Tensor
    lookup: torch.Tensor
    by_class: tuple


def prepare_plan(bank, metadata):
    classes = tuple(bank.class_names)
    aliases = tuple(bank.alias_names)
    if (tuple(metadata['classes']) != classes or tuple(metadata['aliases']) != aliases
            or metadata['target_images_loaded'] or metadata['target_masks_loaded']):
        raise ValueError('Frozen image-free task contract and unchanged bank required.')
    members = torch.stack([(bank.parent_indices == c).nonzero().flatten()
                           for c in range(bank.class_count)])
    if members.shape[1] != 20:
        raise ValueError('The paired experiment retains twenty slots per class.')
    canonical = []
    for ids in members:
        anchors = bank.canonical_mask[ids].nonzero().flatten().tolist()
        if len(anchors) != 1:
            raise ValueError('One protected existing canonical slot required.')
        canonical.append(anchors[0])
    edges, rows = [], []
    for entry in metadata['edges']:
        c, r = entry['owner'], entry['rival']
        row = entry['weights']
        if (not 0 <= c < len(classes) or not 0 <= r < len(classes) or c == r
                or (c, r) in edges or len(row) != 20
                or any(v not in (0., .5, 1.) for v in row)
                or row[canonical[c]] != 1. or all(v == 1. for v in row)):
            raise ValueError('Unique active sparse contract with protected canonical required.')
        edges.append((c, r)); rows.append(row)
    weights = torch.tensor(rows, device=members.device, dtype=torch.float32).reshape(-1, 20)
    lookup = torch.full((len(classes), len(classes)), -1, device=members.device, dtype=torch.long)
    by_class = []
    for c in range(len(classes)):
        current = tuple(i for i, (owner, _) in enumerate(edges) if owner == c)
        by_class.append(current)
        for i in current:
            lookup[edges[i]] = i
    return Plan(members, tuple(canonical), tuple(edges), weights.log(),
                (1-weights).log(), lookup, tuple(by_class))


def attenuate_log_evidence(values, reference, log_weights, log_complements):
    """min(exp(v),exp(b)) + w*[exp(v)-exp(b)]+, stably in log units."""
    mixed = torch.logaddexp(values + log_weights, reference + log_complements)
    return torch.where(values > reference, mixed, values)


@torch.inference_mode()
def wide_fields(source, query, plan, policy, treatment):
    h, w = source['wide_size']
    nc, ne = len(query.class_names), len(plan.edges)
    device = query.features.device
    output = torch.zeros(nc, h, w, device=device)
    corrections = torch.zeros(ne, h, w, device=device) if treatment else None
    count = torch.zeros(h, w, device=device)
    active_fraction = torch.zeros((), device=device)
    active_elements = 0
    # This is the inherited wide scoring calculation, shared by off/on arms.
    for crop in source['wide']:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            raw = torch.einsum('bnd,mtd->bnmt', crop['features'], query.features.float()).mean(-1)[0]
            visual = F.normalize(crop['features'].mean(1), dim=-1)
            text = F.normalize(query.features.float().mean(1), dim=-1)
            salience = ((visual @ text.T)[0] / policy.tem).float()
            aliases = (raw * 40.).T.reshape(-1, 21, 21)
            rows, changes = [], [None] * ne
            for c, ids in enumerate(plan.members):
                weights = salience[ids].softmax(0)
                evidence = aliases[ids] * (weights / weights.mean())[:, None, None]
                value = policy.tau * evidence
                base = value.logsumexp(0) / policy.tau
                rows.append(base)
                if treatment and plan.by_class[c]:
                    edge_ids = list(plan.by_class[c])
                    logw = plan.log_weights[edge_ids, :, None, None]
                    logu = plan.log_complements[edge_ids, :, None, None]
                    reference = value[plan.canonical[c]][None, None]
                    modified = attenuate_log_evidence(value[None], reference, logw, logu)
                    delta = modified.logsumexp(1) / policy.tau - base[None]
                    # FP rounding must never turn an attenuation into promotion.
                    delta = delta.clamp_max(0.)
                    active_fraction += ((value[None] > reference) & (logw < 0)).sum()
                    active_elements += modified.numel()
                    for i, edge in enumerate(edge_ids):
                        changes[edge] = delta[i]
            dense = F.interpolate(torch.stack(rows)[None], (336, 336), mode='bilinear',
                                  align_corners=False)[0].float()
            if treatment and ne:
                delta_dense = F.interpolate(torch.stack(changes)[None], (336, 336),
                                            mode='bilinear', align_corners=False)[0].float()
        top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
        output[:, top:top+ah, left:left+aw] += dense[:, :ah, :aw]
        if treatment and ne:
            corrections[:, top:top+ah, left:left+aw] += delta_dense[:, :ah, :aw]
        count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()):
        raise RuntimeError('Wide coverage gap.')
    output /= count[None]
    if treatment:
        corrections /= count[None]
    return output, corrections, dict(excess_eligible_fraction=float(active_fraction/max(active_elements, 1)),
        sparse_pair_fields=ne if treatment else 0,
        pair_field_mib=ne*h*w*4/1048576 if treatment else 0.)


def conditional_delta(fields, pairs, plan, classes):
    """Select two directed contracts per patch, never patch x alias x all rivals."""
    p = len(pairs)
    ids = plan.lookup[pairs, pairs.flip(-1)]
    delta = fields.new_zeros(p, classes)
    if not len(plan.edges):
        return delta, ids >= 0
    values = fields.gather(1, ids.clamp_min(0)) * (ids >= 0)
    delta.scatter_(1, pairs, values)
    return delta, ids >= 0


@torch.inference_mode()
def from_source(source, bank, query, plan, policy, methods=METHODS, verify=False):
    from eval_geometry_vip_reliability import sample_broad
    from .development_readout import coupled
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window

    treatment = PRIMARY in methods
    broad, corrections, stats = wide_fields(source, query, plan, policy, treatment)
    if verify:
        from eval_development_readout import wide_scores
        inherited = wide_scores(source, query, policy.tau, policy.tem)
        if not torch.equal(broad, inherited):
            raise RuntimeError('Shared wide scoring differs from inherited baseline.')
        if treatment and bool((corrections > 0).any()):
            raise RuntimeError('Contract attenuation raised a wide score.')
        stats['inherited_wide_scores_exact'] = True
    h, w = source['size']
    text = F.normalize(bank.features.float(), dim=-1)
    blend = torch.from_numpy(hann_blend_window(512)).to(text.device)
    affected = torch.zeros((), device=text.device)
    dose = torch.zeros((), device=text.device)
    patches = 0
    results = {}
    with ExitStack() as stack:
        accumulators = {name: stack.enter_context(DeviceProbabilityAccumulator(bank.class_count, h, w, text.device))
                        for name in methods}
        for tile in source['local']:
            local_alias = tile['features'][policy.strength].float() @ text.T
            local = alias_class_scores(local_alias, bank.parent_indices, bank.class_count) / policy.temperature
            top, left = tile['top'], tile['left']
            wide = sample_broad(broad, top, left, h, w).reshape_as(local)
            baseline = coupled(local, wide, tile['operator'], policy.coupling)
            values = {BASE: baseline}
            if treatment:
                pairs = baseline.topk(2, -1).indices
                if len(plan.edges):
                    sampled = sample_broad(corrections, top, left, h, w).reshape(len(local), -1)
                else:
                    sampled = local.new_empty(len(local), 0)
                delta, eligible = conditional_delta(sampled, pairs, plan, bank.class_count)
                # Reconstruct with the same expression/order as the off arm.
                values[PRIMARY] = coupled(local, wide+delta, tile['operator'], policy.coupling)
                affected += (delta.abs().sum(-1) > 0).sum()
                dose += delta.abs().sum()
                patches += len(local)
            for name in methods:
                dense = F.interpolate(values[name].T.reshape(1, bank.class_count, 32, 32),
                                      (512, 512), mode='bilinear', align_corners=False)[0]
                ah, aw = min(512, h-top), min(512, w-left)
                accumulators[name].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
        for name in methods:
            a = accumulators[name]
            if not bool((a.normalizer > 0).all()):
                raise RuntimeError('Local coverage gap.')
            probability = F.interpolate((a.probabilities/a.normalizer[None])[None], source['output_size'],
                                        mode='bilinear', align_corners=False)[0]
            if not bool(torch.isfinite(probability).all()):
                raise RuntimeError('Nonfinite contract probability.')
            results[name] = probability.argmax(0).cpu().numpy()
    stats.update(geometry_encodings=len(source['local']), wide_encodings=len(source['wide']),
        fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0,
        alias_slots_per_class=20, altered_patch_fraction=float(affected/max(patches, 1)),
        mean_absolute_wide_delta=float(dose/max(patches*bank.class_count, 1)),
        semantic_contract_edges=len(plan.edges), class_square_alias_tensor=False)
    return results, stats


@torch.inference_mode()
def predict(image, geometry, vip, bank, query, plan, policy, methods=METHODS, verify=False):
    from eval_development_readout import observations

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared off/on treatment required.')
    source = observations(image, geometry, vip, (policy.strength,), wide_policy=policy.wide_policy)
    return from_source(source, bank, query, plan, policy, methods, verify)
