"""Fixed per-task configurations of the retained Geometry/wide reconstruction."""
from dataclasses import dataclass, asdict, replace

import numpy as np
import torch
import torch.nn.functional as F

from .device_probability_accumulator import DeviceProbabilityAccumulator
from .inference import hann_blend_window
from .packed_alias_readout import PackedAliasReadout
from .matched_readout_controls import run_head


IMPLEMENTATION = 'geometry-static-natural-configuration-search-v1-20261007'
THRESHOLDS = (0., .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .6, .7, .8, .9, .95)
BIASES = (0., -1., 1., -2., 2., -4., 4.)


@dataclass(frozen=True)
class StaticProfile:
    bank: str
    strength: object = 2.
    coupling: float = 1.
    temperature: float = .07
    tau: float = 1.
    tem: float = 1.
    wide_policy: str = 'long448'
    background_rule: str = 'retained'

    def record(self):
        return asdict(self)


def projected(head, prepared, strength):
    if strength == 'original':
        return prepared.geometry_projected
    if strength not in (.5, 1., 2., 3., 4.):
        raise ValueError('Undeclared fixed Geometry read strength.')
    return run_head(head, prepared.backbone_tokens,
        prepared.backbone_tokens[:, prepared.prefix_tokens:],
        prepared.geometry_patch_conditional * strength, prepared.prefix_tokens,
        'Geometry_BlockPrefix', prepared.block_index)[0]


def stage_profiles(stage, selected, banks, background, has_residual):
    """Finite coordinate search; the same menu is fixed before seeing new scores."""
    candidates = [selected]
    if stage in ('words', 'revisit_words'):
        candidates += [replace(selected, bank=name) for name in banks]
        if background is not None:
            rules = ('retained', 'union_max') + (('residual_max', 'residual_protected') if has_residual else ())
            candidates += [replace(selected, background_rule=rule) for rule in rules]
    elif stage == 'readout':
        candidates += [replace(selected, strength=s, coupling=g)
            for s in ('original', .5, 1., 2., 3., 4.) for g in (.25, .5, .75, 1., 1.5, 2., 3.)]
    elif stage == 'calibration':
        candidates += [replace(selected, tau=t, tem=m) for t in (.5, 1., 2.2, 3., 4., 6.) for m in (.3, 1., 3., 10.)]
        candidates += [replace(selected, temperature=t) for t in (.03, .05, .07, .1, .14)]
        candidates += [replace(selected, wide_policy=p) for p in ('long448', 'natural_short336_cap672')]
    else:
        raise ValueError('Undeclared search stage.')
    return tuple(dict.fromkeys(candidates))


class StaticReader:
    """Inference uses a supplied fixed profile; it never selects a task parameter."""
    def __init__(self, banks, queries, background=None, residual_features=None):
        self.banks, self.queries, self.background = banks, queries, background
        self.text = {k: F.normalize(v.features.float(), dim=-1) for k, v in banks.items()}
        self.scorers = {k: PackedAliasReadout(v) for k, v in banks.items()}
        self.members = {k: torch.nonzero(v.parent_indices == background, as_tuple=False).flatten()
                        for k, v in banks.items()} if background is not None else {}
        self.residual_features = residual_features
        self.residual_text = None if residual_features is None else F.normalize(residual_features.float().mean(1), dim=-1)
        self.residual_readers = {}
        if residual_features is not None:
            from .residual_rival_weight import ResidualRivalWeight
            from .taxonomy_readout import canonical_features
            self.residual_readers = {k: ResidualRivalWeight(self.residual_text, canonical_features(v)) for k, v in banks.items()}
        self.blend = torch.from_numpy(hann_blend_window(512)).to(next(iter(banks.values())).features.device)

    @torch.inference_mode()
    def probabilities(self, source, profile, cache):
        import eval_development_readout as base
        from eval_geometry_vip_reliability import sample_broad
        bank = self.banks[profile.bank]
        wkey = ('wide', profile.bank, profile.tau, profile.tem)
        if wkey not in cache:
            cache[wkey] = base.wide_scores(source, self.queries[profile.bank], profile.tau, profile.tem)
        original_broad = cache[wkey]
        broad = original_broad
        if profile.background_rule.startswith('residual_'):
            if self.residual_text is None or self.background is None:
                raise ValueError('Residual ontology required.')
            rkey = ('residual_wide',)
            if rkey not in cache:
                from eval_residual_ontology import residual_wide
                cache[rkey] = residual_wide(source, self.residual_features)['max'][0]
            broad = broad.clone()
            broad[self.background] = cache[rkey]
        h, w = source['size']
        with DeviceProbabilityAccumulator(bank.class_count, h, w, self.blend.device) as accumulator:
            for i, tile in enumerate(source['local']):
                key = ('local', profile.bank, profile.strength, profile.background_rule, i)
                if key not in cache:
                    alias = tile['features'][profile.strength].float() @ self.text[profile.bank].T
                    local = self.scorers[profile.bank].uniform_reference(alias)
                    if profile.background_rule == 'union_max' and self.background is not None:
                        local[:, self.background] = alias[:, self.members[profile.bank]].amax(-1)
                    if profile.background_rule == 'residual_max':
                        local[:, self.background] = (tile['features'][profile.strength].float() @ self.residual_text.T).amax(-1)
                    cache[key] = local
                local = cache[key] / profile.temperature
                top, left = tile['top'], tile['left']
                wide = sample_broad(broad, top, left, h, w).reshape_as(local)
                if profile.background_rule == 'residual_protected':
                    previous = sample_broad(original_broad, top, left, h, w).reshape_as(local)
                    foreground = local.double() + profile.coupling * (tile['operator'].double() @ (previous.double() - local.double()))
                    foreground[:, self.background] = -torch.inf
                    cosine = tile['features'][profile.strength].float() @ self.residual_text.T
                    local = local.clone()
                    local[:, self.background] = self.residual_readers[profile.bank].score(cosine, foreground.argmax(-1)) / profile.temperature
                logits = local.double() + profile.coupling * (tile['operator'].double() @ (wide.double() - local.double()))
                dense = F.interpolate(logits.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                ah, aw = min(512, h - top), min(512, w - left)
                accumulator.add(dense[:, :ah, :aw].softmax(0).float(), self.blend[:ah, :aw], left, top)
            probability = F.interpolate((accumulator.probabilities / accumulator.normalizer[None])[None],
                source['output_size'], mode='bilinear', align_corners=False)[0]
        if not bool(torch.isfinite(probability).all()):
            raise RuntimeError('Nonfinite static readout.')
        return probability


def rank(histograms, candidates, background):
    from .development_readout import threshold_matrices, score
    support = histograms[0, 0].sum(0).sum(1) > 0
    rows = []
    biases = BIASES if background is not None else (0.,)
    for i, candidate in enumerate(candidates):
        for j, bias in enumerate(biases):
            for threshold, cm in threshold_matrices(histograms[i, j], background, THRESHOLDS).items():
                rows.append(dict(profile=candidate.record(), background_bias=bias,
                    background_threshold=threshold, development_miou=score(cm, support),
                    profile_index=i, bias_index=j))
    rows.sort(key=lambda v: -v['development_miou'])
    best = rows[0]
    best['confusion_matrix'] = threshold_matrices(histograms[best['profile_index'], best['bias_index']],
        background, THRESHOLDS)[best['background_threshold']].tolist()
    return rows
