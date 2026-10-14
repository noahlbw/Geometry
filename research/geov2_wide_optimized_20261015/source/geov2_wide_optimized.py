"""Opt-in Geov2 wide/readout candidates with an unchanged DonorBefore local.

The shared-backbone option changes only the wide model's weight/input rounding.
The direct reader skips a dense wide-score canvas, so it is a model candidate,
not a claim of pixel-equivalent execution.  The streamed quadratic correction
uses a factor solve against C right-hand sides instead of forming N by N H.
"""
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .context_memory_observer import (
    context_assignments, read_context_head, regression_factors, regression_read,
)
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .query_slot_memory import _sample_global, query_slot_features, slot_prototypes


WIDE_READOUT_CHOICES = ('incumbent', 'shared_canvas', 'optimized48')


@dataclass(frozen=True)
class WideReadoutConfig:
    shared_backbone: bool = True
    head: str = 'slots'
    beta: float = .5
    query_mode: str = 'raw'
    # Keeping the old canvas permits isolating backbone sharing from readout.
    direct: bool = True
    solve_dtype: str = 'float64'
    wide_long_edge: int = 0
    capacity: int = 24
    shared_visual: bool = False
    fast_assignment: bool = False
    regression_ridge: float = .01


def wide_readout_config(name):
    """Return the two supported deployment candidates from one stable API.

    ``shared_canvas`` is the speed/memory path: it shares the frozen visual
    backbone storage but retains the incumbent QuerySlot canvas reader.
    ``optimized48`` additionally uses the direct 48-slot patch reader and the
    streamed factorized writer.  ``incumbent`` is represented by ``None`` so
    callers can keep the historical ``FrozenGeometry.predict`` path intact.
    """
    if name == 'incumbent':
        return None
    if name == 'shared_canvas':
        return WideReadoutConfig(direct=False)
    if name == 'optimized48':
        return WideReadoutConfig(capacity=48, shared_visual=True, fast_assignment=True)
    raise ValueError(f'Unknown wide readout: {name}')


def attach_wide_readout(model, name):
    """Attach an explicitly selected candidate and return its forward method."""
    config = wide_readout_config(name)
    if config is None:
        return model.predict
    return OptimizedWideReadout(model, config).predict


@torch.inference_mode()
def wide_assignments(raw, config):
    """Same spherical slots, without indexing CUDA tensors by CUDA scalars.

    GPU scalar subscripts and bincount's dynamic output size can synchronize
    the host. Explicit gather/scatter retains the fixed farthest-point and
    Lloyd rules while keeping their indices on the GPU.
    """
    with torch.autocast(device_type=raw.device.type, enabled=False):
        data = raw[0].float()
        if config.center:
            centered = data-data.mean(0, keepdim=True)
            data = torch.where(centered.norm(dim=-1, keepdim=True)>1e-7, centered, data)
        data = F.normalize(data, dim=-1)
        n, width = data.shape
        k = min(config.capacity, n)
        chosen = torch.zeros(n, device=data.device, dtype=torch.bool)
        distance = torch.full((n,), torch.inf, device=data.device)
        indices=[]
        next_index=data.norm(dim=-1).argmax()
        for _ in range(k):
            indices.append(next_index)
            chosen.scatter_(0, next_index.reshape(1), True)
            point=data.index_select(0,next_index.reshape(1))[0]
            distance=torch.minimum(distance,(1-data@point).clamp_min(0))
            next_index=distance.masked_fill(chosen,-1).argmax()
        centers=data.index_select(0,torch.stack(indices))
        for _ in range(config.iterations):
            owner=(data@centers.T).argmax(-1)
            total=torch.zeros((k,width),device=data.device)
            total.index_add_(0,owner,data)
            occupied=torch.zeros(k,device=data.device,dtype=torch.bool)
            occupied.scatter_(0,owner,True)
            centers=torch.where(occupied[:,None],F.normalize(total,dim=-1),centers)
        similarity=data@centers.T
        assignment=(F.one_hot(similarity.argmax(-1),k).float() if config.hard
                    else (similarity/config.temperature).softmax(-1))
        return assignment[None]


@torch.inference_mode()
def factor_reconstruction(relation, valid, dtype=torch.float64):
    """Factor I+A^T A with exactly the incumbent masking/normalization."""
    weights = relation.to(dtype).masked_fill(~valid[None], 0.)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    weights = weights.masked_fill(~valid[:, None], 0.)
    gram = weights.T @ weights
    system = gram + torch.eye(len(valid), device=gram.device, dtype=gram.dtype)
    return (torch.linalg.cholesky(system), valid), None


@torch.inference_mode()
def factor_correction(factors, difference):
    factor, valid = factors
    d = difference.to(factor.dtype)
    # (I+G)^-1 G d = d-(I+G)^-1 d. Invalid rows have G=0.
    return (d - torch.cholesky_solve(d, factor)).masked_fill(~valid[:, None], 0.)


class PatchClassScorer:
    """Inherited template-mean alias scores, evaluated only on patch grids."""
    def __init__(self, query, tau):
        self.query, self.tau = query, tau
        self.text = query.features.float().mean(1)
        self.parents = query.parents
        self.classes = len(query.class_names)

    @torch.inference_mode()
    def __call__(self, features):
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16,
                            enabled=features.is_cuda):
            aliases = ((features @ self.text.T)[0] * 40.).float() * self.tau
        indices = self.parents[None].expand(aliases.shape[0], -1)
        maxima = aliases.new_full((aliases.shape[0], self.classes), -torch.inf)
        maxima.scatter_reduce_(1, indices, aliases, reduce='amax', include_self=True)
        mass = torch.zeros_like(maxima)
        mass.scatter_add_(1, indices, (aliases-maxima.gather(1, indices)).exp())
        return (maxima + mass.log()) / self.tau


class OptimizedWideReadout:
    """Wrap an explicitly loaded QuerySlot model; keep all local code intact."""
    def __init__(self, model, config=WideReadoutConfig()):
        if not model.efficient:
            raise ValueError('OptimizedWideReadout requires efficient=True.')
        if model.strengths != ('original',) or model.local_branch != 'donor_before':
            raise ValueError('The original DonorBefore local path is required.')
        if model.writeback != 'coupled':
            raise ValueError('OptimizedWideReadout requires the fixed coupled writer.')
        if model.profile.background_rule.startswith('residual_'):
            raise ValueError('Residual ontology is outside this first optimization.')
        if config.head not in ('slots', 'native', 'self', 'regression'):
            raise ValueError('Unknown wide head.')
        if config.query_mode not in ('raw', 'semantic', 'hybrid'):
            raise ValueError('Unknown query mode.')
        if config.solve_dtype not in ('float32', 'float64'):
            raise ValueError('Unknown factor precision.')
        self.model, self.config = model, config
        if config.shared_visual:
            model.vip.backbone.model.visual_model = model.geometry.backbone.model.visual_model
        elif config.shared_backbone:
            # Do not cast/mutate the local module. Keep the separate wide head
            # and the two existing text caches, while releasing duplicate ViT
            # parameters. Sharing is only an inference storage choice.
            model.vip.backbone.model.visual_model.backbone = model.geometry.backbone.model.visual_model.backbone
        self.scorer = PatchClassScorer(model.queries[model.profile.bank], model.profile.tau)

    @torch.inference_mode()
    def wide_memory(self, image):
        from .context_memory_deployment import memory_rgb
        m, config = self.model, self.config
        observer = m.query_slot_observer
        if config.wide_long_edge:
            from eval_matched_head_fov import resize_rgb
            rgb = resize_rgb(image, config.wide_long_edge)
            original = tuple(rgb.shape[-2:])
            rgb = F.pad(rgb, (0, (-rgb.shape[-1])%16, 0, (-rgb.shape[-2])%16))
        else:
            rgb, original = memory_rgb(image, wide_policy=m.profile.wide_policy)
        ph, pw = rgb.shape[-2:]
        if config.head == 'native':
            tokens, raw = observer.raw_context_tokens(rgb)
        else:
            raw = observer.raw_patch_features(rgb)
        assignment_function = wide_assignments if config.fast_assignment else context_assignments
        assignment = assignment_function(raw, replace(observer.config, capacity=config.capacity))
        head = observer.backbone.model.visual_model.head
        with observer.backbone._autocast():
            if config.head == 'native':
                semantic = F.normalize(head(tokens)[:, -raw.shape[1]:].float(), dim=-1)
            elif config.head == 'self':
                semantic = read_context_head(head, raw, assignment, lambda _p, value: value)
            elif config.head == 'regression':
                factors = regression_factors(assignment, ridge=config.regression_ridge)
                semantic = read_context_head(head, raw, factors, regression_read)
            else:
                semantic = read_context_head(head, raw, assignment)
        centers, values = slot_prototypes(raw, assignment, semantic)
        return dict(semantic=semantic, centers=centers, values=values,
                    grid_hw=(ph//16, pw//16), wide_size=original)

    @torch.inference_mode()
    def tile_wide_features(self, tile, memory, source_hw):
        m, c = self.model, self.config
        baseline = _sample_global(memory['semantic'], memory['grid_hw'], source_hw, tile['top'], tile['left'])
        raw_query = tile['raw']
        semantic_query = tile['features']['original'][None]
        if c.query_mode == 'semantic':
            query, keys = semantic_query, memory['values']
            extra = {}
        elif c.query_mode == 'hybrid':
            query, keys = raw_query, memory['centers']
            extra = dict(secondary_query=semantic_query, secondary_centers=memory['values'], hybrid_weight=.5)
        else:
            query, keys, extra = raw_query, memory['centers'], {}
        return query_slot_features(query, keys, memory['values'], baseline,
                                   beta=c.beta, temperature=.1, **extra)

    @torch.inference_mode()
    def predict(self, image, *, gain=None):
        m, c = self.model, self.config
        if not c.direct:
            if c.head != 'slots' or c.query_mode != 'raw' or c.beta != .5:
                raise ValueError('Canvas control currently requires the incumbent slot reader.')
            if gain is None:
                return m.predict(image)
            from dataclasses import replace
            previous = m.profile
            m.profile = replace(previous, coupling=gain)
            try:
                return m.predict(image)
            finally:
                m.profile = previous
        from .bounded_physical_coupling import resize_geometry
        from .compact_deployment import _iter_local_observations
        from .efficient_deployment import restore_and_finalize, write_dense
        memory = self.wide_memory(image)
        resized = resize_geometry(image)
        h, w = resized.shape[-2:]
        p = m.profile
        bank = m.banks[p.bank]
        reader = m.reader
        g0 = p.coupling if gain is None else gain
        with DeviceProbabilityAccumulator(bank.class_count, h, w, reader.blend.device) as accumulator:
            def reconstruct(relation, valid):
                return factor_reconstruction(relation, valid, getattr(torch, c.solve_dtype))
            for tile in _iter_local_observations(resized, (h, w), m.compact_geometry, ('original',),
                                                 reconstruction=reconstruct, include_raw=True):
                wide_features = self.tile_wide_features(tile, memory, (h,w))
                wide = self.scorer(wide_features)
                alias = tile['features']['original'].float() @ reader.text[p.bank].T
                local = m.efficient_local_scorers[p.bank].uniform_reference(alias)
                if p.background_rule == 'union_max' and reader.background is not None:
                    local[:, reader.background] = alias[:, reader.members[p.bank]].amax(-1)
                local = local / p.temperature
                correction = factor_correction(tile['operator'], wide.double()-local.double())
                logits = local.double() + g0*correction
                top, left = tile['top'], tile['left']
                ah, aw = min(512, h-top), min(512, w-left)
                write_dense(accumulator, logits, bank.class_count, ah, aw, reader.blend, left, top)
                del tile, wide_features, wide, alias, local, correction, logits
            return restore_and_finalize(accumulator.probabilities / accumulator.normalizer[None],
                                        tuple(image.shape[-2:]), m.config['background_bias'],
                                        m.config['background_threshold'], reader.background)
