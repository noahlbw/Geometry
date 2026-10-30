"""Opt-in Geov2 wide/readout candidates with an unchanged DonorBefore local.

The shared-backbone option changes only the wide model's weight/input rounding.
The direct reader skips a dense wide-score canvas, so it is a model candidate,
not a claim of pixel-equivalent execution.  The streamed quadratic correction
uses a factor solve against C right-hand sides instead of forming N by N H.
"""
from dataclasses import dataclass, replace
import json
import os
import time

import torch
import torch.nn.functional as F

from .context_memory_observer import (
    context_assignments, read_context_head, regression_factors, regression_read,
)
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .query_slot_memory import _sample_global, query_slot_features, slot_prototypes
from .anchored_slot_memory import anchored_slot_features


WIDE_READOUT_CHOICES = ('incumbent', 'shared_canvas', 'anchored_infra',
                        'anchored_infra_fast', 'anchored_infra_bf16',
                        'anchored_infra_shared_visual',
                        'optimized48')


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
    # Optional micro-batching of the independent 512px local crops. A value
    # of one is the frozen execution path; larger values only change kernel
    # scheduling and keep DonorBefore algebra per crop unchanged.
    local_batch_size: int = 1
    batch_wide: bool = False
    readout: str = 'legacy'
    batch_anchored: bool = False
    local_memory: bool = False
    compile_visual: bool = False
    compile_project: bool = False
    weight_dtype: str = 'none'


def wide_readout_config(name):
    """Return the two supported deployment candidates from one stable API.

    ``shared_canvas`` is the speed/memory path: it shares the frozen visual
    backbone storage but retains the incumbent QuerySlot canvas reader.
    ``anchored_infra`` uses the direct 24-slot patch reader with the frozen
    Anchored slot count, shares only the frozen backbone storage, and applies
    the factorized FP32 writer.  It is the execution-only deployment path for
    the final Anchored model: the wide head, slots, DonorBefore local path and
    equations are unchanged.  ``optimized48`` additionally uses the direct
    48-slot patch reader and is retained as a memory-oriented research
    candidate.  ``incumbent`` is represented by ``None`` so
    callers can keep the historical ``FrozenGeometry.predict`` path intact.
    """
    if name == 'incumbent':
        return None
    if name == 'shared_canvas':
        return WideReadoutConfig(direct=False)
    if name == 'anchored_infra':
        return WideReadoutConfig(
            shared_backbone=True, shared_visual=False, direct=True,
            capacity=24, fast_assignment=True, solve_dtype='float32',
            batch_wide=False, readout='anchored', batch_anchored=True)
    if name == 'anchored_infra_fast':
        # Execution-only candidate.  The local DonorBefore equations and the
        # Anchored residual read are unchanged; only independent fixed-size
        # crops are scheduled as a micro-batch and static visual kernels are
        # compiled after model construction.
        return WideReadoutConfig(
            shared_backbone=True, shared_visual=False, direct=True,
            capacity=24, fast_assignment=True, solve_dtype='float32',
            batch_wide=False, readout='anchored', batch_anchored=True,
            local_batch_size=4, compile_visual=True, compile_project=True)
    if name == 'anchored_infra_bf16':
        # Inference-only storage reduction.  It retains the independent wide
        # head and all Anchored/DonorBefore equations; only frozen Linear/Conv
        # parameter storage is converted after the backbone is wired.
        return WideReadoutConfig(
            shared_backbone=True, shared_visual=False, direct=True,
            capacity=24, fast_assignment=True, solve_dtype='float32',
            batch_wide=False, readout='anchored', batch_anchored=True,
            weight_dtype='bfloat16')
    if name == 'anchored_infra_shared_visual':
        return WideReadoutConfig(
            shared_backbone=True, shared_visual=True, direct=True,
            capacity=24, fast_assignment=True, solve_dtype='float32',
            batch_wide=False, readout='anchored', batch_anchored=True)
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
    weights = relation.to(dtype)
    weights.masked_fill_(~valid[None], 0.)
    weights.div_(weights.sum(-1, keepdim=True).clamp_min(1e-12))
    weights.masked_fill_(~valid[:, None], 0.)
    gram = weights.T @ weights
    # I + G differs from G only on the diagonal.  Updating the Gram
    # matrix in place avoids materializing a second N x N system matrix.
    gram.diagonal(dim1=-2, dim2=-1).add_(1.)
    return (torch.linalg.cholesky(gram), valid), None


@torch.inference_mode()
def factor_reconstruction_batch(relation, valid, dtype=torch.float64):
    """Batched equivalent of :func:`factor_reconstruction`.

    The per-crop matrices and masks are unchanged; batching only lets cuSOLVER
    schedule the independent Cholesky factorizations together.
    """
    if relation.ndim != 3 or valid.ndim != 2 or relation.shape[:2] != (valid.shape[0], valid.shape[1]):
        raise ValueError('Expected BxNxN relations and BxN validity masks.')
    weights = relation.to(dtype)
    weights = weights.masked_fill(~valid[:, None, :], 0.)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    weights = weights.masked_fill(~valid[:, :, None], 0.)
    gram = weights.transpose(-1, -2) @ weights
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None]
    factors = torch.linalg.cholesky(gram + eye)
    return (factors, valid), None


@torch.inference_mode()
def factor_correction(factors, difference):
    factor, valid = factors
    d = difference.to(factor.dtype)
    # (I+G)^-1 G d = d-(I+G)^-1 d. Invalid rows have G=0.
    return (d - torch.cholesky_solve(d, factor)).masked_fill(~valid[:, None], 0.)


@torch.inference_mode()
def _sample_global_batch(features, grid_hw, source_hw, positions, token_hw=(32, 32)):
    """Sample a bounded set of local token grids in one CUDA kernel.

    The frozen 512px crop normally gives a 32x32 grid.  Keeping the target
    grid explicit also makes this helper safe for the small edge crops used
    by the deployment iterator and avoids silently changing their token
    count when batching.
    """
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError('Expected [T,2] tile positions.')
    gh, gw = grid_hw
    h, w = source_hw
    th, tw = token_hw
    offsets_y = torch.arange(th, device=features.device, dtype=torch.float32) + .5
    offsets_x = torch.arange(tw, device=features.device, dtype=torch.float32) + .5
    yy = (positions[:, None, 0, None] + offsets_y[None, :, None] * 16) / h * 2 - 1
    xx = (positions[:, None, 1, None] + offsets_x[None, None, :] * 16) / w * 2 - 1
    grid = torch.stack((xx.expand(-1, th, tw), yy.expand(-1, th, tw)), -1)
    image = features.transpose(1, 2).reshape(1, -1, gh, gw)
    image = image.expand(len(positions), -1, -1, -1)
    sampled = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return sampled.permute(0, 2, 3, 1).reshape(len(positions), th*tw, -1)


class PatchClassScorer:
    """Inherited template-mean alias scores, evaluated only on patch grids."""
    def __init__(self, query, tau):
        self.query, self.tau = query, tau
        self.text = query.features.float().mean(1)
        self.parents = query.parents
        self.indices = self.parents[None]
        self.classes = len(query.class_names)

    @torch.inference_mode()
    def __call__(self, features):
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16,
                            enabled=features.is_cuda):
            aliases = ((features @ self.text.T)[0] * 40.).float() * self.tau
        indices = self.indices.expand(aliases.shape[0], -1)
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
        if config.readout not in ('legacy', 'anchored'):
            raise ValueError('Unknown wide readout.')
        if config.solve_dtype not in ('float32', 'float64'):
            raise ValueError('Unknown factor precision.')
        if config.weight_dtype not in ('none', 'bfloat16'):
            raise ValueError('Unknown visual weight precision.')
        self.model, self.config = model, config
        if config.shared_visual:
            model.vip.backbone.model.visual_model = model.geometry.backbone.model.visual_model
        elif config.shared_backbone:
            # Do not cast/mutate the local module. Keep the separate wide head
            # and the two existing text caches, while releasing duplicate ViT
            # parameters. Sharing is only an inference storage choice.
            wide_visual = model.vip.backbone.model.visual_model
            old_backbone = wide_visual.backbone
            shared_backbone = model.geometry.backbone.model.visual_model.backbone
            if old_backbone is not shared_backbone:
                wide_visual.backbone = shared_backbone
                # The old wide backbone is now unreachable.  Emptying the
                # allocator once here prevents its cached blocks from being
                # counted as resident memory for the first image.
                del old_backbone
                if model.geometry.device.type == 'cuda':
                    torch.cuda.empty_cache()
        self.scorer = PatchClassScorer(model.queries[model.profile.bank], model.profile.tau)
        from .anchored_infra import cast_anchored_visual_weights
        self.weight_storage = cast_anchored_visual_weights(model, config.weight_dtype)
        if config.compile_visual or config.compile_project:
            self._compile_static_kernels()

    def _compile_static_kernels(self):
        """Compile only shape-static inference kernels for the fast candidate.

        Compilation is deliberately opt-in and happens after the shared
        backbone wiring.  It does not change a module's weights or any
        Geov2 equation; it removes Python/module dispatch from repeated
        fixed-size crop calls.  The first compiled call is paid during the
        existing warm-up in the benchmark protocol.
        """
        compiler = getattr(torch, 'compile', None)
        if compiler is None:
            raise RuntimeError('anchored_infra_fast requires torch.compile support')
        model = self.model
        if self.config.compile_visual:
            visuals = [model.geometry.backbone.model.visual_model,
                       model.vip.backbone.model.visual_model]
            seen = set()
            for visual in visuals:
                if id(visual) in seen or getattr(visual, '_geov2_visual_compiled', False):
                    continue
                visual.get_backbone_features = compiler(
                    visual.get_backbone_features, mode='reduce-overhead',
                    fullgraph=False, dynamic=False)
                visual._geov2_visual_compiled = True
                seen.add(id(visual))
        if self.config.compile_project:
            project = getattr(model, 'compact_geometry', None)
            if project is not None and not getattr(project, '_geov2_project_compiled', False):
                project.project = compiler(project.project, mode='reduce-overhead',
                                           fullgraph=False, dynamic=False)
                project._geov2_project_compiled = True

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
        # Anchored residual readout needs the same normalized global raw
        # reference and spatial prior as anchored_slot_wide_observations.
        # They are compact (N*D and N*K) and replace the old per-tile source
        # retention; no model feature or slot count is changed.
        global_raw = F.normalize(raw.float(), dim=-1)
        global_prior = assignment.float()
        return dict(semantic=semantic, centers=centers, values=values,
                    global_raw=global_raw, global_prior=global_prior,
                    grid_hw=(ph//16, pw//16), wide_size=original)

    @torch.inference_mode()
    def local_memory_from_tiles(self, tiles, source_hw):
        """Build the compact slot memory from already-computed local tiles.

        The local DonorBefore crop, relation solve, and writer stay unchanged;
        only the redundant second RGB forward is removed. Overlapping local
        tokens are averaged on the native 16px grid before slot construction.
        """
        if not tiles:
            raise ValueError('At least one local tile is required.')
        h, w = source_hw
        gh, gw = h // 16, w // 16
        channels = int(tiles[0]['raw'].shape[-1])
        raw_grid = torch.zeros((1, gh, gw, channels), device=tiles[0]['raw'].device,
                               dtype=torch.float32)
        sem_grid = torch.zeros_like(raw_grid)
        count = torch.zeros((1, gh, gw, 1), device=raw_grid.device, dtype=torch.float32)
        for tile in tiles:
            top, left = tile['top'] // 16, tile['left'] // 16
            raw = tile['raw'].float().reshape(1, 32, 32, channels)
            sem = tile['features']['original'].float().reshape(1, 32, 32, channels)
            raw_grid[:, top:top+32, left:left+32] += raw
            sem_grid[:, top:top+32, left:left+32] += sem
            count[:, top:top+32, left:left+32] += 1.
        raw = F.normalize((raw_grid / count.clamp_min(1.)).reshape(1, gh*gw, channels), dim=-1)
        semantic = F.normalize((sem_grid / count.clamp_min(1.)).reshape(1, gh*gw, channels), dim=-1)
        observer = self.model.query_slot_observer
        assign_fn = wide_assignments if self.config.fast_assignment else context_assignments
        assignment = assign_fn(raw, replace(observer.config, capacity=self.config.capacity))
        centers, values = slot_prototypes(raw, assignment, semantic)
        return dict(semantic=semantic, centers=centers, values=values,
                    global_raw=raw, global_prior=assignment,
                    grid_hw=(gh, gw), wide_size=(h, w))

    @torch.inference_mode()
    def tile_wide_features(self, tile, memory, source_hw):
        m, c = self.model, self.config
        baseline = _sample_global(memory['semantic'], memory['grid_hw'], source_hw, tile['top'], tile['left'])
        raw_query = tile['raw']
        if c.readout == 'anchored':
            reference = _sample_global(memory['global_raw'], memory['grid_hw'], source_hw,
                                       tile['top'], tile['left'])
            prior = _sample_global(memory['global_prior'], memory['grid_hw'], source_hw,
                                  tile['top'], tile['left'])
            return anchored_slot_features(
                raw_query, reference, memory['centers'], memory['values'], baseline,
                prior, temperature=.1, beta=c.beta, mode='anchored')
        if c.readout != 'legacy':
            raise ValueError('Unknown wide readout: ' + str(c.readout))
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
    def tile_wide_features_batch(self, tiles, memory, source_hw):
        """Apply the frozen Anchored residual read to all local tiles."""
        if not tiles:
            return []
        if self.config.readout != 'anchored':
            return [self.tile_wide_features(tile, memory, source_hw) for tile in tiles]
        device = tiles[0]['raw'].device
        positions = torch.tensor([[tile['top'], tile['left']] for tile in tiles],
                                 device=device, dtype=torch.float32)
        query = torch.cat([tile['raw'] for tile in tiles], dim=0)
        side = int(query.shape[1] ** .5)
        if side * side != query.shape[1]:
            raise ValueError('Batched Anchored readout requires a square token grid.')
        token_hw = (side, side)
        baseline = _sample_global_batch(memory['semantic'], memory['grid_hw'], source_hw,
                                        positions, token_hw)
        reference = _sample_global_batch(memory['global_raw'], memory['grid_hw'], source_hw,
                                         positions, token_hw)
        prior = _sample_global_batch(memory['global_prior'], memory['grid_hw'], source_hw,
                                     positions, token_hw)
        centers = memory['centers'].expand(len(tiles), -1, -1)
        values = memory['values'].expand(len(tiles), -1, -1)
        return anchored_slot_features(
            query, reference, centers, values, baseline, prior,
            temperature=.1, beta=self.config.beta, mode='anchored')

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
        profile_timing = os.environ.get('GEOV2_PROFILE_TIMING') == '1'
        if profile_timing and image.device.type == 'cuda':
            torch.cuda.synchronize(image.device)
        t0 = time.perf_counter()
        memory = None if c.local_memory else self.wide_memory(image)
        if profile_timing and image.device.type == 'cuda':
            torch.cuda.synchronize(image.device)
        t1 = time.perf_counter()
        resized = resize_geometry(image)
        h, w = resized.shape[-2:]
        p = m.profile
        bank = m.banks[p.bank]
        reader = m.reader
        g0 = p.coupling if gain is None else gain
        with DeviceProbabilityAccumulator(bank.class_count, h, w, reader.blend.device) as accumulator:
            def reconstruct(relation, valid):
                return factor_reconstruction(relation, valid, getattr(torch, c.solve_dtype))
            tiles = list(_iter_local_observations(
                resized, (h, w), m.compact_geometry, ('original',),
                reconstruction=reconstruct, include_raw=True,
                batch_size=c.local_batch_size,
                reconstruction_batch=(
                    (lambda relation, valid: factor_reconstruction_batch(
                        relation, valid, getattr(torch, c.solve_dtype)))
                    if c.local_batch_size > 1 else None)))
            if c.local_memory:
                memory = self.local_memory_from_tiles(tiles, (h, w))
            if profile_timing and image.device.type == 'cuda':
                torch.cuda.synchronize(image.device)
            t2 = time.perf_counter()
            wide_features = (self.tile_wide_features_batch(tiles, memory, (h, w))
                             if c.batch_anchored else
                             [self.tile_wide_features(tile, memory, (h, w)) for tile in tiles])
            if profile_timing and image.device.type == 'cuda':
                torch.cuda.synchronize(image.device)
            t3 = time.perf_counter()
            profile_rows = []
            for tile, wide_features_tile in zip(tiles, wide_features):
                if profile_timing and image.device.type == 'cuda':
                    torch.cuda.synchronize(image.device)
                ts = time.perf_counter()
                wide = self.scorer(wide_features_tile[None] if wide_features_tile.ndim == 2 else wide_features_tile)
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
                if profile_timing and image.device.type == 'cuda':
                    torch.cuda.synchronize(image.device)
                te = time.perf_counter()
                if profile_timing:
                    profile_rows.append((te-ts)*1000.)
                del tile, wide_features_tile, wide, alias, local, correction, logits
            del tiles, wide_features
            result = restore_and_finalize(accumulator.probabilities / accumulator.normalizer[None],
                                        tuple(image.shape[-2:]), m.config['background_bias'],
                                        m.config['background_threshold'], reader.background)
            if profile_timing:
                print(json.dumps({'geov2_profile': {
                    'wide_memory_ms': (t1-t0)*1000.,
                    'local_tiles_ms': (t2-t1)*1000.,
                    'wide_route_ms': (t3-t2)*1000.,
                    'tile_score_write_ms': profile_rows}}, sort_keys=True), flush=True)
            return result
