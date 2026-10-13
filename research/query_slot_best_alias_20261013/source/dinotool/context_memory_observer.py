"""Capacity-limited, mass-conserving whole-image semantic context memory.

Patch-to-slot assignments P define a symmetric read P diag(P^T 1)^-1 P^T.
The implementation stores only P and slot values, never the N by N read.
Clustering/factorized pooling are established operations, not a priority claim.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


IMPLEMENTATION = 'geometry-context-memory-v1-20261013'


@dataclass(frozen=True)
class ContextMemoryConfig:
    capacity: int = 32
    temperature: float = .1
    iterations: int = 5
    center: bool = True
    hard: bool = False


@torch.inference_mode()
def context_assignments(raw, config=ContextMemoryConfig(), return_logits=False):
    """Deterministic spherical Lloyd slots, followed by soft assignment.

    A farthest-point initialization uses only the image's raw descriptors.
    Ties are resolved by token order. Empty Lloyd slots retain their centers.
    Constant centered inputs use their uncentered descriptors.
    """
    if (raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] < 1
            or config.capacity < 1 or config.iterations < 1 or config.temperature <= 0):
        raise ValueError('Nonempty batch-one raw tokens and positive memory settings required.')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        data = raw[0].float()
        if config.center:
            centered = data-data.mean(0, keepdim=True)
            data = torch.where(centered.norm(dim=-1, keepdim=True) > 1e-7, centered, data)
        data = F.normalize(data, dim=-1)
        n, width = data.shape
        k = min(config.capacity, n)
        chosen = torch.zeros(n, device=data.device, dtype=torch.bool)
        distance = torch.full((n,), torch.inf, device=data.device)
        indices = []
        next_index = data.norm(dim=-1).argmax()
        for _ in range(k):
            indices.append(next_index)
            chosen[next_index] = True
            distance = torch.minimum(distance, (1-data @ data[next_index]).clamp_min(0))
            next_index = distance.masked_fill(chosen, -1).argmax()
        centers = data[torch.stack(indices)]
        for _ in range(config.iterations):
            owner = (data @ centers.T).argmax(-1)
            total = torch.zeros((k, width), device=data.device)
            total.index_add_(0, owner, data)
            count = torch.bincount(owner, minlength=k)
            proposed = F.normalize(total, dim=-1)
            centers = torch.where((count > 0)[:, None], proposed, centers)
        similarity = data @ centers.T
        if config.hard:
            assignment = F.one_hot(similarity.argmax(-1), k).float()
        else:
            assignment = (similarity/config.temperature).softmax(-1)
        return (assignment[None], (similarity/config.temperature)[None]) if return_logits else assignment[None]


def memory_read(assignment, values):
    """Read frozen semantic Values through slots in O(N*K*D) work.

    For row-stochastic P, the implicit relation is symmetric, positive
    semidefinite and doubly stochastic, including empty hard-assignment slots.
    It preserves spatially constant Values and the global Value sum in exact
    arithmetic. These are algebraic properties, not correctness guarantees.
    """
    if (assignment.ndim != 3 or values.ndim != 3
            or assignment.shape[:2] != values.shape[:2]):
        raise ValueError('Matching [B,N,K] assignments and [B,N,D] values required.')
    with torch.autocast(device_type=values.device.type, enabled=False):
        p, v = assignment.float(), values.float()
        mass = p.sum(1)
        slots = (p.transpose(1, 2) @ v)/mass.clamp_min(1e-20)[..., None]
        return (p @ slots).to(values.dtype)


def regression_factors(assignment, ridge=.01):
    """Mean-anchored ridge reconstruction of overlapping semantic slots.

    Unlike convex prototype pooling, the induced donor coefficients may be
    negative. The mean mode has eigenvalue1 and contrast eigenvalues lie in
    [0,1]. Ridge regression itself is established machinery.
    """
    with torch.autocast(device_type=assignment.device.type, enabled=False):
        p = assignment.float()
        contrast = p-p.mean(1, keepdim=True)
        gram = contrast.transpose(1,2) @ contrast
        k = gram.shape[-1]
        scale = gram.diagonal(dim1=-2, dim2=-1).sum(-1)/k
        penalty = (ridge*scale).clamp_min(1e-8)
        system = gram+penalty[:, None, None]*torch.eye(k, device=p.device)[None]
        lift = torch.cholesky_solve(contrast.transpose(1,2), torch.linalg.cholesky(system))
        return contrast, lift


def regression_read(factors, values):
    with torch.autocast(device_type=values.device.type, enabled=False):
        contrast, lift = factors
        v = values.float()
        return (v.mean(1, keepdim=True)+contrast @ (lift @ v)).to(values.dtype)


def independent_factor_read(factors, values):
    """A2/factorized Gaussian-style independent normalization control."""
    with torch.autocast(device_type=values.device.type, enabled=False):
        distribute, collect = factors
        return (distribute @ (collect.transpose(1,2) @ values.float())).to(values.dtype)


@torch.inference_mode()
def read_context_head(head, raw, assignment, read_function=memory_read):
    """Two frozen Value/projection/residual/MLP blocks using one memory P."""
    current = raw
    for block in head.blocks:
        normalized = block.norm1(current)
        linear = block.attn.qkv
        width = current.shape[-1]
        bias = None if linear.bias is None else linear.bias[2*width:]
        value = F.linear(normalized, linear.weight[2*width:], bias)
        read = read_function(assignment, value)
        current = current+block.ls1(block.attn.proj_drop(block.attn.proj(read)))
        current = current+block.ls2(block.mlp(block.norm2(current)))
    return F.normalize(head.linear_projection(head.ln_final(current)).float(), dim=-1)


@torch.inference_mode()
def read_response_before_head(head, tokens, assignment):
    """Transport frozen native attention increments before each frozen MLP."""
    if tokens.ndim != 3 or assignment.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError('Batch-one tokens and assignments required.')
    prefix = tokens.shape[1] - assignment.shape[1]
    if prefix < 1 or assignment.shape[1] < 1:
        raise ValueError('At least one prefix and patch token required.')
    current = tokens
    for block in head.blocks:
        width = current.shape[-1]
        encoded = block.attn.qkv(block.norm1(current))
        heads = block.attn.num_heads
        query, key, value = tuple(part.transpose(1, 2) for part in
            encoded.reshape(1, current.shape[1], 3, heads, width // heads).unbind(2))
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        attended = native @ value
        merged = attended.transpose(1, 2).reshape(current.shape)
        increment = block.ls1(block.attn.proj_drop(block.attn.proj(merged)))
        mixed = torch.cat((increment[:, :prefix], memory_read(assignment, increment[:, prefix:])), dim=1)
        current = current.float() + mixed.float()
        current = current + block.ls2(block.mlp(block.norm2(current))).float()
    return F.normalize(head.linear_projection(head.ln_final(current)).float(), dim=-1)


class ContextMemoryObserver:
    """One complete low-resolution image; no VIP source or salience required."""
    def __init__(self, backbone, config=ContextMemoryConfig()):
        if backbone.patch_size != 16 or backbone.model.training:
            raise ValueError('Pass a frozen evaluation DINO.text backbone with patch size16.')
        if any(p.requires_grad for p in backbone.model.parameters()):
            raise ValueError('The observer requires frozen weights.')
        self.backbone, self.config = backbone, config
        self.device = backbone.device

    @torch.inference_mode()
    def raw_patch_features(self, rgb):
        if rgb.ndim != 3 or rgb.shape[0] != 3 or any(x % 16 for x in rgb.shape[-2:]):
            raise ValueError('Expected [3,H,W] RGB on the patch lattice.')
        self.backbone._validate_rgb(rgb[None])
        visual = self.backbone.model.visual_model
        image = (rgb[None].to(self.device)-self.backbone._imagenet_mean)/self.backbone._imagenet_std
        image = image.to(next(visual.parameters()).dtype)
        with self.backbone._autocast():
            _, raw, _ = visual.get_backbone_features(image)
        return raw

    @torch.inference_mode()
    def raw_context_tokens(self, rgb):
        """Return prefix plus patch tokens for response-order readouts."""
        if rgb.ndim != 3 or rgb.shape[0] != 3 or any(x % 16 for x in rgb.shape[-2:]):
            raise ValueError('Expected [3,H,W] RGB on the patch lattice.')
        self.backbone._validate_rgb(rgb[None])
        visual = self.backbone.model.visual_model
        image = (rgb[None].to(self.device)-self.backbone._imagenet_mean)/self.backbone._imagenet_std
        image = image.to(next(visual.parameters()).dtype)
        with self.backbone._autocast():
            cls, raw, registers = visual.get_backbone_features(image)
        return torch.cat((cls[:, None], registers, raw), dim=1), raw

    @torch.inference_mode()
    def project(self, raw):
        assignment = context_assignments(raw, self.config)
        with self.backbone._autocast():
            return read_context_head(self.backbone.model.visual_model.head, raw, assignment)

    @torch.inference_mode()
    def crop_patch_features(self, rgb):
        return self.project(self.raw_patch_features(rgb))


class CompactContextScorer:
    """Template-mean dot products and vectorized class LSE, without salience.

    Template averaging commutes with cosine dot products in real arithmetic;
    mixed-precision rounding differs from a template-level implementation.
    The inherited 40*cosine scale, tau and alias-count offsets are retained.
    """
    def __init__(self, salience=False):
        self.cache, self.salience = {}, salience

    @torch.inference_mode()
    def __call__(self, source, query, tau=1., tem=None):
        key = id(query)
        if key not in self.cache:
            parents = query.parents.detach().cpu().tolist()
            if set(parents) != set(range(len(query.class_names))):
                raise ValueError('Every scored class requires aliases.')
            text = query.features.float().mean(1)
            self.cache[key] = (query, text, len(query.class_names), F.normalize(text, dim=-1),
                               torch.bincount(query.parents, minlength=len(query.class_names)))
        _, text, classes, normalized_text, alias_counts = self.cache[key]
        h, w = source['wide_size']
        single = (len(source['wide']) == 1 and source['wide'][0]['top'] == 0
                  and source['wide'][0]['left'] == 0 and source['wide'][0]['ah'] == h
                  and source['wide'][0]['aw'] == w)
        output = None if single else torch.zeros(classes, h, w, device=text.device)
        count = None if single else torch.zeros(h, w, device=text.device)
        for crop in source['wide']:
            features = crop['features']
            with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16,
                                enabled=features.is_cuda):
                aliases = ((features @ text.T)[0]*40.).float()*tau
                if self.salience:
                    # Attributed incumbent image-conditioned alias weights;
                    # retained as a scoring factor, not a new attention rule.
                    global_feature = F.normalize(features.mean(1), dim=-1)
                    s = ((global_feature @ normalized_text.T)[0]/tem).float()
                    maximum = s.new_full((classes,), -torch.inf)
                    maximum.scatter_reduce_(0, query.parents, s, reduce='amax', include_self=True)
                    mass_s = (s-maximum[query.parents]).exp()
                    totals = s.new_zeros(classes).scatter_add_(0, query.parents, mass_s)
                    weights = mass_s*alias_counts[query.parents]/totals[query.parents]
                    aliases = aliases*weights[None]
            indices = query.parents[None].expand(aliases.shape[0], -1)
            maxima = aliases.new_full((aliases.shape[0], classes), -torch.inf)
            maxima.scatter_reduce_(1, indices, aliases, reduce='amax', include_self=True)
            mass = torch.zeros_like(maxima)
            mass.scatter_add_(1, indices, (aliases-maxima.gather(1, indices)).exp())
            logits = (maxima+mass.log())/tau
            grid = crop.get('grid_hw', (21, 21))
            size = crop.get('output_hw', (grid[0]*16, grid[1]*16))
            dense = F.interpolate(logits.T.reshape(1, classes, *grid), size,
                                  mode='bilinear', align_corners=False)[0]
            top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
            if single:
                return dense[:, :ah, :aw]
            output[:, top:top+ah, left:left+aw] += dense[:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
        return output/count[None]
