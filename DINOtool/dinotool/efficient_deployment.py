"""Optional execution backend; frozen words, views and readout equations stay fixed.

The original compact backend remains available for paired verification. No weight,
alias admission, calibration or resolution search is performed by this module.
"""
import torch
import torch.nn.functional as F

from .compact_deployment import CachedWideScorer, CompactGeometry
from .finite_vip_observer import FiniteVIPObserver
from .tcpr import geometry_relation
from .vip_official_adapter import VIPSettings


class EfficientGeometry(CompactGeometry):
    """Cache the static spatial term; keep the original floating point order."""
    def __init__(self, geometry):
        super().__init__(geometry)
        self.spatial = {}

    @torch.inference_mode()
    def prepare_image(self, rgb):
        from types import SimpleNamespace
        from .tcpr import _resolve_block_index
        backbone, config = self.backbone, self.config
        backbone._validate_rgb(rgb)
        normalized = (rgb.to(self.device, non_blocking=True) - backbone._imagenet_mean) / backbone._imagenet_std
        with backbone._autocast():
            visual = backbone.model.visual_model
            cls, raw, registers = visual.get_backbone_features(normalized)
            tokens = torch.cat((cls[:, None], registers, raw), dim=1)
            h, w = rgb.shape[-2] // 16, rgb.shape[-1] // 16
            if (h, w) not in self.spatial:
                y = torch.linspace(0., 1., h, device=raw.device)
                x = torch.linspace(0., 1., w, device=raw.device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                xy = torch.stack((yy, xx), -1).reshape(h*w, 2)
                self.spatial[h, w] = (xy[:, None]-xy[None, :]).square().sum(-1) / (2.*config.spatial_sigma*config.spatial_sigma)
            feature = F.normalize(raw.float(), dim=-1)
            logits = feature @ feature.transpose(-1, -2) / config.geometry_temperature - self.spatial[h, w]
            relation = geometry_relation(logits, raw, config.relation_policy).to(
                torch.get_autocast_dtype(self.device.type) if torch.is_autocast_enabled(self.device.type) else tokens.dtype)
        return SimpleNamespace(backbone_tokens=tokens, geometry_patch_conditional=relation,
            prefix_tokens=registers.shape[1]+1, block_index=_resolve_block_index(visual.head, config.head_block))


@torch.inference_mode()
def reconstruction(relation, valid):
    """Same FP64 solve, without an unused diagnostic residual GEMM/CPU sync."""
    weights = relation.double().masked_fill(~valid[None], 0.)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    weights = weights.masked_fill(~valid[:, None], 0.)
    gram = weights.T @ weights
    system = torch.eye(len(valid), device=relation.device, dtype=torch.float64) + gram
    operator = torch.cholesky_solve(gram, torch.linalg.cholesky(system))
    operator = (operator + operator.T)*.5
    return operator.masked_fill(~valid[:, None] | ~valid[None], 0.), None


class EfficientVIPObserver(FiniteVIPObserver):
    """Compute identical raw-patch proxy weights once for the two head blocks."""
    @torch.inference_mode()
    def crop_patch_features(self, rgb):
        if rgb.shape != (3, 336, 336):
            raise ValueError('Expected a336x336 RGB crop.')
        image = rgb[None].to(self.device)
        image = ((image-self.backbone._imagenet_mean)/self.backbone._imagenet_std).half()
        visual = self.backbone.model.visual_model
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            cls, raw, registers = visual.get_backbone_features(image)
            if registers.shape[1] != 4:
                raise ValueError('The pinned proxy requires four register tokens.')
            tokens = torch.cat((cls[:, None], registers, raw), 1)
            normalized = F.normalize(raw.permute(0, 2, 1), dim=1)
            similarity = torch.einsum('b c m,b c n->b m n', normalized, normalized)
            similarity = (similarity-similarity.mean()*1.5)*.5
            similarity[similarity <= 0.] = -torch.inf
            empty = ~torch.isfinite(similarity).any(-1)
            diagonal = torch.eye(raw.shape[1], device=raw.device, dtype=torch.bool)[None]
            similarity = similarity.masked_fill(empty[..., None] & diagonal, 0.)
            weights = None
            for block in visual.head.blocks:
                attention = block.attn
                normalized_tokens = block.norm1(tokens)
                prefix, patches = normalized_tokens[:, :5].clone(), normalized_tokens[:, 5:]
                batch, count, channels = patches.shape
                qkv = attention.qkv(patches).reshape(batch, count, 3, attention.num_heads, channels//attention.num_heads)
                query, key, value = [v.transpose(1, 2).squeeze(0).contiguous() for v in qkv.unbind(2)]
                if weights is None:
                    weights = similarity.to(query.dtype).unsqueeze(1).repeat(1, attention.num_heads, 1, 1)
                    weights = weights.reshape(batch*attention.num_heads, count, count).softmax(-1)
                value = value.reshape(batch*attention.num_heads, 21, 21, channels//attention.num_heads).permute(0, 3, 1, 2)
                value = F.interpolate(value, size=(21, 21), mode='bilinear', align_corners=False)
                value = value.permute(0, 2, 3, 1).reshape(batch*attention.num_heads, count, -1)
                read = (weights @ value).transpose(0, 1).contiguous().view(batch, -1, channels)
                read = torch.cat((prefix, attention.proj_drop(attention.proj(read))), 1)
                tokens = tokens+block.ls1(read)
                tokens = tokens+block.ls2(block.mlp(block.norm2(tokens)))
            projected = visual.head.linear_projection(visual.head.ln_final(tokens))
            result = F.normalize(projected[:, 5:].float(), dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite efficient observer features.')
        return result


class GroupedWideScorer(CachedWideScorer):
    """Batch classes with equal alias counts, without padding or changing mass."""
    def __init__(self):
        super().__init__()
        self.batches = {}

    @torch.inference_mode()
    def logits(self, features, query, settings):
        key = id(query)
        if key not in self.cache:
            parents = query.parents.detach().cpu().tolist()
            groups = [[i for i, p in enumerate(parents) if p == c] for c in range(len(query.class_names))]
            text = query.features.float()
            self.cache[key] = query, text, F.normalize(text.mean(1), dim=-1), groups
            counts = sorted(set(map(len, groups)))
            self.batches[key] = tuple((torch.tensor([c for c, g in enumerate(groups) if len(g) == n], device=text.device),
                torch.tensor([g for g in groups if len(g) == n], device=text.device)) for n in counts)
        _, text, mean, _ = self.cache[key]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            similarities = torch.einsum('bnd,mtd->bnmt', features, text).mean(-1)[0]
            patch_mean = F.normalize(features.mean(1), dim=-1)
            salience = ((patch_mean @ mean.T)[0] / settings.tem).float()
            aliases = (similarities*settings.logit_scale).T.reshape(-1, 21, 21)
            output = torch.empty(len(query.class_names), 21, 21, device=features.device)
            for classes, indices in self.batches[key]:
                weights = salience[indices].softmax(-1)
                scaled = aliases[indices]*(weights/weights.mean(-1, keepdim=True))[:, :, None, None]
                output[classes] = torch.logsumexp(settings.tau*scaled, dim=1)/settings.tau
            return F.interpolate(output[None], (336, 336), mode='bilinear', align_corners=False)[0].float()


class GroupedLocalScorer:
    """Keep each class's FP32 log-mean-exp, batching equal-size reductions."""
    def __init__(self, bank, temperature=.07):
        import math
        self.temperature=temperature
        parents=bank.parent_indices.cpu().tolist()
        groups=[[i for i,p in enumerate(parents) if p==c] for c in range(bank.class_count)]
        self.classes=bank.class_count
        self.batches=tuple((torch.tensor([c for c,g in enumerate(groups) if len(g)==n],device=bank.features.device),
            torch.tensor([g for g in groups if len(g)==n],device=bank.features.device),math.log(n))
            for n in sorted(set(map(len,groups))))

    @torch.inference_mode()
    def uniform_reference(self, alias):
        output=torch.empty(len(alias),self.classes,device=alias.device,dtype=alias.dtype)
        for classes,indices,log_count in self.batches:
            output[:,classes]=self.temperature*(torch.logsumexp(alias[:,indices]/self.temperature,-1)-log_count)
        return output


@torch.inference_mode()
def write_dense(accumulator, logits, classes, ah, aw, blend, left, top):
    """Release FP64 dense storage before stitching; bound FP64 softmax workspace."""
    dense = F.interpolate(logits.T.reshape(1, classes, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
    probability = torch.empty(classes, ah, aw, device=dense.device)
    for start in range(0, ah, 32):
        probability[:, start:start+32] = dense[:, start:min(ah, start+32), :aw].softmax(0).float()
    del dense
    accumulator.add(probability, blend[:ah, :aw], left, top)


@torch.inference_mode()
def finalize(probability, bias, threshold, background):
    """Apply original calibration in bounded spatial slabs, including reduction order."""
    from .development_readout import biased_probabilities
    _, h, w = probability.shape
    prediction = torch.empty(h, w, dtype=torch.uint8, device=probability.device)
    for start in range(0, h, 128):
        slab = biased_probabilities(probability[:, start:start+128], bias if background is not None else 0., background)
        confidence, labels = slab.max(0)
        if background is not None:
            labels = labels.masked_fill(confidence < threshold, background)
        prediction[start:start+128] = labels.to(torch.uint8)
    return prediction.cpu().numpy()


@torch.inference_mode()
def restore_and_finalize(normalized, output_size, bias, threshold, background):
    """Bound class-plane restoration when calibration needs no classwise sum."""
    finite=torch.ones((),device=normalized.device,dtype=torch.bool)
    for start in range(0,normalized.shape[0],16):
        finite &= torch.isfinite(normalized[start:start+16]).all()
    if not bool(finite):
        raise RuntimeError('Nonfinite normalized probability.')
    if background is not None and bias != 0.:
        restored=F.interpolate(normalized[None],output_size,mode='bilinear',align_corners=False)[0]
        return finalize(restored,bias,threshold,background)
    classes=normalized.shape[0]
    confidence=torch.full(tuple(output_size),-torch.inf,device=normalized.device)
    labels=torch.zeros(tuple(output_size),dtype=torch.uint8,device=normalized.device)
    for start in range(0,classes,16):
        dense=F.interpolate(normalized[None,start:start+16],output_size,mode='bilinear',align_corners=False)[0]
        value,index=dense.max(0)
        take=value>confidence
        labels=torch.where(take,(index+start).to(torch.uint8),labels)
        confidence=torch.maximum(confidence,value)
        del dense
    if background is not None:labels=labels.masked_fill(confidence<threshold,background)
    return labels.cpu().numpy()
