"""Query-anchored transport of frozen nonlinear semantic responses."""
from math import isqrt
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv, run_head
from .parallel_readout import structural_logits
from .tcpr import _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-query-anchored-semantic-response-v1-20261004"
PRIMARY = "Geometry_ResponseTransport"
BASELINES = ("Geometry", "Geometry_BlockPrefix", "SCLIP_Two", "VIPProxy_Two")
CANDIDATES = (PRIMARY, "Geometry_DonorBefore", "Geometry_FullDonor",
              "Geometry_QueryResponse", "Geometry_ResponseUniform",
              "Geometry_ResponseSpatial", "Geometry_SelfConditional")
METHODS = (*BASELINES, *CANDIDATES)


def settings():
    return {"implementation": IMPLEMENTATION,
            "primary_rule": "x+f(x)+G*(u_self+f(x+u_self)-f(x))",
            "f": "frozen ls2*MLP(norm2(x)); no final projection inside donor response",
            "self_read": "native special contribution and patch mass, conditional patch donor=self",
            "geometry_relation": "original cosine/.10, spatial sigma .25, both evolving head blocks",
            "prefix": "native prefix attention/MLP; prefix never transported",
            "precision": "frozen head bf16 AMP; response differences/transport/addition fp32",
            "controls": list(CANDIDATES[1:]),
            "before_control": "x+G*u_self+f(x+G*u_self); identical attention transport",
            "full_donor_control": "x+G*(u_self+f(x+u_self)); donor baseline not protected",
            "query_response_control": "x+u_G+f(x)+G*(f(x+u_G)-f(x)); original Geometry attention",
            "relation_controls": "uniform and same sigma Gaussian-only; identical local semantic donors",
            "text": "unchanged20 aliases/six RS templates/normalized LME .07",
            "view": "native512/128/Hann probability blending",
            "source": "one backbone, one selected head, one extra tokenwise MLP per block; no extra view/encoder",
            "novelty": "nonlinear-response ordering hypothesis; no first-in-literature claim"}


def transport(relation, value):
    with torch.autocast(device_type=value.device.type, enabled=False):
        return relation.float() @ value.float()


def self_attended(native, value, prefix):
    prefix_value = native[..., :prefix, :] @ value
    special = native[..., prefix:, :prefix] @ value[..., :prefix, :]
    mass = native[..., prefix:, prefix:].sum(-1, keepdim=True)
    return torch.cat((prefix_value, special+mass*value[..., prefix:, :]), dim=2)


def project_increment(block, current, attended):
    merged = attended.transpose(1, 2).reshape(current.shape)
    return block.ls1(block.attn.proj_drop(block.attn.proj(merged)))


def semantic_change(block, current, increment):
    baseline = block.ls2(block.mlp(block.norm2(current)))
    treated = block.ls2(block.mlp(block.norm2(current+increment)))
    return baseline, treated.float()-baseline.float(), treated


@torch.inference_mode()
def semantic_block(block, current, relation, prefix, mode, *, trace=False):
    query, key, value = qkv(block, current)
    native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
    attended = (_geometry_attended(native, relation, value, prefix)
                if mode == 'query_response' else self_attended(native, value, prefix))
    increment = project_increment(block, current, attended)
    diagnostics = {}
    if mode == 'before':
        mixed = transport(relation, increment[:, prefix:])
        mixed_all = torch.cat((increment[:, :prefix].float(), mixed), dim=1)
        after = current.float()+mixed_all
        result = after+block.ls2(block.mlp(block.norm2(after)))
    elif mode == 'self':
        after = current+increment
        result = after+block.ls2(block.mlp(block.norm2(after)))
    elif mode == 'full_donor':
        after = current+increment
        semantic = block.ls2(block.mlp(block.norm2(after)))
        delta = increment.float()+semantic.float()
        patch = current[:, prefix:].float()+transport(relation, delta[:, prefix:])
        result = torch.cat((after[:, :prefix]+semantic[:, :prefix], patch), dim=1)
    elif mode in ('response', 'query_response'):
        baseline, response, treated = semantic_change(block, current, increment)
        mixed_response = transport(relation, response[:, prefix:])
        mixed_increment = (increment[:, prefix:].float() if mode == 'query_response' else
                           transport(relation, increment[:, prefix:]))
        patch = current[:, prefix:].float()+mixed_increment+(baseline[:, prefix:].float()+mixed_response)
        prefix_state = current[:, :prefix]+increment[:, :prefix]+treated[:, :prefix]
        result = torch.cat((prefix_state, patch), dim=1)
        if trace:
            before = current[:, prefix:].float()+mixed_increment
            before_semantic = block.ls2(block.mlp(block.norm2(before))).float()
            commute = baseline[:, prefix:].float()+mixed_response-before_semantic
            diagnostics.update(response_norm=float(response[:, prefix:].norm(dim=-1).mean()),
                baseline_mlp_norm=float(baseline[:, prefix:].float().norm(dim=-1).mean()),
                mixed_response_norm=float(mixed_response.norm(dim=-1).mean()),
                nonlinear_order_gap_norm=float(commute.norm(dim=-1).mean()))
    else:
        raise ValueError('Unknown semantic response mode.')
    if trace:
        diagnostics['attention_increment_norm'] = float(increment[:, prefix:].float().norm(dim=-1).mean())
        diagnostics['native_patch_mass'] = float(native[..., prefix:, prefix:].float().sum(-1).mean())
        diagnostics['output_norm'] = float(result[:, prefix:].float().norm(dim=-1).mean())
    return result, diagnostics


def relation_controls(prepared):
    g = prepared.geometry_patch_conditional
    side = isqrt(g.shape[-1])
    if side*side != g.shape[-1]:
        raise ValueError('Fixed development uses square patch grids.')
    raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
    coordinates = torch.linspace(0., 1., side, device=raw.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    xy = torch.stack((yy, xx), -1).reshape(-1, 2)
    distance = (xy[:, None]-xy[None, :]).square().sum(-1)
    spatial = (-distance/(2*.25**2)).softmax(-1)[None].expand(g.shape[0], -1, -1).to(g.dtype)
    return {'geometry': g, 'uniform': torch.full_like(g, 1/g.shape[-1]), 'spatial': spatial}


@torch.inference_mode()
def read_head(head, prepared, method, *, cache=None, trace=False):
    if method not in METHODS or prepared.block_index != 1:
        raise ValueError('Declared method and the original two-block head required.')
    if method in BASELINES:
        return run_head(head, prepared.backbone_tokens, prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens, method, 1)
    relations = {'geometry': prepared.geometry_patch_conditional} if cache is None else cache
    relation_key = ('uniform' if method == 'Geometry_ResponseUniform' else
                    'spatial' if method == 'Geometry_ResponseSpatial' else 'geometry')
    if relation_key not in relations:
        relations = relation_controls(prepared)
    relation = relations[relation_key]
    mode = {'Geometry_DonorBefore': 'before', 'Geometry_FullDonor': 'full_donor',
            'Geometry_QueryResponse': 'query_response', 'Geometry_SelfConditional': 'self'}.get(method, 'response')
    current, diagnostics = prepared.backbone_tokens, {}
    for index, block in enumerate(head.blocks[:2]):
        current, row = semantic_block(block, current, relation, prepared.prefix_tokens, mode, trace=trace)
        diagnostics.update({f'block{index}_'+key: value for key, value in row.items()})
    projected = _finish_head(head, current, 1)
    return F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1), diagnostics


@torch.inference_mode()
def read_responses(geometry, prepared):
    head = geometry.backbone.model.visual_model.head
    features, diagnostics = {}, {}
    with geometry.backbone._autocast():
        cache = relation_controls(prepared)
        for method in METHODS:
            features[method], row = read_head(head, prepared, method, cache=cache, trace=True)
            diagnostics.update({method+'__'+key: value for key, value in row.items()})
    if not torch.equal(features['Geometry'], prepared.geometry_projected):
        raise RuntimeError('Original Geometry replay differs.')
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise RuntimeError('Nonfinite semantic response features.')
    return features, diagnostics


@torch.inference_mode()
def encode_single(backbone, rgb, method):
    backbone._validate_rgb(rgb)
    with backbone._autocast():
        visual = backbone.model.visual_model
        normalized = (rgb.to(backbone.device)-backbone._imagenet_mean)/backbone._imagenet_std
        cls, raw, registers = visual.get_backbone_features(normalized)
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        value_dtype = torch.bfloat16 if backbone.use_amp else raw.dtype
        relation = structural_logits(raw, rgb.shape[-2]//16, rgb.shape[-1]//16,
                                     temperature=.10, spatial_sigma=.25).softmax(-1).to(value_dtype)
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=registers.shape[1]+1,
                                   geometry_patch_conditional=relation, block_index=1)
        return read_head(visual.head, prepared, method)[0]
