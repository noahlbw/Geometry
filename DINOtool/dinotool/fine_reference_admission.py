"""Physically fine RGB references with the frozen semantic admission writer."""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .native_ownership_reader import query_permutation
from .semantic_reference_admission import (SemanticReferenceConfig,
    PRIMARY as PREVIOUS_PRIMARY, SHUFFLES as PREVIOUS_SHUFFLES,
    source_controls as semantic_controls)


IMPLEMENTATION = 'geometry-physical-fine-reference-admission-v1-20261003'
PRIMARY = 'FineReferenceJoint_Exact'
SHUFFLES = tuple('FineReferenceShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = {'Geometry': 'Geometry', 'NoAdmission_Exact': 'NoAdmission_Exact',
    'OldClassView_Exact': 'CoherentNoHoldout_Exact',
    'PreviousSemanticReference_Exact': PREVIOUS_PRIMARY}
METHODS = (*REPLAY, PRIMARY, 'FineReferenceOnly_Exact', 'FineReferenceCosine_Exact',
    'FineReferenceHardDelete_Exact', 'FineReferenceSpatialShuffle_Exact', *SHUFFLES)
GATE_CONTROLS = ('OldClassView_Exact', 'PreviousSemanticReference_Exact',
    'FineReferenceCosine_Exact', 'FineReferenceHardDelete_Exact',
    'FineReferenceSpatialShuffle_Exact', *SHUFFLES)
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_alias_increment_over_same_fine_reference_pp': .1,
    'minimum_wrong_parent_domain_wins': 5,
    'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1,
    'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'no_automatic_full_rollout': True, 'no_post_result_control_promotion': True}


@dataclass(frozen=True)
class FineReferenceConfig(SemanticReferenceConfig):
    physical_crop_side: int = 256
    encoder_side: int = 512
    patch_side: int = 16

    def validate(self):
        if (self.physical_crop_side != 256 or self.encoder_side != 512
                or self.patch_side != 16 or self.beta <= 0
                or self.query_chunk < 1 or self.family_chunk < 1):
            raise ValueError('Require the fixed physical8 reference and positive writer configuration.')


CONFIG = FineReferenceConfig()


def fine_crop_positions(height, width, config=CONFIG):
    config.validate()
    if not 1 <= height <= 512 or not 1 <= width <= 512:
        raise ValueError('Require valid dimensions of one fixed512 window.')
    side = config.physical_crop_side
    return [(t, l, min(side, height-t), min(side, width-l))
            for t in range(0, height, side) for l in range(0, width, side)]


@torch.inference_mode()
def fine_patch_features(observer, rgb, *, capture_safe=False):
    from .finite_vip_observer import finite_proxy_attention

    if (rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.shape[1] != rgb.shape[2]
            or rgb.shape[-1] not in (336, CONFIG.encoder_side)):
        raise ValueError('Require a336 replay or512 fine-reference RGB crop.')
    image = rgb[None].to(observer.device)
    image = ((image-observer.backbone._imagenet_mean)/observer.backbone._imagenet_std).half()
    visual = observer.backbone.model.visual_model
    grid = rgb.shape[-1]//CONFIG.patch_side
    # Only the value reshape is resolution-dependent; do not mutate the original head.
    layout = SimpleNamespace(patch_size=grid)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        cls, raw, registers = visual.get_backbone_features(image)
        if registers.shape[1] != 4 or raw.shape[1] != grid*grid:
            raise RuntimeError('Changed pinned prefix or actual patch resolution.')
        tokens = torch.cat((cls[:, None], registers, raw), dim=1)
        for block in visual.head.blocks:
            attention, _, _ = finite_proxy_attention(layout, block.attn, block.norm1(tokens), raw,
                                                    count_empty=not capture_safe)
            tokens = tokens+block.ls1(attention)
            tokens = tokens+block.ls2(block.mlp(block.norm2(tokens)))
        projected = visual.head.linear_projection(visual.head.ln_final(tokens))
        result = F.normalize(projected[:, 5:].float(), dim=-1)
    if not capture_safe and not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite physically fine reference features.')
    return result


def fine_source_controls(crops, count, coordinates, members, excluded, valid,
                         assignments, parents, canonical, semantic, cosine,
                         view_pair, config=CONFIG):
    config.validate()
    original, fields = semantic_controls(crops, count, coordinates, members,
        excluded, valid, assignments, parents, canonical, semantic, cosine, view_pair, config)
    sources = {PRIMARY: original[PREVIOUS_PRIMARY],
        'FineReferenceOnly_Exact': original['SemanticReferenceOnly_Exact'],
        'FineReferenceCosine_Exact': original['SemanticReferenceCosine_Exact'],
        **{name: original[old] for name, old in zip(SHUFFLES, PREVIOUS_SHUFFLES)}}
    index = query_permutation(valid, config.random_seed)
    sources['FineReferenceSpatialShuffle_Exact'] = sources[PRIMARY][index]
    fields['spatial_risk_spectrum_error'] = float((sources[PRIMARY][valid].sort(0).values-
        sources['FineReferenceSpatialShuffle_Exact'][valid].sort(0).values).abs().max()) if bool(valid.any()) else 0.
    return sources, fields
