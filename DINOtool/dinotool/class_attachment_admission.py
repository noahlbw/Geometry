"""Separate multiclass visual disagreement from held-family word attachment."""
from dataclasses import dataclass

import torch

from .coherent_native_admission import (METHODS as LEGACY, CONFIG as WRITER_CONFIG,
    native_risk, union_risk)
from .native_class_ownership import ownership_risk
from .native_ownership_reader import query_permutation
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'geometry-native-class-attachment-admission-v1-20261003'
PRIMARY = 'ClassAttachmentJoint_Exact'
ALIAS_SHUFFLES = tuple('AttachmentAliasShuffle'+str(i)+'_Exact' for i in range(3))
REFERENCE_SHUFFLES = tuple('AttachmentReferenceShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'ClassAttachmentNoView_Exact', 'ClassAttachmentOnly_Exact',
    'ClassAttachmentTextOnly_Exact', *ALIAS_SHUFFLES, *REFERENCE_SHUFFLES,
    'ClassAttachmentMeanLogit')
METHODS = (*LEGACY, *NEW_METHODS)
GATE_CONTROLS = ('CoherentNativeJoint_Exact', 'CoherentNoHoldout_Exact',
    'CoherentHardDelete_Exact', 'CoherentMeanLogit', 'ClassAttachmentNoView_Exact',
    'ClassAttachmentOnly_Exact', 'ClassAttachmentTextOnly_Exact',
    *ALIAS_SHUFFLES, *REFERENCE_SHUFFLES, 'ClassAttachmentMeanLogit')
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_wrong_parent_domain_wins': 5, 'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1,
    'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'earlier_failed_gates_unchanged': True, 'no_post_result_control_promotion': True,
    'full_image_implementation_must_be_frozen_before_rollout': True}


@dataclass(frozen=True)
class ClassAttachmentConfig:
    beta: float = WRITER_CONFIG.beta
    query_chunk: int = WRITER_CONFIG.query_chunk
    random_seed: int = WRITER_CONFIG.random_seed


CONFIG = ClassAttachmentConfig()


def source_controls(view_pair, field, known, assignments, parents, canonical, conflict,
                    valid, members, full, full_known, config=CONFIG):
    view = view_pair.double().amax(-1)
    class_risk = native_risk(full, full_known, torch.zeros_like(assignments), parents, valid)
    base = union_risk(view, class_risk)
    attachment = ownership_risk(field, known, assignments, parents, canonical, conflict, valid).amax(-1)
    text = conflict.double().amax(-1)[None].expand_as(base).masked_fill(~valid[:, None], 0.)
    output = {PRIMARY: union_risk(base, attachment),
        'ClassAttachmentNoView_Exact': union_risk(class_risk, attachment),
        'ClassAttachmentOnly_Exact': attachment,
        'ClassAttachmentTextOnly_Exact': union_risk(base, text)}
    for name, permutation in zip(ALIAS_SHUFFLES, alias_permutations(members, canonical, config.random_seed)):
        changed = torch.empty_like(attachment)
        grouped = attachment[:, members]
        changed[:, members] = grouped.gather(-1, permutation[None].expand_as(grouped))
        output[name] = union_risk(base, changed)
    for i, name in enumerate(REFERENCE_SHUFFLES):
        index = query_permutation(valid, config.random_seed+i)
        changed = ownership_risk(field[index], known, assignments, parents, canonical, conflict, valid).amax(-1)
        output[name] = union_risk(base, changed)
    return output, base, attachment
