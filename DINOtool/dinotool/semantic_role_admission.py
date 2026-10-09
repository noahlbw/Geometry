"""Vocabulary-time frozen meaning evidence, distinct from image activation."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .class_attachment_admission import METHODS as LEGACY
from .coherent_native_admission import CONFIG as WRITER_CONFIG, union_risk
from .geometry_language_observation import REPOSITORY, REVISION


IMPLEMENTATION = 'geometry-frozen-semantic-role-admission-v1-20261003'
PRIMARY = 'SemanticRoleJoint_Exact'
ALIAS_SHUFFLES = tuple('SemanticRoleAliasShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'SemanticRoleOnly_Exact', 'SemanticRoleVisualSupported_Exact',
    'SemanticRoleHard_Exact', 'SemanticRoleMeanLogit', *ALIAS_SHUFFLES)
METHODS = (*LEGACY, *NEW_METHODS)
GATE_CONTROLS = ('CoherentNativeJoint_Exact', 'CoherentNoHoldout_Exact',
    'ClassAttachmentJoint_Exact', 'ClassAttachmentTextOnly_Exact',
    'SemanticRoleOnly_Exact', 'SemanticRoleVisualSupported_Exact',
    'SemanticRoleHard_Exact', 'SemanticRoleMeanLogit', *ALIAS_SHUFFLES)
GATE = {'minimum_clean_gain_pp': .1, 'minimum_wrong_parent_gain_pp': .1,
    'minimum_wrong_parent_domain_wins': 5, 'maximum_clean_paraphrase_protocol_loss_pp': 1.,
    'minimum_observed_wrong_parent_damage_pp': .1,
    'wrong_parent_mean_above_controls': list(GATE_CONTROLS),
    'earlier_failed_gates_unchanged': True, 'no_post_result_control_promotion': True,
    'full_image_implementation_must_be_frozen_before_rollout': True}
SOURCE = Path('/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/Qwen25VL7B_GeometryDiagnostic_20261002')
ROLE_NAMES = ('direct', 'shared_material', 'part', 'scene_cooccurrence', 'unknown')


@dataclass(frozen=True)
class SemanticRoleConfig:
    beta: float = WRITER_CONFIG.beta
    query_chunk: int = WRITER_CONFIG.query_chunk
    random_seed: int = WRITER_CONFIG.random_seed
    source_batch_size: int = 16


CONFIG = SemanticRoleConfig()


def roles(names, parent):
    if len(names) < 2 or len(set(names)) != len(names) or not 0 <= parent < len(names) or len(names)+4 > 26:
        raise ValueError('Require distinct declared classes and a valid parent.')
    return (*ROLE_NAMES, *('competitor:'+name for i, name in enumerate(names) if i != parent))


def role_prompt(names, parent, alias, order):
    labels = roles(names, parent)
    if sorted(order) != list(range(len(labels))) or not alias.strip():
        raise ValueError('Invalid role order or phrase.')
    descriptions = ('Directly denotes the target class, a subtype or an appearance of that class.',
        'Shared material or appearance, not a reliable exclusive category denotation.',
        'A part of the target object rather than an exclusive rival category.',
        'A scene or nearby object co-occurring with the target, not the target itself.',
        'Unknown, ambiguous, unsupported, or not decidable from the supplied class names.')
    descriptions += tuple('Directly denotes the competing category '+json.dumps(name)+
        ', rather than the target category.' for i, name in enumerate(names) if i != parent)
    options = '\n'.join(chr(65+i)+'. '+descriptions[role] for i, role in enumerate(order))
    return ('Judge only the meaning of a candidate phrase for a semantic segmentation vocabulary. '
        'There is no image and no information about model activation. Do not judge usefulness or frequency. '
        'Declared categories: '+json.dumps(list(names))+'. Target category: '+json.dumps(names[parent])+'. '
        'Candidate phrase: '+json.dumps(alias)+'. '
        'Choose a competitor only when the phrase explicitly denotes that category instead of the target. '
        'Do not reject shared materials, parts or scene descriptions as if they were exclusive competitors. '
        'Category names define the available ontology; do not invent dataset-specific label rules. '
        'If multiple meanings remain plausible choose unknown. '
        'Answer with exactly one uppercase option letter and no explanation.\n'+options)


def semantic_conflict(logits, parents, classes):
    if (logits.shape != (2, len(parents), classes+4) or not bool(torch.isfinite(logits).all())
            or classes < 2 or bool(((parents < 0) | (parents >= classes)).any())):
        raise ValueError('Require restored forward/reverse logits and declared parents.')
    probability = logits.double().softmax(-1)
    choice = logits.argmax(-1)
    agree = choice[0] == choice[1]
    risk = torch.zeros(len(parents), classes, dtype=torch.float64, device=logits.device)
    for a, parent in enumerate(parents.tolist()):
        selected = int(choice[0, a])
        if bool(agree[a]) and selected >= len(ROLE_NAMES):
            rivals = [c for c in range(classes) if c != parent]
            risk[a, rivals[selected-len(ROLE_NAMES)]] = probability[:, a, selected].min()
    return risk


def source_controls(base, conflict, field, known, assignments, parents, valid, members, config=CONFIG):
    if conflict.shape != (len(parents), field.shape[-1]) or base.shape != (len(valid), len(parents)):
        raise ValueError('Semantic/visual sources must match the unchanged aliases.')
    semantic = conflict.amax(-1)[None].expand_as(base).masked_fill(~valid[:, None], 0.)
    reference = field[:, assignments].double()
    own = reference.gather(-1, parents[None, :, None].expand(len(valid), -1, 1))
    supported = known[assignments] & known[assignments, parents][:, None]
    visual = (((reference-own)/2).tanh().clamp_min(0)*conflict[None]*supported[None]).amax(-1)
    visual.masked_fill_(~valid[:, None], 0.)
    output = {PRIMARY: union_risk(base, semantic), 'SemanticRoleOnly_Exact': semantic,
        'SemanticRoleVisualSupported_Exact': union_risk(base, visual),
        'SemanticRoleHard_Exact': union_risk(base, (semantic > 0).double())}
    generator = torch.Generator().manual_seed(config.random_seed)
    for name in ALIAS_SHUFFLES:
        permutation = torch.stack([torch.randperm(members.shape[-1], generator=generator) for _ in members]).to(members.device)
        changed = torch.empty_like(semantic)
        grouped = semantic[:, members]
        changed[:, members] = grouped.gather(-1, permutation[None].expand_as(grouped))
        output[name] = union_risk(base, changed)
    return output, semantic


class FrozenSemanticRoles:
    def __init__(self, source=SOURCE, device='cuda'):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.manifest = json.loads((Path(source)/'language_source_manifest.json').read_text())
        if (self.manifest['repository'] != REPOSITORY or self.manifest['revision'] != REVISION
                or not self.manifest['frozen']):
            raise ValueError('Require the existing pinned frozen semantic source.')
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to(self.device).eval()
        self.model.requires_grad_(False)
        ids = [self.processor.tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(26)]
        if any(len(v) != 1 for v in ids) or len({v[0] for v in ids}) != 26:
            raise ValueError('Option letters must be distinct single tokens.')
        self.option_ids = torch.tensor([v[0] for v in ids], device=self.device)

    @torch.inference_mode()
    def score(self, specifications):
        rendered, orders = [], []
        for names, parent, alias, reverse in specifications:
            size = len(roles(names, parent))
            order = tuple(reversed(range(size))) if reverse else tuple(range(size))
            message = [{'role': 'user', 'content': [{'type': 'text',
                'text': role_prompt(names, parent, alias, order)}]}]
            rendered.append(self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            orders.append(order)
        inputs = self.processor.tokenizer(rendered, padding=True, return_tensors='pt').to(self.device)
        logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
        output = []
        for i, order in enumerate(orders):
            selected = logits[i, self.option_ids[:len(order)]]
            restored = torch.empty_like(selected)
            restored[torch.tensor(order, device=self.device)] = selected
            mass = float((selected.logsumexp(0)-logits[i].logsumexp(0)).exp())
            output.append((restored.cpu().tolist(), mass, int(inputs.attention_mask[i].sum())))
        return output
