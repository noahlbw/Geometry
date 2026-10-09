"""Same-field controls for the frozen patch-only2 model, not new candidates."""
import torch

from .bounded_patch_only import PRIMARY, PROTOCOL as ORIGINAL_PROTOCOL, predict_image as original_predict


IMPLEMENTATION = 'geometry-bounded-patch2-same-field-coupling-audit-v1-20261005'
METHODS = ('Geometry', 'NoAdmission_Exact', PRIMARY, 'PatchOnly2_Local',
           'WideOnly_Lifted', 'MeanLogit_PatchOnly2', 'PermutedH_PatchOnly2')
PROTOCOL = {**ORIGINAL_PROTOCOL,
    'controls': 'exact frozen patch-only2, its local scores, same-field wide, equal logit mean, valid-token permuted H'}


def local_only(local, broad, operator, valid):
    return local


def wide_only(local, broad, operator, valid):
    return broad


def mean_logits(local, broad, operator, valid):
    return .5 * (local + broad)


def shuffled_operator(operator, valid):
    indices = torch.arange(len(valid), device=operator.device)
    active = valid.nonzero(as_tuple=False).flatten()
    generator = torch.Generator(device=operator.device).manual_seed(20261005)
    indices[active] = active[torch.randperm(len(active), generator=generator, device=operator.device)]
    return operator[indices][:, indices]


def permuted_reconstruction(local, broad, operator, valid):
    return local + shuffled_operator(operator, valid) @ (broad - local)


SCORE_CONTROLS = {name: (PRIMARY, rule) for name, rule in zip(METHODS[3:],
    (local_only, wide_only, mean_logits, permuted_reconstruction))}


@torch.inference_mode()
def predict_image(*args, methods=METHODS, **kwargs):
    return original_predict(*args, methods=methods, score_controls=SCORE_CONTROLS, **kwargs)
