"""Complete matched attribution of the unchanged fine-likelihood cap."""
import torch

from . import one_sided_alias_audit as audit
from .supported_positive_alias import predict_image as source_predict
from .witness_cap_alias import witness_risk


IMPLEMENTATION = 'geometry-bounded896-existing-witness-cap-audit-v1-20261006'
OLD_NAMES = {name: name.replace('OneSide_', 'WitnessCapAudit_') for name in audit.METHODS[3:]}
NEW_NAMES = {new: old for old, new in OLD_NAMES.items()}
PRIMARY = OLD_NAMES[audit.PRIMARY]
OBSERVATION_MEAN = OLD_NAMES[audit.OBSERVATION_MEAN]
PREVIOUS_SOFT = 'WitnessCapAudit_PreviousSoft'
METHODS = (*audit.METHODS[:3], *OLD_NAMES.values(), PREVIOUS_SOFT)
DIAGNOSTICS = audit.DIAGNOSTICS
PROTOCOL = {**audit.PROTOCOL,
    'alias_admission': 'EXACT existing fine-likelihood cap; protected wide-positive/fine-negative support',
    'retention': 'exp(fine alias-minus-rival margin) on eligible slots, otherwise1',
    'new_model_equations': False, 'controls': METHODS[4:],
    'interpretation': 'missing matched attribution and real cost of previously rejected candidate; not promotion',
    'matching': 'primary word-only valid class-field Frobenius norm after original H'}


def cap_risk(wide, fine, parents, canonical, valid):
    return witness_risk(wide, fine, parents, canonical, valid)[0]


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared unchanged-cap attribution methods required.')
    args = (local, operator, broad, fine_field, wide, fine, coordinates,
            relation, valid, members, canonical, parents)
    requested = tuple(NEW_NAMES[name] for name in methods if name != PREVIOUS_SOFT)
    values, diagnostics = {}, {}
    if requested:
        old, diagnostics = audit.scores(*args, methods=requested, risk_builder=cap_risk)
        values.update({OLD_NAMES[name]: value for name, value in old.items()})
    if PREVIOUS_SOFT in methods:
        old, stats = audit.scores(*args, methods=(audit.PRIMARY,))
        values[PREVIOUS_SOFT] = old[audit.PRIMARY]
        if not requested:
            diagnostics = stats
    return values, diagnostics


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
