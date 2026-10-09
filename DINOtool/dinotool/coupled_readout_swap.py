"""Replace only the local semantic head in the retained coupled predictor."""
import torch

from .matched_readout_controls import OPERATOR_NOTES, run_head


IMPLEMENTATION = 'fixed-context-rival-reconstruction-local-head-swap-v1-20261004'
READOUTS = ('Geometry', 'SCLIP_Two', 'VIPProxy_Two')
REFERENCE_METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact')
METHODS = (*REFERENCE_METHODS, *(name + suffix for name in READOUTS[1:]
            for suffix in ('', '_NoAdmission', '_RivalFineHard')))
COUPLED = ('RivalFineHard_Exact', *(name + '_RivalFineHard' for name in READOUTS[1:]))
FIXED_PROTOCOL = {
    'replaced': 'Local frozen semantic-head operator only, in both head blocks.',
    'wide': 'Unchanged retained finite VIP observer, text, crop grid and salience.',
    'admission': 'One unchanged physical8 alias/rival admission computed per tile and shared by all arms.',
    'reconstruction': 'Same H from original raw-feature Geometry relation and validity for all arms.',
    'assembly': 'Same interpolation, temperature, Hann probability blending and argmax.',
    'scope': 'Local-head necessity test; original Geometry reconstruction graph remains in every coupled arm.',
    'operator_notes': {name: OPERATOR_NOTES[name] for name in READOUTS[1:]},
}


@torch.inference_mode()
def local_features(geometry, prepared):
    features = {'Geometry': prepared.geometry_projected}
    diagnostics = {}
    with geometry.backbone._autocast():
        for name in READOUTS[1:]:
            features[name], observed = run_head(
                geometry.backbone.model.visual_model.head, prepared.backbone_tokens,
                prepared.backbone_tokens[:, prepared.prefix_tokens:],
                prepared.geometry_patch_conditional, prepared.prefix_tokens, name,
                prepared.block_index)
            diagnostics.update({name + '__' + key: value for key, value in observed.items()})
    if any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise RuntimeError('Nonfinite local readout.')
    return features, diagnostics


def replace_local(local, reference_local, operator, reference_unscreened, reference_screened):
    if (local.shape != reference_local.shape or local.shape != reference_unscreened.shape
            or local.shape != reference_screened.shape or operator.shape != (len(local), len(local))):
        raise ValueError('Matching local scores, reference scores and fixed reconstruction required.')
    # s(l) - s(g) = (I-H)(l-g); shared broad and admitted evidence cancel.
    displacement = local.double() - reference_local.double()
    change = displacement - operator.double() @ displacement
    unscreened, screened = reference_unscreened + change, reference_screened + change
    if not bool(torch.isfinite(unscreened).all() and torch.isfinite(screened).all()):
        raise RuntimeError('Nonfinite fixed-protocol replacement scores.')
    return unscreened, screened
