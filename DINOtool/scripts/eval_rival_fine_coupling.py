"""Cached-source pair-preserving coupling; no fresh visual observations."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from dinotool.rival_fine_coupling import IMPLEMENTATION, METHODS, REPLAY, WRITES, write_admitted_pairs
from eval_fine_alias_view import PREVIOUS as REFERENCE
from eval_geometry_semantic_innovation import parse_args
from eval_native_alias_noise import main, ORIGINAL
from run_region_semantic_suite_a800 import TOOL


PREVIOUS = TOOL/'results/fine_alias_view_admission_r2_20261003'


@torch.inference_mode()
def predict(image, geometry, banks, vip, variants, original, previous):
    relative = Path(original.zip.filename).relative_to(ORIGINAL)
    operator = torch.from_numpy(original['operator']).to(geometry.device)
    valid = torch.from_numpy(original['valid']).to(geometry.device)
    output, frozen = {}, {}
    diag = {'new_image_observation_forwards': 0, 'new_intervention_forwards': 0,
        'source_risks_and_predictions_precede_masks': True, 'scenarios': {}}
    with np.load(PREVIOUS/relative, allow_pickle=False) as cached:
        for scene, (vbanks, _) in variants.items():
            output[scene], diag['scenarios'][scene] = {}, {}
            for p in vbanks:
                key = scene+'__'+p
                load = lambda name: torch.from_numpy(cached[key+'__'+name+'__scores']).to(geometry.device)
                values = {name: load(name) for name in REPLAY}
                observations = {name: load(name) for name in {'ObserverAll20', *WRITES.values()}}
                local = torch.from_numpy(original[p+'__local']).to(geometry.device)
                values.update(write_admitted_pairs(values['NoAdmission_Exact'], local, operator, observations))
                zeros = {name: observations['ObserverAll20'] for name in observations}
                identity = write_admitted_pairs(values['NoAdmission_Exact'], local, operator, zeros)
                error = max(float((identity[name]-values['NoAdmission_Exact']).abs().max()) for name in WRITES)
                if error != 0 or set(values) != set(METHODS):
                    raise RuntimeError('Original coupling identity or declared endpoints changed.')
                pair = torch.from_numpy(cached[key+'__fine_view_pair_risk']).to(geometry.device)
                members = torch.stack([(vbanks[p].parent_indices == c).nonzero().flatten()
                    for c in range(vbanks[p].class_count)])
                kept = (pair[valid][:, members] == 0).sum(2)
                diag['scenarios'][scene][p] = {'zero_risk_identity_max_error': error,
                    'retained_aliases_per_query_class_rival': {'minimum': int(kept.min()),
                        'maximum': int(kept.max()), 'mean': float(kept.double().mean())}}
                for name, scores in values.items():
                    frozen[key+'__'+name+'__scores'] = scores.cpu().numpy()
                output[scene][p] = values
    return output, frozen, diag


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    if json.loads((PREVIOUS/'suite_results.json').read_text())['status'] != 'complete':
        raise RuntimeError('Complete fine alias source required.')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION)
