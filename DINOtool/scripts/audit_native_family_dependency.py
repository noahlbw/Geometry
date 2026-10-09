"""Cached source audit without image loading, target masks or network forwards."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.coherent_native_admission import native_risk, union_risk
from dinotool.family_dependency_audit import IMPLEMENTATION, dependency_fields, summarize
from dinotool.native_class_ownership import family_class_field
from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.stratified_soft_alias import crop_stencil
from eval_pair_context_reader import ORIGINAL
from run_native_ownership_feasibility import native_crops, DATASETS, SOURCE as NOISE
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import TOOL


SOURCE = TOOL/'results/native_class_ownership_feasibility_r2_20261003'
PREDICTION = TOOL/'results/coherent_native_admission_20261003'
SCENARIOS = ('clean', 'wrong_parent', 'paraphrase')


def family_mass(crops, count, coordinates, members, assignments, valid):
    output = torch.zeros(len(valid), members.numel(), dtype=torch.float64)
    parents = torch.empty(members.numel(), dtype=torch.long)
    parents[members] = torch.arange(len(members))[:, None]
    alias_ids = torch.arange(members.numel())
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, (512, 512))
        probabilities = profiled_logits(crop, members).double().softmax(-1)
        for family in assignments.unique().tolist():
            mask = assignments[members] == family
            mass = (probabilities*mask[None]).sum(-1)
            tested = alias_ids[assignments == family]
            output[:, tested] += (mass[ids][:, :, parents[tested]]*coeff.double()[..., None]).sum(1)
    return output.masked_fill(~valid[:, None], 0.)


def add(target, row):
    for name, value in row.items():
        target[name] = target.get(name, 0)+value


@torch.inference_mode()
def run(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    suite = json.loads((PREDICTION/'suite_results.json').read_text())
    if root.exists() or suite['status'] != 'complete' or suite['failures']:
        raise ValueError('New isolated root and completed prediction suite required.')
    root.mkdir()
    torch.set_num_threads(2)
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'prediction_source': str(PREDICTION),
        'ownership_source': str(SOURCE), 'images': 64, 'scenarios': SCENARIOS,
        'audit_only': True, 'target_masks_loaded': False, 'construction_metadata_audit_only': True,
        'used_to_fit_selector': False, 'new_image_forwards': 0, 'new_text_forwards': 0,
        'gpu_used': False, 'no_new_predictor_or_parameter': True})
    begun, completed = time.perf_counter(), []
    for dataset in DATASETS:
        prior = json.loads((PREDICTION/dataset/'merged.json').read_text())
        noise = json.loads((NOISE/dataset/'merged.json').read_text())
        metadata = json.loads((SOURCE/dataset/'families.json').read_text())
        if (not prior['coverage_verified'] or prior['processed_images'] != 8
                or prior['sample_keys'] != noise['sample_keys'] or prior['signature'] != noise['signature']):
            raise RuntimeError('Changed fixed sample/source contract.')
        directory = root/dataset
        (directory/'fields').mkdir(parents=True)
        records, aggregate = {}, {}
        for index, sample in enumerate(prior['sample_keys']):
            positions = noise['diagnostics'][sample]['native_crop_positions']
            fields, entries = {}, {}
            with np.load(ORIGINAL/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                    NOISE/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as cached, np.load(
                    SOURCE/dataset/'fields'/f'{index}.npz', allow_pickle=False) as held, np.load(
                    PREDICTION/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as predicted:
                coordinates, valid = [torch.from_numpy(original[n]) for n in ('coordinates', 'valid')]
                for scene in SCENARIOS:
                    for p in prior['signature']['classes']:
                        key = scene+'__'+p
                        description = metadata[key]
                        parents = torch.tensor(description['parents'])
                        assignments = torch.empty(len(parents), dtype=torch.long)
                        for family, ids in enumerate(description['family_indices']):
                            assignments[ids] = family
                        members = torch.stack([(parents == c).nonzero().flatten() for c in range(int(parents.max())+1)])
                        crops, count = native_crops(cached, key, positions, torch.device('cpu'))
                        full, full_known = family_class_field(crops, count, coordinates, members,
                            torch.zeros(1, len(parents), dtype=torch.bool), valid)
                        reference, known = [torch.from_numpy(held[key+'__'+n]) for n in ('reference', 'known')]
                        entry, guard = dependency_fields(reference, known, full, full_known, assignments, parents, valid)
                        risk = native_risk(reference, known, assignments, parents, valid)
                        error = float((risk-entry['risk_held']).abs().max())
                        if error != 0:
                            raise RuntimeError('Original native-risk formula does not replay.')
                        view = torch.from_numpy(cached[key+'__risk']).double().amax(-1)
                        joint = union_risk(view, risk)
                        replay = float((joint-torch.from_numpy(predicted[key+'__coherent_risk'])).abs().max())
                        if replay > guard:
                            raise RuntimeError('Frozen coherent source replay exceeds FP64 guard: '+str(replay))
                        mass = family_mass(crops, count, coordinates, members, assignments, valid)
                        for name, array in {**entry, 'native_family_mass': mass}.items():
                            fields[key+'__'+name] = array.numpy()
                        entries[key] = {'numerical_guard': guard, 'risk_replay_max_error': error,
                            'joint_risk_replay_max_error': replay, 'all': summarize(entry, mass, guard=guard),
                            'per_class': {name: summarize(entry, mass, members[c], guard) for c, name in enumerate(prior['signature']['classes'][p])}}
            # Construction identity is used only after persistence, never as source input.
            np.savez_compressed(directory/'fields'/f'{index}.npz', **fields)
            construction = {}
            for p, description in prior['vocabularies']['wrong_parent']['replacements'].items():
                names = prior['signature']['classes'][p]
                construction[p] = {}
                for change in description['replacements']:
                    slots = torch.tensor([20*names.index(change['class'])+slot for slot in change['slots']])
                    pair = {}
                    for scene in ('wrong_parent', 'paraphrase'):
                        key = scene+'__'+p
                        entry = {name: torch.from_numpy(fields[key+'__'+name]) for name in
                            ('known', 'margin_loss', 'added_risk', 'risk_held', 'risk_full')}
                        pair[scene] = summarize(entry, torch.from_numpy(fields[key+'__native_family_mass']), slots,
                            entries[key]['numerical_guard'])
                    construction[p][change['class']] = pair
            records[sample] = {'sources': entries, 'construction_audit': construction}
            for key, entry in entries.items():
                add(aggregate.setdefault(key, {}).setdefault('all', {}), entry['all'])
                for name, values in entry['per_class'].items():
                    add(aggregate[key].setdefault('per_class', {}).setdefault(name, {}), values)
            save(root/'suite_status.json', {'status': 'running', 'dataset': dataset, 'processed': index+1,
                'total': 8, 'completed': completed, 'elapsed_seconds': time.perf_counter()-begun})
        row = {'status': 'complete', 'implementation': IMPLEMENTATION, 'sample_keys': prior['sample_keys'],
            'signature': prior['signature'], 'processed_images': 8, 'total_images': 8, 'coverage_verified': True,
            'target_masks_loaded': False, 'audit_only': True, 'used_to_fit_selector': False,
            'source_fields_precede_construction_audit': True, 'records': records, 'aggregate': aggregate}
        save(directory/'results.json', row)
        completed.append(dataset)
    final = {'status': 'complete', 'implementation': IMPLEMENTATION, 'completed': completed,
        'audit_only': True, 'used_to_fit_selector': False, 'target_masks_loaded': False,
        'new_image_forwards': 0, 'new_text_forwards': 0, 'gpu_used': False,
        'no_new_predictor_or_parameter': True, 'wall_seconds': time.perf_counter()-begun}
    save(root/'suite_results.json', final)
    save(root/'suite_status.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
