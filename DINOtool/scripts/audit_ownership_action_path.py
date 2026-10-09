"""CPU-only attribution of completed ownership predictions; no new model run."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from dinotool.ownership_path_audit import IMPLEMENTATION, action_decomposition, target_margin_signs
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from audit_alias_action_capacity import dense
from eval_gear_ov import protocol
from eval_pair_context_reader import ORIGINAL
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import TOOL, SETTINGS


SOURCE = TOOL/'results/native_class_ownership_prediction_r2_20261003'
SCENARIOS = ('clean', 'wrong_parent', 'paraphrase')


def potential(directed):
    a = np.asarray(directed, np.float64)
    return (a-a.transpose(0, 2, 1)).mean(-1)


def predict_scores(scores):
    return dense(torch.from_numpy(scores)).argmax(-1).numpy()


def run(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    suite = json.loads((SOURCE/'suite_results.json').read_text())
    if root.exists() or suite['status'] != 'complete' or suite['failures']:
        raise ValueError('New isolated root and completed verified source required.')
    root.mkdir()
    torch.set_num_threads(2)
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'source': str(SOURCE),
        'scenarios': SCENARIOS, 'images': 64, 'audit_only': True, 'used_to_fit_selector': False,
        'new_image_forwards': 0, 'new_text_forwards': 0, 'gpu_used': False,
        'no_new_predictor_or_parameter': True, 'sign_competitor': 'strongest non-target class in old coupled scores'})
    started, completed = time.perf_counter(), []
    for dataset, data, vocabulary, _, _ in SETTINGS:
        prior = json.loads((SOURCE/dataset/'merged.json').read_text())
        if not prior['coverage_verified'] or prior['processed_images'] != 8 or len(set(prior['sample_keys'])) != 8:
            raise ValueError('Verified fixed sequence required.')
        args = SimpleNamespace(dataset=dataset, data_root=str(Path('/data/test/datasets')/data),
            sample_seed=20260923, vdd_ontology='official')
        samples, specs, load_image, load_mask = protocol(args, load_class_specs(TOOL/'configs'/vocabulary))
        lookup = {sample.key: sample for sample in samples}
        directory = root/dataset
        (directory/'fields').mkdir(parents=True)
        records, stored_counts = {}, {}
        with np.load(SOURCE/dataset/'per_image_confusions.npz', allow_pickle=False) as matrices:
            for index, key in enumerate(prior['sample_keys']):
                sample = lookup[key]
                image = load_image(sample.image_path if dataset == 'loveda' else sample)
                image_size = tuple(image.shape[-2:])
                del image
                fields, image_record = {}, {}
                with np.load(ORIGINAL/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                        SOURCE/dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as cached:
                    operator, valid = original['operator'], original['valid']
                    for scene in SCENARIOS:
                        image_record[scene] = {}
                        for p, classes in specs.items():
                            prefix = scene+'__'+p
                            old_directed, new_directed = cached[prefix+'__directed'], cached[prefix+'__ownership_directed']
                            delta, energy = action_decomposition(old_directed, new_directed, valid)
                            written = operator @ delta
                            old_scores = cached[prefix+'__NativeAlias_Exact__scores']
                            new_scores = cached[prefix+'__OwnershipJoint_Exact__scores']
                            score_error = float(np.abs(new_scores-old_scores-written).max())
                            if score_error > 1e-10:
                                raise RuntimeError('Cached class-action path fails exact score replay.')
                            b = original[prefix+'__broad'].astype(np.float64)
                            old_b, new_b = b+potential(old_directed), b+potential(new_directed)
                            names = ['old_broad_admitted', 'new_broad_admitted', 'old_coupled', 'new_coupled']
                            for name, values in zip(names, (old_b, new_b, old_scores, new_scores)):
                                fields[prefix+'__'+name] = values
                            fields[prefix+'__increment'] = delta
                            fields[prefix+'__written_increment'] = written
                            image_record[scene][p] = {**energy, 'class_score_replay_max_error': score_error}
                # Derived source fields precede this separate labelled path audit.
                np.savez_compressed(directory/'fields'/f'{index}.npz', **fields)
                for p, classes in specs.items():
                    full = load_mask(sample, p, image_size)
                    target = np.full((512, 512), -1, dtype=np.int64)
                    ah, aw = min(512, image_size[0]), min(512, image_size[1])
                    target[:ah, :aw] = full[:ah, :aw]
                    centres = target[8::16, 8::16].flatten()
                    for scene in SCENARIOS:
                        prefix = scene+'__'+p
                        signs, sign_stats = target_margin_signs(fields[prefix+'__increment'],
                            fields[prefix+'__written_increment'], fields[prefix+'__old_coupled'], centres, valid)
                        predictions = {name: predict_scores(fields[prefix+'__'+name]) for name in
                            ('old_broad_admitted', 'new_broad_admitted', 'old_coupled', 'new_coupled')}
                        entry = image_record[scene][p]
                        entry.update(sign_stats)
                        entry['target_margin_signs_per_class'] = signs.tolist()
                        for label, a, b in (('prewrite', 'old_broad_admitted', 'new_broad_admitted'),
                                           ('coupled', 'old_coupled', 'new_coupled')):
                            counts = transition_counts(predictions[a], predictions[b], target, len(classes))
                            stats = transition_summary(counts)
                            if label == 'coupled':
                                for name, method, endpoint in (('old', 'NativeAlias_Exact', 'base_confusion'),
                                                             ('new', 'OwnershipJoint_Exact', 'proposal_confusion')):
                                    if not np.array_equal(stats[endpoint], matrices[prefix+'__'+method][index]):
                                        raise RuntimeError('Prior per-image confusion does not replay: '+name)
                            entry[label+'_transitions'] = stats
                            stored_counts.setdefault(prefix+'__'+label, []).append(counts)
                records[key] = image_record
                save(root/'suite_status.json', {'status': 'running', 'dataset': dataset, 'processed': index+1,
                    'total': 8, 'completed': completed, 'elapsed_seconds': time.perf_counter()-started})
        np.savez_compressed(directory/'transitions.npz', sample_keys=np.asarray(prior['sample_keys']),
            **{name: np.stack(values) for name, values in stored_counts.items()})
        aggregate = {}
        for scene in SCENARIOS:
            aggregate[scene] = {}
            for p in specs:
                values = [record[scene][p] for record in records.values()]
                entry = {name: sum(v[name] for v in values) for name in ('requested_energy', 'realized_energy', 'cycle_energy')}
                entry['retained_edge_energy_fraction'] = entry['realized_energy']/entry['requested_energy'] if entry['requested_energy'] > 0 else None
                entry['target_margin_signs_per_class'] = np.sum([v['target_margin_signs_per_class'] for v in values], 0).tolist()
                for label in ('prewrite', 'coupled'):
                    entry[label+'_transitions'] = transition_summary(np.stack(stored_counts[scene+'__'+p+'__'+label]).sum(0))
                aggregate[scene][p] = entry
        row = {'status': 'complete', 'implementation': IMPLEMENTATION, 'sample_keys': prior['sample_keys'],
            'signature': prior['signature'], 'processed_images': 8, 'total_images': 8, 'coverage_verified': True,
            'audit_only': True, 'used_to_fit_selector': False, 'new_image_forwards': 0, 'new_text_forwards': 0,
            'gpu_used': False, 'target_masks_loaded': True, 'derived_fields_precede_masks': True,
            'exact_class_score_replay': True, 'exact_per_image_confusion_replay': True,
            'records': records, 'aggregate': aggregate}
        save(directory/'results.json', row)
        completed.append(dataset)
    final = {'status': 'complete', 'implementation': IMPLEMENTATION, 'completed': completed,
        'audit_only': True, 'used_to_fit_selector': False, 'new_image_forwards': 0, 'new_text_forwards': 0,
        'gpu_used': False, 'wall_seconds': time.perf_counter()-started}
    save(root/'suite_results.json', final)
    save(root/'suite_status.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
