"""Validate saved predictions after JSON list/tuple identity comparison failed.

Never rerun inference or replace the original result, worker state or failure.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_gear_ov import digest
from eval_rival_fine_full import save


def finalize(root):
    protocol=json.loads((root/'protocol.json').read_text())
    recovered=root/'suite_aggregation_recovery.json'
    if recovered.exists():raise RuntimeError('Existing recovery; preserve it.')
    completed={}
    for dataset,entry in protocol['datasets'].items():
        output=root/dataset
        row=json.loads((output/'merged.json').read_text())
        shard=json.loads((output/'s0/results.json').read_text())
        previous=json.loads(Path(entry['exact_base_reference']).read_text())
        signature=row['signature'];keys=signature['sample_keys']
        if (row['status']!='complete' or shard['status']!='complete' or not row.get('coverage_verified')
                or row['processed_images']!=entry['total_images'] or len(set(keys))!=entry['total_images']
                or digest(keys)!=signature['global_sample_keys_sha256']
                or not shard.get('weights_frozen') or not shard.get('head_weights_unchanged')
                or not shard.get('target_masks_used_only_after_prediction')):
            raise RuntimeError('Incomplete or unfrozen saved predictions: '+dataset)
        if signature['gear']['method_sources']!=protocol['method_sources']:
            raise RuntimeError('Frozen source identity differs: '+dataset)
        for key in ('sample_keys','checkpoints','vocabulary','classes'):
            if signature[key]!=previous['signature'][key]:
                raise RuntimeError('JSON-normalized full identity differs: '+dataset+'/'+key)
        vocabulary=Path(entry['vocabulary_config'])
        if signature['vocabulary']['sha256']!=hashlib.sha256(vocabulary.read_bytes()).hexdigest():
            raise RuntimeError('Vocabulary file changed: '+dataset)
        with np.load(output/'s0/per_image_confusions.npz',allow_pickle=False) as data:
            if data['sample_keys'].tolist()!=keys:raise RuntimeError('Archive sample sequence differs.')
            for p,group in row['metrics'].items():
                for name,item in group.items():
                    if not np.array_equal(data[p+'__'+name].sum(0),item['confusion_matrix']):
                        raise RuntimeError('Archive confusion sum differs: '+dataset+'/'+name)
                if group['Fixed20_TaskCoupled']['confusion_matrix']!=previous['metrics'][p]['Fixed20_TaskCoupled']['confusion_matrix']:
                    raise RuntimeError('Exact Base full confusion differs: '+dataset)
                targets=[np.asarray(item['confusion_matrix']).sum(-1) for item in group.values()]
                if any(not np.array_equal(targets[0],target) for target in targets[1:]):
                    raise RuntimeError('Paired target counts differ: '+dataset)
        timing=json.loads((output/'timing.json').read_text())
        if timing.get('status')!='complete' or not timing.get('exclusive_gpu_verified'):
            raise RuntimeError('Exclusive singleton timing incomplete: '+dataset)
        if row.get('exact_base_confusion_replayed'):
            completed[dataset]=str(output/'merged.json')
            continue
        target=output/'aggregation_recovered_merged.json'
        if target.exists():raise RuntimeError('Existing recovered output; preserve it.')
        row.update(exact_base_confusion_replayed=True,paired_scored_targets_equal=True,
            original20_identity_verified=True,per_image_confusions_verified=True,
            primary_pixel_changes=shard['primary_pixel_changes'],
            aggregation_recovery='JSON-normalized list/tuple vocabulary identity; predictions unchanged.')
        save(target,row)
        save(output/'aggregation_recovery.json',dict(status='complete',predictions_rerun=False,
            original_worker_state=json.loads((output/'worker_status.json').read_text()),
            original_result_sha256=hashlib.sha256((output/'merged.json').read_bytes()).hexdigest(),
            exact_base_confusion_replayed=True,serialized_input_identities_equal=True,
            recovered_result=str(target),processed=entry['total_images'],total=entry['total_images']))
        completed[dataset]=str(target)
    save(recovered,dict(status='complete',completed=completed,predictions_rerun=False,
        original_terminal_state_preserved=True,reason='In-memory tuple aliases were compared with JSON list aliases; all serialized identities and predictions verified.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    finalize(parser.parse_args().root)
