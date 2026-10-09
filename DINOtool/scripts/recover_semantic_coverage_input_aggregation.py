"""Read-only inference recovery: preserve failed controller and verify outputs.

The frozen controller passed a str to a Path-only JSON reader after VDD merge.
This script reruns only merge/identity verification, never any model prediction.
Existing raw merged files and original failed suite/log are left intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_semantic_coverage_input import BASE,FULL_METHODS
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_generation_root_alias import image_hashes
from run_sat_geometry_transport_suite import alive,read_json


def recover(root):
    output=root/'aggregation_recovery.json'
    if output.exists():raise RuntimeError('Existing aggregation recovery; preserve it.')
    failure=read_json(root/'suite_results.json');protocol=read_json(root/'protocol.json')
    if alive('gsci09_controller') or not failure or failure.get('status')!='failed' or "'str' object has no attribute 'exists'" not in failure.get('error',''):
        raise RuntimeError('Expected terminal path-type controller failure missing.')
    result={}
    for dataset,entry in protocol['datasets'].items():
        shards=[root/'full'/dataset/('s'+str(i)) for i in range(protocol['full_shards'][dataset])]
        for i,shard in enumerate(shards):
            worker=read_json(shard/'worker_status.json');row=read_json(shard/'results.json')
            if (alive('gsci09_full_'+dataset+'_s'+str(i)) or not worker or worker.get('status')!='complete'
                    or not row or row.get('status')!='complete' or row['processed_images']!=row['total_images']):
                raise RuntimeError('Worker not terminal and complete: '+str(shard))
        path=root/'full'/dataset/'merged.json'
        row=read_json(path) if path.exists() else merge([str(p) for p in shards],str(path),diagnostic_weight='images')
        ref=read_json(Path(entry['exact_base_reference']))
        if (not row or not row.get('coverage_verified') or row['processed_images']!=entry['total_images']
                or not ref or not ref.get('coverage_verified') or ref['processed_images']!=entry['total_images']
                or ref['signature']['sample_keys']!=row['signature']['sample_keys']
                or ref['signature']['checkpoints']!=row['signature']['checkpoints']
                or ref['signature']['classes']!=row['signature']['classes']
                or ref['signature']['vocabulary']['sha256']!=row['signature']['vocabulary']['sha256']):
            raise RuntimeError('Coverage/checkpoint/class/word identity differs.')
        expected=image_hashes([Path(p).parent/'per_image_confusions.npz' for p in ref['shards']],dataset,'Fixed20_TaskCoupled')
        actual=image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,BASE)
        if expected!=actual or ref['metrics'][dataset]['Fixed20_TaskCoupled']['confusion_matrix']!=row['metrics'][dataset][BASE]['confusion_matrix']:
            raise RuntimeError('Full per-image/aggregate Existing20 replay differs.')
        targets=[np.asarray(row['metrics'][dataset][name]['confusion_matrix']).sum(-1) for name in FULL_METHODS]
        if any(not np.array_equal(targets[0],x) for x in targets[1:]):raise RuntimeError('Paired scored targets differ.')
        vocab=protocol['candidate_input_sha256'][dataset]
        for arm,expected_sha in vocab.items():
            config=root/'vocabularies'/(dataset+'_'+arm+'.json')
            if hashlib.sha256(config.read_bytes()).hexdigest()!=expected_sha:raise RuntimeError('Frozen candidate input changed.')
            if any(len(c['synonyms'])!=20 for c in json.loads(config.read_text())['classes']):
                raise RuntimeError('Candidate twenty-slot counts differ.')
        verified=root/'full'/dataset/'verified_merged.json'
        if verified.exists():raise RuntimeError('Existing verified merge; preserve it.')
        row.update(exact_base_per_image_replayed=True,paired_scored_targets_equal=True,
            candidate_bank_counts_exact20=True,full_unique_samples_verified=True,
            aggregation_recovery='Only Path conversion in post-inference verification; original outputs preserved')
        save(verified,row);result[dataset]=dict(images=row['processed_images'],merged=str(verified))
    save(output,dict(status='complete',outcome='full_verified',datasets=result,
        images=sum(row['images'] for row in result.values()),inference_rerun=False,
        original_failure_sha256=hashlib.sha256((root/'suite_results.json').read_bytes()).hexdigest(),
        recovery_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_failed_suite_and_logs_preserved=True))
    print(json.dumps(read_json(output)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);recover(p.parse_args().root)
