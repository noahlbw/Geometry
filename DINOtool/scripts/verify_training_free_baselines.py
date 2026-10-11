"""Verify completed native baseline artifacts without rerunning any inference."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from eval_training_free_baseline import digest, save
from run_training_free_baselines import read, verify


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    state=read(root/'suite_status.json')
    suite=read(root/'suite_results.json')
    protocol=read(root/'protocol.json')
    samples=read(root/'samples.json')
    costs=read(root/'cost_keys.json')
    checkpoints=read(root/'checkpoint_manifest.json')
    runtime=read(root/'runtime.json')
    require(state['status']==suite['status']=='complete','Suite is incomplete')
    require(not state['active'] and not state['pending'] and not state['failures'],'Unresolved jobs')
    expected=len(protocol['methods'])*len(protocol['datasets'])
    require(len(state['merged'])==len(suite['datasets'])==expected==75,'Missing merged protocols')
    methods={}; shard_count=0; cost_count=0
    for method, source in protocol['methods'].items():
        images=0
        for dataset, entry in protocol['datasets'].items():
            merged=read(root/method/dataset/'merged.json')
            reference=read(root/'references'/(dataset+'.json'))
            keys=[row['key'] for row in samples[dataset]['samples']]
            require(merged['status']=='complete' and merged['coverage_verified'] and
                merged['scored_target_counts_match'],'Unverified merged result')
            require(merged['sample_keys']==keys==reference['coverage']['sample_keys'] and
                len(set(keys))==len(keys)==entry['total_images']==merged['processed_images']==merged['total_images'] and
                merged['sample_keys_sha256']==digest(keys),'Coverage differs: '+method+'/'+dataset)
            signature=merged['signature'];config=source['configs'][dataset]
            require(signature['config']==config and signature['source_commit']==source['commit'] and
                signature['implementation']==protocol['implementation'] and
                signature['checkpoint_manifest']==checkpoints and signature['runtime']==runtime,
                'Frozen identity differs: '+method+'/'+dataset)
            require(hashlib.sha256((root/config['word_file']).read_bytes()).hexdigest()==config['word_sha256'],
                'Query bytes differ: '+method+'/'+dataset)
            shards=8 if entry['total_images']>=1000 else 1
            matrices={p:np.zeros_like(v['confusion_matrix'],dtype=np.int64) for p,v in merged['results'].items()}
            for index in range(shards):
                key=f'{method}/evaluate/{dataset}/{index}'
                output=state.get('completed_outputs',{}).get(key,str(root/method/dataset/'evaluate'/('s'+str(index))))
                row=verify(dict(output=output,phase='evaluate',log=output+'.log'))
                require(row['weights_frozen'] and row['signature']==signature and row['sample_keys']==keys[index::shards],
                    'Incomplete or changed shard: '+key)
                for p in matrices:matrices[p]+=np.asarray(row['results'][p]['confusion_matrix'],np.int64)
                shard_count+=1
            for p,matrix in matrices.items():
                target=np.asarray(next(iter(reference['results'][p].values()))['confusion_matrix'],np.int64).sum(1)
                require(np.array_equal(matrix,merged['results'][p]['confusion_matrix']) and
                    np.array_equal(matrix.sum(1),target),'Confusion/target mismatch: '+method+'/'+dataset+'/'+p)
            key=f'{method}/cost/{dataset}/0'
            output=state.get('completed_outputs',{}).get(key,str(root/method/dataset/'cost/s0'))
            cost_row=verify(dict(output=output,phase='cost',log=output+'.log'))
            cost=read(Path(output)/'cost.json')
            require(cost_row['weights_frozen'] and cost_row['signature']==cost['signature']==signature and
                cost_row['sample_keys']==[r['key'] for r in cost['images']]==costs[dataset],
                'Cost identity differs: '+key)
            require(len(cost['images'])==7 and all(len(r['milliseconds'])==7 and
                np.isfinite(r['milliseconds']).all() and min(r['milliseconds'])>0 for r in cost['images']),
                'Cost samples incomplete: '+key)
            images+=len(keys);cost_count+=1
        require(images==45200,'Incomplete method protocol images: '+method)
        methods[method]=dict(complete_protocols=len(protocol['datasets']),protocol_images=images,
            standalone_costs=len(protocol['datasets']))
    require(cost_count==75 and sum(r['protocol_images'] for r in methods.values())==226000,
        'Incomplete suite totals')
    result=dict(status='complete',complete_protocols=expected,standalone_costs=cost_count,
        per_image_confusion_shards_verified=shard_count,total_protocol_images=226000,methods=methods,
        unique_ordered_coverage_verified=True,per_image_confusion_sums_verified=True,
        vip_scored_target_counts_match=True,frozen_source_config_query_checkpoint_runtime_identities_verified=True,
        cost_7_images_by_7_repeats_verified=True,model_weights_frozen=True,
        note='Protocol-image total includes shared VOC20/21 and PC59/60 photos; LoveDA P/D inference is separate.')
    path=root/'verification.json'
    if path.exists():require(read(path)==result,'Existing verification differs; preserved')
    else:save(path,result)
    print(result,flush=True)


if __name__=='__main__':main()
