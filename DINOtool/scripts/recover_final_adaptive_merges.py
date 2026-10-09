"""Recover completed metadata-only merge failures without rerunning predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import run_evidence_adaptive_readout as queue
from eval_final_adaptive_deployment import METHODS
from eval_rival_fine_full import save
from run_final_adaptive_deployment import verify_dataset
from merge_gear_ov_shards import merge


def main(root,datasets):
    root.resolve().relative_to((queue.TOOL/'results').resolve())
    state=queue.read_json(root/'suite_status.json')
    if not state or state['active'] or state['pending']:
        raise RuntimeError('Wait for original workers/controller to become terminal.')
    if not datasets or len(set(datasets))!=len(datasets) or any(d not in queue.ORDER for d in datasets):
        raise RuntimeError('Explicit unique subset of frozen protocols required.')
    result=root/('recovery_results.json' if tuple(datasets)==queue.ORDER else 'recovery_results_completed_subset.json')
    if result.exists():
        raise RuntimeError('Existing recovery result; preserve it.')
    protocol=queue.read_json(root/'protocol.json')
    queue.METHODS=METHODS
    completed={}
    for dataset in datasets:
        entry=protocol['datasets'][dataset]
        folders=[root/'full'/dataset/('s'+str(i)) for i in range(entry['shards'])]
        if (root/'full'/dataset/'merged.json').exists():
            raise RuntimeError('Unexpected existing merged output; inspect before recovery.')
        checked=[]
        for shard,folder in enumerate(folders):
            queue.verify_job(dict(output=str(folder),dataset=dataset,phase='full',session='recovery_readonly'))
            row=queue.read_json(folder/'results.json')
            checks=queue.read_json(folder/'deployment_checks.json')['samples']
            if [v['key'] for v in checks]!=row['signature']['sample_keys']:
                raise RuntimeError('Deployment diagnostic/sample mismatch.')
            if any(v['diagnostic']['geometry_encodings']>4 or v['diagnostic']['wide_encodings']>4
                   or v['diagnostic']['fine_forwards']!=0 or v['maximum_changed_gap']>2e-6 for v in checks):
                raise RuntimeError('Observation budget or numerical check failed.')
            row['diagnostics'][dataset]['tiles']=sum(v['diagnostic']['geometry_encodings'] for v in checks)
            destination=root/'merge_input_recovery'/dataset/('s'+str(shard))
            if destination.exists():
                raise RuntimeError('Existing adapter copy; do not overwrite.')
            destination.mkdir(parents=True)
            save(destination/'results.json',row)
            checked.append(destination)

        # Reuse the original coverage/reference/heldout/scalar checks, substituting
        # only merge input documents. Original worker output files stay untouched.
        def adapted_merge(paths,output):
            if paths!=[str(f) for f in folders]:
                raise RuntimeError('Unexpected merge input scope.')
            merged=merge([str(f) for f in checked],output)
            merged['shards']=[str(f) for f in folders]
            merged['metadata_recovery']=dict(error='Missing diagnostics.tiles',
                tile_source='Sum of saved per-image geometry_encodings; no inference rerun.',
                adapter_shards=[str(f) for f in checked])
            save(Path(output),merged)
            return merged
        queue.merge=adapted_merge
        path=verify_dataset(root,dataset,entry)
        row=queue.read_json(Path(path))
        row['diagnostics'][dataset]['changed_pixels']=row['deployment_changed_pixels']
        save(Path(path),row)
        completed[dataset]=path
    save(result,dict(status='complete',completed=completed,full_five_protocols=len(completed)==5,
        original_controller_failures=state['failures'],evaluation_relaunched=False,
        original_outputs_preserved=True,metadata_merge_recovery=True))
    print(json.dumps(dict(recovered=list(completed),output=str(result))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',default=list(queue.ORDER))
    args=p.parse_args()
    main(args.root,args.datasets)
