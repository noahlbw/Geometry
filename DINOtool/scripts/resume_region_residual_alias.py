"""Preserve completed/running workers; retry only unlaunched idle-GPU work.

Separate recovery controller: original failed status/log and frozen model/source
contract remain untouched. No existing worker output is restarted or replaced.
"""
import argparse
import hashlib
from pathlib import Path
import time
import traceback

import run_region_residual_alias as scheduler
from eval_rival_fine_full import save


def inventory(root,protocol,mode):
    complete=[];active=[];pending=[]
    for d,n in protocol['full_shards'].items():
        for i in range(n):
            folder=root/mode/d/('s'+str(i));log=root/(mode+'_'+d+'_s'+str(i)+'.log')
            session='grra09_'+mode+'_'+d+'_s'+str(i)
            row=dict(dataset=d,shard=i,mode=mode,output=str(folder),log=str(log),session=session)
            if scheduler.alive(session):active.append(row);continue
            worker=scheduler.read_json(folder/'worker_status.json')
            result=scheduler.read_json(folder/('prepass.json' if mode=='prepass' else 'results.json'))
            if worker and worker.get('status')=='complete' and result and result.get('status')=='complete':
                if result['processed_images']!=result['total_images']:raise RuntimeError('Incomplete terminal coverage: '+session)
                complete.append(row)
            elif folder.exists() or log.exists():raise RuntimeError('Exited incomplete worker; preserve and inspect: '+session)
            else:pending.append(row)
    return complete,active,pending


def main(root):
    state_path=root/'suite_resume_status.json'
    if state_path.exists() or (root/'suite_results.json').exists():raise RuntimeError('Existing recovery state; preserve it.')
    protocol=scheduler.read_json(root/'protocol.json');phase='mask_free_prepass'
    try:
        for d in protocol['datasets']:
            if not (scheduler.read_json(root/'benchmark'/d/'timing.json') or {}).get('cost_gate_passed'):
                raise RuntimeError('Both complete cost gates required.')
        while True:
            complete,active,pending=inventory(root,protocol,'prepass' if phase=='mask_free_prepass' else 'full')
            if not active and not pending:
                if phase=='mask_free_prepass':
                    for d in protocol['datasets']:
                        path=root/'prepass'/d;mapping=path/'image_mapping.json';archive=path/'all_image_q.npz'
                        if mapping.exists() and archive.exists():
                            row=scheduler.read_json(mapping)
                            if not row.get('mask_free_complete') or row['archive_sha256']!=hashlib.sha256(archive.read_bytes()).hexdigest():
                                raise RuntimeError('Existing frozen image mapping differs.')
                        else:scheduler.freeze_mapping(root,protocol,d)
                    phase='full_evaluation';continue
                merged={d:dict(merged=str(root/'full'/d/'merged.json'),processed_images=scheduler.verify(root,protocol,d)['processed_images'])
                        for d in protocol['datasets']}
                save(root/'suite_results.json',dict(status='complete',outcome='full_verified',datasets=merged,
                    image_maps={d:str(root/'prepass'/d/'image_mapping.json') for d in protocol['datasets']},
                    processed_images=sum(row['processed_images'] for row in merged.values()),
                    recovery_controller='preserved original failure and all existing workers'))
                save(state_path,dict(status='complete',phase=phase));return
            for gpu in range(8):
                if not pending:break
                if not scheduler.idle(gpu):continue
                job=pending[0]
                try:new=scheduler.launch(root,job['dataset'],gpu,job['mode'],job['shard'])
                except RuntimeError:
                    folder=Path(job['output']);log=Path(job['log'])
                    if folder.exists() or log.exists() or scheduler.alive(job['session']):raise
                    # The second occupancy check may change after the first;
                    # leave the queue intact and retry an idle GPU later.
                    continue
                active.append(new);pending.pop(0)
            save(state_path,dict(status='running',phase=phase,active=active,pending=pending,completed=complete))
            time.sleep(5)
    except Exception:
        save(state_path,dict(status='failed',phase=phase,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
