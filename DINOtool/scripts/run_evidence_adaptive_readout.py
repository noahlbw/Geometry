"""Five-domain frozen adaptation queue; masks only score frozen predictions."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
from eval_rival_fine_full import save
from eval_geometry_vip_reliability import summary
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json
from eval_evidence_adaptive_readout import METHODS

ORDER=('vdd','potsdam','voc21','context60','ade150')
PREFIX='gear07'
OLD=TOOL/'results/curated20_patchonly2_full_20261006'


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d if phase=='smoke' else root/phase/d/('s'+str(shard))
    log=output.with_suffix('.log')
    session=f'{PREFIX}_{phase}_{d}_s{shard}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing run or occupied GPU: '+session)
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[PYTHON,'-u','scripts/eval_evidence_adaptive_readout.py','--dataset',d,'--mode',phase,
          '--suite-root',str(root),'--data-root',entry['data_root'],'--output-dir',str(output),
          '--original-cache',str(OLD/'text_cache'/(d+'.pt')),'--dinov3-repo',str(TOOL/'dinov3_hub'),
          '--checkpoint-dir',str(BASE/'ckpt/DINO'),'--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),
          '--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,phase=phase,shard=shard,output=str(output),log=str(log),session=session,gpu=gpu)


def verify_job(job):
    row=read_json(Path(job['output'])/'results.json')
    if not row or row['status']!='complete' or row['processed_images']!=row['total_images'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
        raise RuntimeError('Incomplete worker: '+job['session'])
    if job['phase']=='smoke':
        if row['target_masks_loaded'] or not row['exact_retained_prediction']:
            raise RuntimeError('Mask-free/reference smoke failed.')
    else:
        with np.load(Path(job['output'])/'per_image_confusions.npz') as data:
            if data['sample_keys'].tolist()!=row['signature']['sample_keys']:
                raise RuntimeError('Sample mismatch.')
            for m in METHODS:
                if not np.array_equal(data[job['dataset']+'__'+m].sum(0),row['metrics'][job['dataset']][m]['confusion_matrix']):
                    raise RuntimeError('Per-image confusion sum mismatch.')


def verify_dataset(root,d,entry):
    folders=[root/'full'/d/('s'+str(i)) for i in range(entry['shards'])]
    path=root/'full'/d/'merged.json'
    if path.exists():
        raise RuntimeError('Existing merge.')
    row=merge([str(f) for f in folders],str(path))
    metrics=row['metrics'][d]
    prior=read_json(OLD/'full'/d/'merged.json')['metrics'][d+'__original20']['Geometry_PatchOnly2Coupled']
    if not row['coverage_verified'] or row['processed_images']!=entry['total_images'] or not np.array_equal(metrics['Reference']['confusion_matrix'],prior['confusion_matrix']):
        raise RuntimeError('Coverage or retained reference failed.')
    wanted=set(entry['heldout_keys'])
    held={m:np.zeros_like(metrics[m]['confusion_matrix'],dtype=np.int64) for m in METHODS}
    seen=set()
    for f in folders:
        with np.load(f/'per_image_confusions.npz') as data:
            mask=np.asarray([k in wanted for k in data['sample_keys']])
            seen.update(k for k in data['sample_keys'] if k in wanted)
            for m in METHODS:
                held[m]+=data[d+'__'+m][mask].sum(0)
    if seen!=wanted:
        raise RuntimeError('Incomplete nondevelopment complement.')
    targets=np.asarray(metrics['Reference']['confusion_matrix']).sum(1)
    for m in METHODS:
        if not np.array_equal(targets,np.asarray(metrics[m]['confusion_matrix']).sum(1)):
            raise RuntimeError('Different scored target counts.')
    row.update(exact_retained_confusion_replay=True,paired_scored_targets_equal=True,
               heldout_images=len(wanted),heldout_metrics={m:summary(cm,row['signature']['classes'][d],0) for m,cm in held.items()})
    save(path,row)
    return str(path)


def main(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if (root/'suite_status.json').exists():
        raise RuntimeError('Existing suite; do not restart.')
    info=read_json(root/'protocol.json')['datasets']
    active,completed,failures,finished={}, {}, {}, set()
    pending=[(d,'smoke',0) for d in ORDER]
    started=time.time()
    while pending or active:
        for gpu,job in list(active.items()):
            if alive(job['session']):
                continue
            d=job['dataset']
            try:
                verify_job(job)
                if job['phase']=='smoke':
                    pending.extend((d,'full',i) for i in range(info[d]['shards']))
                else:
                    finished.add((d,job['shard']))
                    if all((d,i) in finished for i in range(info[d]['shards'])):
                        completed[d]=verify_dataset(root,d,info[d])
            except Exception as e:
                failures[job['session']]=dict(error=str(e),log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
            del active[gpu]
        if failures:
            pending.clear()
        for d,phase,shard in list(pending):
            gpu=next((g for g in range(8) if g not in active and idle(g)),None)
            if gpu is None:
                break
            active[gpu]=launch(root,d,info[d],gpu,phase,shard)
            pending.remove((d,phase,shard))
        save(root/'suite_status.json',dict(status='running' if active or pending else 'failed' if failures else 'complete',
             active=list(active.values()),pending=pending,completed=completed,failures=failures))
        if active or pending:
            time.sleep(15)
    save(root/'suite_results.json',dict(status='failed' if failures else 'complete',completed=completed,failures=failures,suite_wall_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
