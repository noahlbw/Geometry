"""Finite prepare/smoke/full PC60 queue, preserving all other evaluations."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
from run_evidence_adaptive_readout import PYTHON,TOOL,BASE,THIRD,OLD,idle,alive,read_json
from eval_residual_ontology import METHODS
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge


def launch(root,entry,gpu,phase,shard):
    output=root/phase/('s'+str(shard))
    log=output.with_suffix('.log')
    session=f'gro07_{phase}_s{shard}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing run/output or occupied GPU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[PYTHON,'-u','scripts/eval_residual_ontology.py','--mode',phase,'--suite-root',str(root),
        '--data-root',entry['data_root'],'--output-dir',str(output),'--original-cache',str(OLD/'text_cache/context60.pt'),
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),
        '--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(output=str(output),log=str(log),session=session,phase=phase,shard=shard,gpu=gpu)


def verify(job):
    result=read_json(Path(job['output'])/'results.json')
    if not result or result['status']!='complete' or result['processed_images']!=result['total_images'] or not result['weights_frozen'] or not result['head_weights_unchanged']:
        raise RuntimeError('Incomplete/frozen-state verification failed.')
    if job['phase']!='full' and result['target_masks_loaded']:
        raise RuntimeError('Unexpected masks during preparation/smoke.')
    if job['phase']=='full':
        with np.load(Path(job['output'])/'per_image_confusions.npz') as data:
            if data['sample_keys'].tolist()!=result['signature']['sample_keys']:
                raise RuntimeError('Sample identity mismatch.')
            for method in METHODS:
                if not np.array_equal(data['context60__'+method].sum(0),result['metrics']['context60'][method]['confusion_matrix']):
                    raise RuntimeError('Per-image sums differ.')


def main(root):
    if (root/'suite_status.json').exists():
        raise RuntimeError('Existing suite.')
    root.resolve().relative_to((TOOL/'results').resolve())
    entry=read_json(root/'protocol.json')['datasets']['context60']
    active,failed={},{}
    pending=[('prepare',0)]
    full_finished=set()
    started=time.time()
    while pending or active:
        for gpu,job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify(job)
                if job['phase']=='prepare':
                    pending.append(('smoke',0))
                elif job['phase']=='smoke':
                    pending.extend(('full',i) for i in range(entry['shards']))
                else:
                    full_finished.add(job['shard'])
            except Exception as e:
                failed[job['session']]=dict(error=str(e),log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
            del active[gpu]
        if failed:
            pending.clear()
        for phase,shard in list(pending):
            gpu=next((g for g in range(8) if g not in active and idle(g)),None)
            if gpu is None:
                break
            active[gpu]=launch(root,entry,gpu,phase,shard)
            pending.remove((phase,shard))
        save(root/'suite_status.json',dict(status='running' if active or pending else 'failed' if failed else 'complete',
            active=list(active.values()),pending=pending,failures=failed,full_finished=sorted(full_finished)))
        if active or pending:
            time.sleep(15)
    if failed:
        save(root/'suite_results.json',dict(status='failed',failures=failed))
        return
    if full_finished!=set(range(entry['shards'])):
        raise RuntimeError('Incomplete full workers.')
    row=merge([str(root/'full'/('s'+str(i))) for i in range(entry['shards'])],str(root/'merged.json'))
    old=read_json(TOOL/'results/taxonomy_readout_20261007/full/context60/merged.json')
    expected=np.asarray(old['metrics']['context60']['TaxonomyAdaptive']['confusion_matrix'])
    if not np.array_equal(row['metrics']['context60']['TaxonomyAdaptive']['confusion_matrix'],expected):
        raise RuntimeError('Unchanged automatic route failed full exact replay.')
    if row['signature']['checkpoints']!=old['signature']['checkpoints']:
        raise RuntimeError('Checkpoint identity changed.')
    for method in METHODS:
        if not np.array_equal(np.asarray(row['metrics']['context60'][method]['confusion_matrix']).sum(1),expected.sum(1)):
            raise RuntimeError('Paired target counts differ.')
    if row['processed_images']!=5105 or not row['coverage_verified']:
        raise RuntimeError('Incomplete5105 unique coverage.')
    row.update(exact_auto_confusion_replay=True,paired_scored_targets_equal=True)
    save(root/'merged.json',row)
    save(root/'suite_results.json',dict(status='complete',merged=str(root/'merged.json'),failures={},
        suite_wall_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root)
