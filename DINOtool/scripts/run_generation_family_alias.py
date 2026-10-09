"""Schedule one mask-free benchmark, then four frozen full ADE shards."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_alias_curated_mixture import reserve_dispatcher
from run_canonical_rival_trial import pause_dispatcher, resume_dispatcher
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root,gpu,mode,shard=0):
    entry = read_json(root/'protocol.json')['datasets']['ade150']
    folder = root/'benchmark/ade150' if mode=='benchmark' else root/'full/ade150'/('s'+str(shard))
    log = root/(mode+'_s'+str(shard)+'.log')
    session = 'ggfa08_'+mode+'_ade150_s'+str(shard)
    if folder.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args = [PYTHON,'-u','scripts/eval_generation_family_alias.py','--dataset','ade150',
        '--suite-root',str(root),'--output-dir',str(folder),'--mode',mode,
        '--data-root',entry['data_root'],'--vocabulary-config',entry['vocabulary_config'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee')]
    if mode=='full':args += ['--num-shards','4','--shard-index',str(shard)]
    env = ['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(gpu=gpu,session=session,output=str(folder),log=str(log),shard=shard,mode=mode)


def reference_image_hashes(paths,method):
    values = {}
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            keys = archive['sample_keys'].tolist();cms=archive['ade150__'+method]
            for key,cm in zip(keys,cms):
                if key in values:raise RuntimeError('Duplicate reference archive key.')
                values[key] = hashlib.sha256(cm.astype(np.int64,copy=False).tobytes()).hexdigest()
    return values


def verify(root,protocol):
    shards = [root/'full/ade150'/('s'+str(i)) for i in range(4)]
    path = root/'full/ade150/merged.json'
    if path.exists():raise RuntimeError('Existing merged output; preserve it.')
    row = merge([str(p) for p in shards],str(path),diagnostic_weight='images')
    references = {'CC':(Path(protocol['datasets']['ade150']['exact_base_reference']),'Fixed20_TaskCoupled'),
                  'HH':(Path(protocol['historical_result']),'Frozen')}
    for method,(refpath,refmethod) in references.items():
        ref = read_json(refpath)
        if (not ref or not ref.get('coverage_verified') or ref['processed_images']!=2000
                or ref['signature']['sample_keys']!=row['signature']['sample_keys']
                or ref['signature']['checkpoints']!=row['signature']['checkpoints']
                or ref['signature']['classes']!=row['signature']['classes']):
            raise RuntimeError('Full reference coverage/source identity differs: '+method)
        if ref['metrics']['ade150'][refmethod]['confusion_matrix']!=row['metrics']['ade150'][method]['confusion_matrix']:
            raise RuntimeError('Full exact reference confusion differs: '+method)
        oldpaths = [Path(p).parent/'per_image_confusions.npz' for p in ref['shards']]
        expected = reference_image_hashes(oldpaths,refmethod)
        actual = reference_image_hashes([p/'per_image_confusions.npz' for p in shards],method)
        if expected!=actual:raise RuntimeError('Full per-image reference confusions differ: '+method)
    targets = [np.asarray(item['confusion_matrix']).sum(-1) for item in row['metrics']['ade150'].values()]
    if any(not np.array_equal(targets[0],t) for t in targets[1:]):
        raise RuntimeError('Paired scored target counts differ.')
    row.update(exact_cc_confusion_replayed=True,exact_historical_hh_confusion_replayed=True,
               exact_per_image_cc_hh_replayed=True,paired_scored_targets_equal=True)
    save(path,row)
    return row


def main(root):
    if (root/'suite_status.json').exists() or (root/'suite_results.json').exists():
        raise RuntimeError('Existing controller state; preserve it.')
    protocol = read_json(root/'protocol.json')
    reserve_dispatcher(root)
    active={};completed=[];failures={}
    try:
        time.sleep(10)
        while True:
            pause_dispatcher(root)
            gpu = next((g for g in (7,6,5,4,3,2,1,0) if idle(g)),None)
            if gpu is not None:break
            save(root/'suite_status.json',dict(status='waiting',phase='benchmark_idle_gpu'))
            time.sleep(10)
        job = launch(root,gpu,'benchmark')
        while alive(job['session']):
            save(root/'suite_status.json',dict(status='running',phase='mask_free_benchmark',active=[job]))
            time.sleep(10)
        worker = read_json(Path(job['output'])/'worker_status.json')
        timing = read_json(Path(job['output'])/'timing.json')
        if worker and worker.get('status')=='cost_rejected':
            save(root/'suite_results.json',dict(status='complete',outcome='cost_rejected',
                target_masks_loaded=False,timing=timing));return
        if not worker or worker.get('status')!='complete' or not timing or not timing.get('cost_gate_passed'):
            raise RuntimeError('Benchmark worker exited before passing; exact log:\n'+Path(job['log']).read_text(errors='replace')[-7000:])
        pending=list(range(4))
        while pending or active:
            pause_dispatcher(root)
            for gpu,job in list(active.items()):
                if alive(job['session']):continue
                folder=Path(job['output']);worker=read_json(folder/'worker_status.json');row=read_json(folder/'results.json')
                if (worker and worker.get('status')=='complete' and row and row.get('status')=='complete'
                        and row['processed_images']==row['total_images']==500):
                    completed.append(job['shard'])
                else:
                    failures[job['session']]=dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
                del active[gpu]
            if failures:pending.clear()
            for shard in list(pending):
                gpu=next((g for g in (7,6,5,4,3,2,1,0) if g not in active and idle(g)),None)
                if gpu is None:break
                active[gpu]=launch(root,gpu,'full',shard);pending.remove(shard)
            save(root/'suite_status.json',dict(status='running',phase='full',active=list(active.values()),
                pending=pending,completed=completed,failures=failures))
            if pending or active:time.sleep(10)
        if failures:
            save(root/'suite_results.json',dict(status='failed',failures=failures));return
        save(root/'suite_status.json',dict(status='running',phase='merge_verify',completed=completed))
        row=verify(root,protocol)
        save(root/'suite_results.json',dict(status='complete',outcome='full_verified',images=row['processed_images'],
            merged=str(root/'full/ade150/merged.json'),exact_cc_and_hh_per_image_replay=True))
    except Exception:
        save(root/'suite_results.json',dict(status='failed',phase='controller',error=traceback.format_exc(),
            active=list(active.values()),completed=completed,failures=failures))
        raise
    finally:
        resume_dispatcher(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
