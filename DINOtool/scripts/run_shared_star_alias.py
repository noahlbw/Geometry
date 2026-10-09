"""Reserve two idle GPUs, then run bounded VDD/ADE experiments to completion."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from eval_rival_fine_full import save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json

PREFIX = 'gstar08r2'


def resume_scheduler(root):
    pause = read_json(root/'scheduler_pause.json')
    if not pause or pause.get('resumed'):
        return
    pid = pause['pid']
    path = Path('/proc')/str(pid)/'cmdline'
    if path.exists():
        command = path.read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/run_alias_full_comparison_r3.py' not in command or pause['root'] not in command:
            raise RuntimeError('Original scheduler PID identity changed; inspect safely.')
        os.kill(pid,signal.SIGCONT)
    pause.update(resumed=True,resumed_at=time.time());save(root/'scheduler_pause.json',pause)


def launch(root,d,entry,gpu):
    folder=root/d;session=PREFIX+'_'+d;log=root/(d+'.log')
    if folder.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Occupied GPU or existing output: '+session)
    args=[PYTHON,'-u','scripts/eval_shared_star_alias.py','--dataset',d,'--suite-root',str(root),
        '--data-root',entry['data_root'],'--vocabulary-config',entry['vocabulary_config'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),'--output-dir',str(folder)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,gpu=gpu,session=session,output=str(folder),log=str(log))


def main(root):
    if (root/'suite_status.json').exists():
        raise RuntimeError('Existing controller state; preserve it.')
    protocol=read_json(root/'protocol.json');pending=list(protocol['datasets']);active={};done={};failures={}
    try:
        while pending or active:
            for gpu,job in list(active.items()):
                if alive(job['session']):continue
                row=read_json(Path(job['output'])/'worker_status.json')
                result=read_json(Path(job['output'])/'merged.json')
                if row and row.get('status')=='complete' and result and result.get('coverage_verified'):
                    done[job['dataset']]=str(Path(job['output'])/'merged.json')
                else:
                    failures[job['session']]=dict(log=job['log'],log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
                del active[gpu]
            if failures:pending.clear()
            for d in list(pending):
                gpu=next((g for g in range(8) if g not in active and idle(g)),None)
                if gpu is None:break
                active[gpu]=launch(root,d,protocol['datasets'][d],gpu);pending.remove(d)
                # Worker acquires its CUDA lease before loading/checking models.
                # Do not resume the old dispatcher before its occupancy is visible.
            acquired=all(read_json(Path(job['output'])/'worker_status.json') for job in active.values())
            if not pending and acquired:resume_scheduler(root)
            save(root/'suite_status.json',dict(status='running',phase='reserving_gpus' if pending else 'evaluation',
                pending=pending,active=list(active.values()),completed=done,failures=failures))
            if pending or active:time.sleep(10)
    finally:
        resume_scheduler(root)
    save(root/'suite_results.json',dict(status='failed' if failures else 'complete',completed=done,failures=failures))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
