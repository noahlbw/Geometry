"""One dispatcher for recovered timing and the two frozen alias trials.

Only the incumbent dispatcher's verified process group is suspended. Its GPU
workers are independent tmux groups and continue normally.
"""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from eval_rival_fine_full import save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


PREFIX = 'gcrp08'
OLD_SESSION = 'gafc08r3_controller'
OLD_ROOT = str(TOOL/'results/alias_full_comparison_20261008_r3')


def pause_dispatcher(root):
    record = root/'scheduler_pause.json'
    pause = read_json(record)
    if pause and pause.get('mode') == 'dispatcher_exited_workers_preserved':
        if alive(OLD_SESSION):
            raise RuntimeError('Incumbent dispatcher unexpectedly restarted during reservation.')
        return
    if not alive(OLD_SESSION):
        return
    pid = int(subprocess.check_output(['tmux','display-message','-p','-t',OLD_SESSION,'#{pane_pid}'],text=True))
    command = (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'scripts/run_alias_full_comparison_r3.py' not in command or OLD_ROOT not in command or os.getpgid(pid) != pid:
        raise RuntimeError('Dispatcher process-group identity changed; inspect safely.')
    pause = read_json(record)
    if pause and pause['pid'] != pid:
        raise RuntimeError('Dispatcher PID changed during reservation.')
    # Stop the whole dispatcher group, including in-flight probe/launch children.
    # Independently created worker sessions are not members of this group.
    os.killpg(pid,signal.SIGSTOP)
    for _ in range(20):
        state = (Path('/proc')/str(pid)/'status').read_text()
        if any(line.startswith('State:') and line.split(':',1)[1].strip().split()[0] in ('T','t') for line in state.splitlines()):
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Dispatcher did not stop.')
    if not pause:
        save(record,dict(pid=pid,pgid=pid,root=OLD_ROOT,resumed=False,group_pause=True))


def resume_dispatcher(root):
    path = root/'scheduler_pause.json';pause = read_json(path)
    if not pause or pause.get('resumed'):
        return
    if pause.get('mode') == 'dispatcher_exited_workers_preserved':
        if not alive(OLD_SESSION) and not read_json(Path(OLD_ROOT)/'suite_results.json'):
            command = [PYTHON,'-u','scripts/run_alias_full_comparison_resume_r3.py','--root',OLD_ROOT]
            env = ['env',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}','OMP_NUM_THREADS=2',
                   'MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
            log = root/'incumbent_resume.log'
            shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+command)+' > '+shlex.quote(str(log))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',OLD_SESSION,shell],check=True)
        save(path,dict(pause,resumed=True,resumed_at=time.time(),resume_preserves_existing_workers=True))
        return
    process = Path('/proc')/str(pause['pid'])
    if process.exists():
        command = (process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/run_alias_full_comparison_r3.py' not in command or pause['root'] not in command:
            raise RuntimeError('Dispatcher identity changed; cannot resume safely.')
        os.killpg(pause['pgid'],signal.SIGCONT)
    save(path,dict(pause,resumed=True,resumed_at=time.time()))


def launch(root,d,gpu,*,recovery=False):
    protocol = read_json(root/'protocol.json');entry = protocol['datasets'][d]
    suite = Path(protocol['star_root']) if recovery else root
    destination = Path(protocol['star_timing_recovery'])/d if recovery else root/d
    session = PREFIX+('_timing_' if recovery else '_full_')+d
    log = root/(('timing_' if recovery else '')+d+'.log')
    if destination.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    script = 'scripts/benchmark_shared_star_alias.py' if recovery else 'scripts/eval_canonical_rival_trial.py'
    command = [PYTHON,'-u',script,'--dataset',d,'--suite-root',str(suite),
        '--data-root',entry['data_root'],'--vocabulary-config',entry['vocabulary_config'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),'--output-dir',str(destination)]
    env = ['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+command)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,gpu=gpu,session=session,output=str(destination),log=str(log),recovery=recovery)


def main(root, resume=False):
    previous = read_json(root/'suite_status.json')
    if previous and not resume:
        raise RuntimeError('Existing controller state; preserve it.')
    protocol = read_json(root/'protocol.json')
    recovery_root = Path(protocol['star_timing_recovery']);recovery_root.mkdir(exist_ok=resume)
    if resume:
        if not previous or previous.get('active') or previous.get('completed') or previous.get('failures'):
            raise RuntimeError('Only untouched waiting queue can resume after scheduling repair.')
        recovery = previous['timing_pending'];pending = previous['pending']
    else:
        recovery = [d for d in protocol['datasets'] if d not in protocol.get('star_timing_reused',{})]
        pending = list(protocol['datasets'])
    active = {};done = {};failures = {}
    for d,folder in protocol.get('star_timing_reused',{}).items():
        result = read_json(Path(folder)/'results.json')
        if not result or result.get('status') != 'complete' or not result.get('exclusive_gpu_verified'):
            raise RuntimeError('Reused exclusive timing unverified: '+d)
        done['timing_'+d] = folder
    try:
        pause_dispatcher(root)
        # Let an already accepted tmux IPC launch become visible before any
        # new idle decision; the stopped dispatcher cannot issue new requests.
        time.sleep(10)
        while recovery or pending or active:
            # Never trust a pause flag alone; check the real dispatcher state.
            pause_dispatcher(root)
            for gpu,job in list(active.items()):
                if alive(job['session']):
                    continue
                folder = Path(job['output'])
                result = read_json(folder/('results.json' if job['recovery'] else 'worker_status.json'))
                if (result and result.get('status') == 'complete'
                        and (result.get('exclusive_gpu_verified') if job['recovery'] else result.get('total') == protocol['datasets'][job['dataset']]['total_images'])):
                    done[('timing_' if job['recovery'] else '')+job['dataset']] = str(folder)
                else:
                    failures[job['session']] = dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
                del active[gpu]
            if failures:
                recovery.clear();pending.clear()
            if recovery and not active:
                gpu = next((g for g in range(8) if idle(g)),None)
                if gpu is not None:
                    active[gpu] = launch(root,recovery.pop(0),gpu,recovery=True)
            elif not recovery and not any(job['recovery'] for job in active.values()):
                for d in list(pending):
                    gpu = next((g for g in range(8) if g not in active and idle(g)),None)
                    if gpu is None:
                        break
                    active[gpu] = launch(root,d,gpu);pending.remove(d)
            save(root/'suite_status.json',dict(status='running',timing_pending=recovery,pending=pending,
                active=list(active.values()),completed=done,failures=failures))
            if recovery or pending or active:
                time.sleep(10)
        save(recovery_root/'suite_results.json',dict(status='failed' if any('timing' in key for key in failures) else 'complete',
            completed={k:v for k,v in done.items() if k.startswith('timing_')}))
        save(root/'suite_results.json',dict(status='failed' if failures else 'complete',completed=done,failures=failures))
    finally:
        resume_dispatcher(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args = p.parse_args()
    main(args.root,args.resume)
