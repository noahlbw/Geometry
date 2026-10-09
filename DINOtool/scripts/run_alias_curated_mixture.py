"""Deferred ADE input factorial after the frozen original20 suite finishes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from eval_rival_fine_full import save
from run_canonical_rival_trial import OLD_ROOT, OLD_SESSION, pause_dispatcher, resume_dispatcher
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def reserve_dispatcher(root):
    pid=None
    if alive(OLD_SESSION):
        pid=int(subprocess.check_output(['tmux','display-message','-p','-t',OLD_SESSION,'#{pane_pid}'],text=True))
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/run_alias_full_comparison_resume_r3.py' not in command or OLD_ROOT not in command or os.getpgid(pid)!=pid:
            raise RuntimeError('Incumbent dispatcher identity differs; preserve it.')
        os.killpg(pid,signal.SIGTERM)
    save(root/'scheduler_pause.json',dict(pid=pid,root=OLD_ROOT,resumed=False,
        mode='dispatcher_exited_workers_preserved',reason='Reserve frozen ADE semantic-input factorial; preserve all GPU workers.'))


def main(root):
    if (root/'suite_status.json').exists():
        raise RuntimeError('Existing deferred controller state; preserve it.')
    protocol=read_json(root/'protocol.json');parent=Path(protocol['prerequisite_root'])
    while True:
        finished=read_json(parent/'suite_results.json')
        if finished:
            if finished.get('status')!='complete':
                save(root/'suite_results.json',dict(status='failed',phase='prerequisite',failure=finished))
                return
            break
        if not alive('galm08_controller'):
            save(root/'suite_results.json',dict(status='failed',phase='prerequisite',
                error='Original20 controller exited without a complete suite.',
                log_tail=(parent/'controller.log').read_text(errors='replace')[-6000:]))
            return
        save(root/'suite_status.json',dict(status='waiting',phase='original20_suite',parent_live=True))
        time.sleep(10)
    while alive('galm08_controller'):
        # The parent's finally must complete its dispatcher restoration first.
        time.sleep(2)
    reserve_dispatcher(root)
    session='gacm08_full_ade150';output=root/'ade150';log=root/'ade150.log'
    try:
        time.sleep(10)
        while True:
            pause_dispatcher(root)
            gpu=next((g for g in (7,6,5,4,3,2,1,0) if idle(g)),None)
            if gpu is not None:break
            save(root/'suite_status.json',dict(status='waiting',phase='idle_gpu'))
            time.sleep(10)
        entry=protocol['datasets']['ade150']
        if output.exists() or log.exists() or alive(session):
            raise RuntimeError('Existing curated worker/output; preserve it.')
        args=[PYTHON,'-u','scripts/eval_alias_curated_mixture.py','--dataset','ade150',
            '--suite-root',str(root),'--output-dir',str(output),'--data-root',entry['data_root'],
            '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
            '--checkpoint-dir',str(BASE/'ckpt/DINO'),'--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee')]
        env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
            'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
        shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
        while alive(session):
            save(root/'suite_status.json',dict(status='running',phase='curated20',gpu=gpu,worker=session))
            time.sleep(10)
        worker=read_json(output/'worker_status.json');result=read_json(output/'merged.json')
        valid=bool(worker and worker.get('status')=='complete' and result and result.get('coverage_verified')
            and result.get('curated20_identity_verified') and result.get('original_curated_inputs_verified')
            and result.get('processed_images')==2000)
        save(root/'suite_results.json',dict(status='complete' if valid else 'failed',
            result=str(output/'merged.json') if valid else None,
            log_tail=None if valid else log.read_text(errors='replace')[-6000:]))
    finally:
        resume_dispatcher(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
