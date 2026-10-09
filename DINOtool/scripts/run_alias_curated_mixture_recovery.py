"""Resume the untouched curated GPU phase after verified parent aggregation.

Preserve the original failed prerequisite state and all source/model rules.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

from eval_rival_fine_full import save
from run_alias_curated_mixture import reserve_dispatcher
from run_canonical_rival_trial import pause_dispatcher, resume_dispatcher
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def main(root):
    if (root/'suite_recovery_status.json').exists() or (root/'ade150').exists() or (root/'ade150.log').exists():
        raise RuntimeError('Only an untouched curated GPU phase can recover; preserve outputs.')
    original=read_json(root/'suite_results.json')
    if not original or original.get('status')!='failed' or original.get('phase')!='prerequisite':
        raise RuntimeError('Unexpected original controller state.')
    protocol=read_json(root/'protocol.json');parent=Path(protocol['prerequisite_root'])
    recovered=read_json(parent/'suite_aggregation_recovery.json')
    if not recovered or recovered.get('status')!='complete' or recovered.get('predictions_rerun'):
        raise RuntimeError('Parent aggregation recovery not verified.')
    # tmux treats an unprefixed target as a possible session-name prefix. The
    # exact target avoids mistaking this recovery controller for its predecessor.
    if alive('=galm08_controller') or alive('=gacm08_controller'):
        raise RuntimeError('An original predecessor/controller is still live.')
    reserve_dispatcher(root)
    output=root/'ade150';log=root/'ade150.log';session='gacm08_full_ade150'
    try:
        time.sleep(10)
        while True:
            pause_dispatcher(root)
            gpu=next((g for g in (7,6,5,4,3,2,1,0) if idle(g)),None)
            if gpu is not None:break
            save(root/'suite_recovery_status.json',dict(status='waiting',phase='idle_gpu'))
            time.sleep(10)
        entry=protocol['datasets']['ade150']
        if output.exists() or log.exists() or alive(session):
            raise RuntimeError('Existing curated output/worker; preserve it.')
        args=[PYTHON,'-u','scripts/eval_alias_curated_mixture.py','--dataset','ade150',
            '--suite-root',str(root),'--output-dir',str(output),'--data-root',entry['data_root'],
            '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
            '--checkpoint-dir',str(BASE/'ckpt/DINO'),'--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee')]
        env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
             'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
        shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
        while alive(session):
            save(root/'suite_recovery_status.json',dict(status='running',phase='curated20',gpu=gpu,worker=session))
            time.sleep(10)
        worker=read_json(output/'worker_status.json');result=read_json(output/'merged.json')
        valid=bool(worker and worker.get('status')=='complete' and result and result.get('coverage_verified')
            and result.get('curated20_identity_verified') and result.get('original_curated_inputs_verified')
            and result.get('processed_images')==2000)
        save(root/'suite_recovery_results.json',dict(status='complete' if valid else 'failed',
            result=str(output/'merged.json') if valid else None,prediction_rules_unchanged=True,
            original_prerequisite_failure_preserved=True,log_tail=None if valid else log.read_text(errors='replace')[-6000:]))
    finally:
        resume_dispatcher(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
