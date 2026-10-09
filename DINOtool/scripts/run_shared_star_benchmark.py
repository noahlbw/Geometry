"""Hold the old dispatcher until two exclusive timing jobs finish."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

from eval_rival_fine_full import save
from run_region_semantic_suite_a800 import BASE,TOOL,THIRD,PYTHON,idle
from run_sat_geometry_transport_suite import alive,read_json
from run_shared_star_alias import resume_scheduler


def main(root):
    suite=root.parent;protocol=read_json(suite/'protocol.json');done={};pending=['vdd','ade150'];active=None
    try:
        while pending or active:
            if active and not alive(active['session']):
                row=read_json(Path(active['output'])/'results.json')
                if not row or row.get('status')!='complete' or not row.get('exclusive_gpu_verified'):
                    save(root/'suite_results.json',dict(status='failed',job=active,
                        log_tail=Path(active['log']).read_text(errors='replace')[-6000:]));return
                done[active['dataset']]=active['output']+'/results.json';active=None
            if pending and active is None:
                gpu=next((g for g in range(8) if idle(g)),None)
                if gpu is not None:
                    d=pending.pop(0);entry=protocol['datasets'][d];output=root/d;session='gstar08r2_bench_'+d;log=root/(d+'.log')
                    if output.exists() or log.exists() or alive(session):raise RuntimeError('Existing timing attempt.')
                    args=[PYTHON,'-u','scripts/benchmark_shared_star_alias.py','--dataset',d,'--suite-root',str(suite),
                        '--data-root',entry['data_root'],'--vocabulary-config',entry['vocabulary_config'],
                        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
                        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),'--output-dir',str(output)]
                    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
                        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
                    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
                    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
                    active=dict(dataset=d,gpu=gpu,session=session,output=str(output),log=str(log))
            save(root/'suite_status.json',dict(status='running',pending=pending,active=active,completed=done))
            if pending or active:time.sleep(10)
        save(root/'suite_results.json',dict(status='complete',completed=done))
    finally:resume_scheduler(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
