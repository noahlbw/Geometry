"""Run the frozen two-domain trial on naturally idle physical GPUs."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

from eval_rival_fine_full import save
from run_canonical_rival_trial import pause_dispatcher, resume_dispatcher
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root,dataset,gpu):
    entry = read_json(root/'protocol.json')['datasets'][dataset]
    output = root/dataset;log = root/(dataset+'.log');session = 'galm08_full_'+dataset
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args = [PYTHON,'-u','scripts/eval_alias_logprior_mixture.py','--dataset',dataset,
        '--suite-root',str(root),'--output-dir',str(output),'--data-root',entry['data_root'],
        '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(BASE/'ckpt/DINO'),'--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee')]
    env = ['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=dataset,gpu=gpu,session=session,output=str(output),log=str(log))


def main(root):
    if (root/'suite_status.json').exists():
        raise RuntimeError('Existing controller state; preserve it.')
    protocol = read_json(root/'protocol.json')
    pending = list(protocol['datasets']);active = {};completed = {};failures = {}
    try:
        pause_dispatcher(root)
        time.sleep(10)
        while pending or active:
            pause_dispatcher(root)
            for gpu,job in list(active.items()):
                if alive(job['session']):
                    continue
                folder = Path(job['output'])
                state = read_json(folder/'worker_status.json')
                result = read_json(folder/'merged.json')
                if (state and state.get('status')=='complete' and result and result.get('coverage_verified')
                        and result.get('exact_base_confusion_replayed')
                        and result.get('processed_images')==protocol['datasets'][job['dataset']]['total_images']):
                    completed[job['dataset']] = str(folder/'merged.json')
                else:
                    failures[job['session']] = dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
                del active[gpu]
            if failures:
                pending.clear()
            for dataset in list(pending):
                gpu = next((g for g in (7,6,5,4,3,2,1,0) if g not in active and idle(g)),None)
                if gpu is None:
                    break
                active[gpu] = launch(root,dataset,gpu);pending.remove(dataset)
            save(root/'suite_status.json',dict(status='running',pending=pending,active=list(active.values()),
                completed=completed,failures=failures))
            if pending or active:
                time.sleep(10)
        save(root/'suite_results.json',dict(status='failed' if failures else 'complete',
            completed=completed,failures=failures))
    finally:
        resume_dispatcher(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
