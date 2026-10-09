"""Schedule the frozen two-domain sparse-fine trial on naturally idle GPUs."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

from eval_rival_fine_full import save
from run_alias_curated_mixture import reserve_dispatcher
from run_canonical_rival_trial import pause_dispatcher,resume_dispatcher
from run_region_semantic_suite_a800 import BASE,TOOL,THIRD,PYTHON,idle
from run_sat_geometry_transport_suite import alive,read_json


def launch(root,dataset,gpu):
    entry = read_json(root/'protocol.json')['datasets'][dataset]
    output = root/dataset;log = root/(dataset+'.log');session = 'gsrf08_full_'+dataset
    if output.exists() or log.exists() or alive('='+session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args = [PYTHON,'-u','scripts/eval_single_rival_fine_alias.py','--dataset',dataset,
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
    parent = Path(protocol['prerequisite_root'])
    previous = read_json(parent/'suite_results.json')
    if not previous or previous.get('status')!='complete' or alive('=gtra08_controller'):
        raise RuntimeError('Previous frozen trial not completely finalized.')
    for dataset,entry in protocol['datasets'].items():
        result = read_json(Path(entry['exact_base_reference']))
        if not (result and result.get('coverage_verified') and result.get('paired_scored_targets_equal')
                and result.get('processed_images')==entry['total_images']):
            raise RuntimeError('Same-input Base incomplete: '+dataset)
    reserve_dispatcher(root)
    pending = list(protocol['datasets']);active = {};completed = {};rejected = {};failures = {}
    try:
        time.sleep(10)
        while pending or active:
            pause_dispatcher(root)
            for gpu,job in list(active.items()):
                if alive('='+job['session']):continue
                folder = Path(job['output']);worker = read_json(folder/'worker_status.json') or {}
                result = read_json(folder/'merged.json');timing = read_json(folder/'timing.json')
                if (worker.get('status')=='complete' and result and result.get('coverage_verified')
                        and result.get('exact_base_confusion_replayed')
                        and result.get('processed_images')==protocol['datasets'][job['dataset']]['total_images']):
                    completed[job['dataset']] = str(folder/'merged.json')
                elif (worker.get('status')=='cost_rejected' and timing and timing.get('status')=='complete'
                        and timing.get('cost_gate_passed') is False):
                    rejected[job['dataset']] = worker
                else:
                    failures[job['session']] = dict(job=job,worker=worker,
                        log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
                del active[gpu]
            if failures:pending.clear()
            for dataset in list(pending):
                gpu = next((g for g in (7,6,5,4,3,2,1,0) if g not in active and idle(g)),None)
                if gpu is None:break
                active[gpu] = launch(root,dataset,gpu);pending.remove(dataset)
            save(root/'suite_status.json',dict(status='running',pending=pending,active=list(active.values()),
                completed=completed,cost_rejected=rejected,failures=failures))
            if pending or active:time.sleep(10)
        save(root/'suite_results.json',dict(status='failed' if failures else 'complete',
            completed=completed,cost_rejected=rejected,failures=failures))
    finally:
        resume_dispatcher(root)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
