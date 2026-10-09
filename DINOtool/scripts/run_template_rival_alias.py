"""Deferred frozen template-rival trial, preserving existing GPU workers."""
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


def launch(root,dataset,gpu):
    entry = read_json(root/'protocol.json')['datasets'][dataset]
    output = root/dataset;log = root/(dataset+'.log');session = 'gtra08_full_'+dataset
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args = [PYTHON,'-u','scripts/eval_template_rival_alias.py','--dataset',dataset,
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
    prerequisite = Path(protocol['prerequisite_root'])
    while True:
        recovering = bool(read_json(prerequisite/'recovery_controller.json'))
        finished = read_json(prerequisite/('suite_recovery_results.json' if recovering else 'suite_results.json'))
        if finished:
            if finished.get('status')!='complete':
                save(root/'suite_results.json',dict(status='failed',phase='prerequisite',failure=finished))
                return
            break
        if not alive('gacm08_controller') and not alive('gacm08_controller_recovery'):
            logfile=prerequisite/('controller_recovery.log' if recovering else 'controller.log')
            save(root/'suite_results.json',dict(status='failed',phase='prerequisite',
                error='Curated controller exited before a complete suite.',
                log_tail=logfile.read_text(errors='replace')[-6000:]))
            return
        save(root/'suite_status.json',dict(status='waiting',phase='curated_suite'))
        time.sleep(10)
    while alive('gacm08_controller') or alive('gacm08_controller_recovery'):
        time.sleep(2)
    for dataset,entry in protocol['datasets'].items():
        previous = read_json(Path(entry['exact_base_reference']))
        if not (previous and previous.get('coverage_verified')
                and previous.get('paired_scored_targets_equal')
                and previous.get('processed_images')==entry['total_images']):
            save(root/'suite_results.json',dict(status='failed',phase='baseline',dataset=dataset,
                error='Same-input predecessor incomplete; preserve outputs.'))
            return
    reserve_dispatcher(root)
    pending = list(protocol['datasets']);active = {};completed = {};failures = {}
    try:
        time.sleep(10)
        while pending or active:
            pause_dispatcher(root)
            for gpu,job in list(active.items()):
                if alive(job['session']):continue
                folder = Path(job['output']);worker = read_json(folder/'worker_status.json')
                result = read_json(folder/'merged.json')
                if (worker and worker.get('status')=='complete' and result and result.get('coverage_verified')
                        and result.get('exact_base_confusion_replayed')
                        and result.get('processed_images')==protocol['datasets'][job['dataset']]['total_images']):
                    completed[job['dataset']] = str(folder/'merged.json')
                else:
                    failures[job['session']] = dict(job=job,
                        log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
                del active[gpu]
            if failures:pending.clear()
            for dataset in list(pending):
                gpu = next((g for g in (7,6,5,4,3,2,1,0) if g not in active and idle(g)),None)
                if gpu is None:break
                active[gpu] = launch(root,dataset,gpu);pending.remove(dataset)
            save(root/'suite_status.json',dict(status='running',pending=pending,active=list(active.values()),
                completed=completed,failures=failures))
            if pending or active:time.sleep(10)
        save(root/'suite_results.json',dict(status='failed' if failures else 'complete',
            completed=completed,failures=failures))
    finally:
        resume_dispatcher(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
