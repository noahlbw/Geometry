"""Collect all eight vocabulary-only semantic sources on idle physical GPUs0-7."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

from dinotool.semantic_role_admission import CONFIG, IMPLEMENTATION, SOURCE
from run_bounded_alias_suite import save
from run_pair_context_reader import ORDER
from run_region_semantic_suite_a800 import TOOL, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def main(root):
    if root.exists() or any(not idle(g) for g in range(8)):
        raise RuntimeError('New root and eight idle authorized GPUs required.')
    root.mkdir(parents=True)
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'config': asdict(CONFIG),
        'source': str(SOURCE), 'datasets': ORDER, 'text_only': True, 'target_masks_loaded': False})
    started, jobs = time.perf_counter(), []
    for gpu, dataset in enumerate(ORDER):
        session, output, log = 'gsrs03_'+dataset, root/(dataset+'.json'), root/(dataset+'.log')
        if alive(session) or not idle(gpu):
            raise RuntimeError('Source GPU/session changed.')
        env = ['env', 'CUDA_VISIBLE_DEVICES='+str(gpu), 'PYTHONPATH='+str(TOOL)+':'+str(TOOL)+'/scripts',
            'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
        args = [PYTHON, '-u', 'scripts/collect_semantic_roles.py', '--vocabularies',
            str(TOOL/'results/class_attachment_admission_20261003'/dataset/'vocabularies.json'), '--output', str(output)]
        shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
        jobs.append({'dataset': dataset, 'gpu': gpu, 'session': session, 'output': str(output), 'log': str(log)})
    save(root/'jobs.json', jobs)
    active, completed, failures = list(jobs), [], {}
    while active:
        for job in active[:]:
            if alive(job['session']):
                continue
            row = read_json(Path(job['output']))
            if (not row or row['status'] != 'complete' or row['implementation'] != IMPLEMENTATION
                    or row['config'] != asdict(CONFIG) or not row['weights_frozen']
                    or row['target_images_loaded'] or row['target_masks_loaded']):
                failures[job['dataset']] = Path(job['log']).read_text()[-4000:]
            elif not row['sentinel_contract_passed']:
                failures[job['dataset']] = {'sentinel_contract': row['sentinels']}
            else:
                completed.append(job['dataset'])
            active.remove(job)
        state = {'status': 'running' if active else 'failed' if failures else 'complete',
            'active': active, 'completed': completed, 'failures': failures, 'elapsed_seconds': time.perf_counter()-started}
        save(root/'suite_status.json', state)
        if active:
            time.sleep(10)
    save(root/'suite_results.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
