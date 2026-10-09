"""Finite frozen eight-domain full evaluation, then isolated serial timing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.shared_rival_soft_alias import CONFIG, IMPLEMENTATION, METHODS, PRIMARY
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from merge_vip_official_shards import merge as merge_vip
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, read_json


TOTALS = {d: total for d, _, _, _, total in SETTINGS}
SHARDS = {'loveda': 4, 'flair1': 4}


def launch(root, dataset, gpu, phase, shard=0, shards=1, vip=False, *, evaluator=None,
           session_prefix='gsrs05'):
    if not idle(gpu):
        raise RuntimeError('GPU occupied: ' + str(gpu))
    _, data, vocabulary, _, _ = next(s for s in SETTINGS if s[0] == dataset)
    name = 'vip_all20_vaihingen' if vip else dataset
    output = root / phase / name / ('s' + str(shard))
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.log')
    session = f'{session_prefix}_{phase}_{name}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output/session; preserve it: ' + session)
    script = evaluator or ('scripts/eval_vip_official_eight.py' if vip else 'scripts/eval_shared_rival_soft_full.py')
    args = [PYTHON, '-u', script,
        '--dataset', dataset, '--dinov3-repo', str(TOOL / 'dinov3_hub'), '--checkpoint-dir', str(BASE / 'ckpt/DINO'),
        '--data-root', str(Path('/data/test/datasets') / data), '--vocabulary-config', str(TOOL / 'configs' / vocabulary),
        '--upstream-root', str(BASE / 'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
        '--num-shards', str(shards), '--shard-index', str(shard)]
    args += ['--vocabulary-source', 'json', '--finite-empty-rows'] if vip else ['--mode', phase]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd ' + shlex.quote(str(TOOL)) + ' && exec ' + shlex.join(env + args) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, name=name, gpu=gpu, phase=phase, shard=shard, shards=shards,
                output=str(output), log=str(log), session=session, vip=vip)


def verify_job(job):
    folder = Path(job['output'])
    row = read_json(folder / 'results.json')
    if not row or row['status'] != 'complete':
        raise RuntimeError('Worker exited without complete results: ' + job['session'])
    if job['phase'] == 'smoke':
        if row['target_masks_loaded'] or not row['prepared_fields_bitwise_equal'] or not row['original_endpoints_exact']:
            raise RuntimeError('Smoke equivalence failed.')
    elif job['phase'] == 'full':
        if row['processed_images'] != row['total_images']:
            raise RuntimeError('Incomplete full shard.')
        if not job['vip']:
            if not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_used_only_after_prediction']:
                raise RuntimeError('Frozen evaluation failed.')
            with np.load(folder / 'per_image_confusions.npz', allow_pickle=False) as arrays:
                if arrays['sample_keys'].tolist() != row['signature']['sample_keys']:
                    raise RuntimeError('Per-image coverage differs.')
                for p, group in row['metrics'].items():
                    for m, value in group.items():
                        if not np.array_equal(arrays[p + '__' + m].sum(0), value['confusion_matrix']):
                            raise RuntimeError('Per-image confusion sum differs.')
    elif row['target_masks_loaded'] or not row['weights_frozen']:
        raise RuntimeError('Timing loaded masks or changed weights.')
    return row


def verify_dataset(root, dataset, vip=False):
    name = 'vip_all20_vaihingen' if vip else dataset
    shards = 1 if vip else SHARDS.get(dataset, 1)
    paths = [str(root / 'full' / name / ('s' + str(s))) for s in range(shards)]
    output = root / 'full' / name / 'merged.json'
    if output.exists():
        raise RuntimeError('Merged output already exists; preserve it.')
    row = (merge_vip if vip else merge)(paths, str(output))
    old = read_json(TOOL / 'results/rival_fine_hard20_full_20261003/full' / dataset / 'merged.json')
    if (not row['coverage_verified'] or row['processed_images'] != TOTALS[dataset]
            or row['signature']['sample_keys'] != old['signature']['sample_keys']):
        raise RuntimeError('Full unique coverage differs: ' + name)
    if not vip:
        if (row['signature']['implementation'] != IMPLEMENTATION
                or row['signature']['vocabulary'] != old['signature']['vocabulary']
                or row['signature']['checkpoints'] != old['signature']['checkpoints']):
            raise RuntimeError('Frozen identities changed: ' + name)
        for p in row['metrics']:
            for method in METHODS[:2]:
                if not np.array_equal(row['metrics'][p][method]['confusion_matrix'], old['metrics'][p][method]['confusion_matrix']):
                    raise RuntimeError('Original full endpoint differs: ' + name + '/' + p + '/' + method)
        row.update(original_full_geometry_and_noadmission_exact=True, weights_frozen=True, head_weights_unchanged=True)
        save(output, row)
    return str(output)


def monitor(root, active, pending, phase, completed, failures, *, launcher=launch,
            dataset_verifier=verify_dataset):
    finished = set()
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify_job(job)
                finished.add((job['name'], job['shard']))
                if phase == 'full':
                    if all((job['name'], s) in finished for s in range(job['shards'])):
                        completed[job['name']] = dataset_verifier(root, job['dataset'], job['vip'])
                else:
                    completed[job['name']] = job['output'] + '/results.json'
            except Exception as error:
                failures[job['session']] = dict(error=str(error), log_tail=Path(job['log']).read_text()[-6000:])
            del active[gpu]
        if failures:
            pending.clear()
        for dataset, shard, shards, vip in list(pending):
            allowed = (7,) if phase == 'benchmark' else range(4)
            gpu = next((g for g in allowed if g not in active and idle(g)), None)
            if gpu is None:
                break
            job = launcher(root, dataset, gpu, phase, shard, shards, vip)
            active[gpu] = job
            pending.remove((dataset, shard, shards, vip))
        save(root / 'suite_status.json', dict(status='running' if active or pending else 'failed' if failures else 'complete',
            phase=phase, active=list(active.values()), pending=pending, completed=completed, failures=failures))
        if active or pending:
            time.sleep(15)


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if root.exists() or any(not idle(g) for g in range(8)):
        raise RuntimeError('Existing suite or occupied GPU; no launch.')
    root.mkdir(parents=True)
    save(root / 'protocol.json', dict(implementation=IMPLEMENTATION, primary=PRIMARY, methods=METHODS,
        admission=asdict(CONFIG), totals=TOTALS, fixed_aliases_per_class=20, fine_forwards=0,
        rule='positive wide margin times minimum native/Geometry contradiction; canonical protection; normalized weighted LME',
        hard_control='same risk/source, weight>=0.5', shuffle_control='same within-class spectrum, fixed canonical slots',
        target_label_tuning=False, development_status='Rule motivated by prior developed evaluations; exploratory full validation.',
        vip='Reuse finite All20 on seven domains; fill missing corrected-input Vaihingen All20 only; timing matched frozen20.'))
    started = time.time()
    failures, smoke = {}, {}
    active = {g: launch(root, setting[0], g, 'smoke') for g, setting in enumerate(SETTINGS)}
    monitor(root, active, [], 'smoke', smoke, failures)
    save(root / 'smoke_summary.json', dict(completed=smoke, failures=failures))
    if failures:
        save(root / 'suite_results.json', dict(status='failed', phase='smoke', failures=failures))
        return
    full = {}
    active = {g: launch(root, d, g, 'full') for g, d in enumerate(('udd5', 'vdd', 'potsdam', 'oem'))}
    active.update({g: launch(root, 'flair1', g, 'full', g - 4, 4) for g in range(4, 8)})
    pending = [('loveda', s, 4, False) for s in range(4)] + [
        ('vaihingen', 0, 1, False), ('landcoverai', 0, 1, False), ('vaihingen', 0, 1, True)]
    monitor(root, active, pending, 'full', full, failures)
    save(root / 'full_summary.json', dict(completed=full, failures=failures))
    if failures:
        save(root / 'suite_results.json', dict(status='failed', phase='full', completed=full, failures=failures))
        return
    timing = {}
    pending = [(s[0], 0, 1, False) for s in SETTINGS]
    monitor(root, {}, pending, 'benchmark', timing, failures)
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete', completed=full,
        timing=timing, failures=failures, unique_full_images=sum(TOTALS.values()), suite_wall_seconds=time.time() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
