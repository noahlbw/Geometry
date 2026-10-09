"""Replay the independent historical LoveDA P reference, never candidate arms."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_alias_finalization import verify_dataset, verify_job
from run_alias_finalization_cost_protocol import main as complete_cost
from run_alias_finalization_statistics import run as complete_statistics
from run_evidence_adaptive_readout import TOOL, THIRD, PYTHON, OLD, idle, alive, read_json


PRIMARY = 'Geometry_PatchOnly2Coupled'
PROTOCOL = 'P__original20'


def launch_reference(root, historical, shard, gpu):
    output = root / 'reference_replay/loveda' / ('s' + str(shard))
    log = output.with_suffix('.log')
    session = 'gaf09_reference_loveda_s' + str(shard)
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing reference output/session or occupied GPU: ' + session)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = dict(historical['signature']['config'], output_dir=str(output),
                  num_shards=8, shard_index=shard, only_protocol=PROTOCOL)
    args = [PYTHON, '-u', 'scripts/eval_curated20_patchonly2.py']
    for key, value in config.items():
        args += ['--' + key.replace('_', '-'), str(value)]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
           'TOKENIZERS_PARALLELISM=false']
    shell = 'cd ' + shlex.quote(str(TOOL)) + ' && exec ' + shlex.join(env + args)
    shell += ' > ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(shard=shard, gpu=gpu, output=str(output), log=str(log), session=session)


def verify_reference(root, historical):
    folders = [root / 'reference_replay/loveda' / ('s' + str(i)) for i in range(8)]
    target = root / 'reference_replay/loveda/merged.json'
    if target.exists():
        raise RuntimeError('Existing reference merge; preserve it.')
    row = merge([str(f) for f in folders], str(target))
    sig, old = row['signature'], historical['signature']
    if (not row['coverage_verified'] or row['processed_images'] != 1669
            or sig['global_sample_keys_sha256'] != old['global_sample_keys_sha256']
            or sig['checkpoints'] != old['checkpoints'] or sig['gear'] != old['gear']
            or sig['competitive'] != old['competitive']
            or sig['implementation'] != old['implementation']
            or sig['vocabulary']['sha256'] != old['vocabulary']['sha256']
            or sig['classes'][PROTOCOL] != old['classes'][PROTOCOL]
            or sig['vocabulary']['aliases'][PROTOCOL] != old['vocabulary']['aliases'][PROTOCOL]):
        raise RuntimeError('Historical reference identity/coverage changed.')
    expected_config = dict(old['config'])
    for field in ('output_dir', 'num_shards', 'shard_index'):
        expected_config.pop(field)
    actual_config = dict(sig['config'])
    for field in ('output_dir', 'num_shards', 'shard_index'):
        actual_config.pop(field)
    if actual_config != dict(expected_config, only_protocol=PROTOCOL):
        raise RuntimeError('Historical reference settings changed.')
    metric = row['metrics'][PROTOCOL][PRIMARY]
    if not np.array_equal(metric['confusion_matrix'], historical['metrics'][PROTOCOL][PRIMARY]['confusion_matrix']):
        raise RuntimeError('Replayed historical P aggregate confusion differs.')
    historical_images = {}
    for i in range(old['num_shards']):
        with np.load(OLD / 'full/loveda' / ('s' + str(i)) / 'per_image_confusions.npz', allow_pickle=False) as data:
            historical_images.update(zip(data['sample_keys'].tolist(), data[PROTOCOL + '__' + PRIMARY]))
    observed = set()
    for folder in folders:
        shard = read_json(folder / 'results.json')
        if not shard.get('weights_frozen') or not shard.get('head_weights_unchanged'):
            raise RuntimeError('Reference model weights changed.')
        with np.load(folder / 'per_image_confusions.npz', allow_pickle=False) as data:
            if data['sample_keys'].tolist() != shard['signature']['sample_keys']:
                raise RuntimeError('Reference shard sample order differs.')
            cms = data[PROTOCOL + '__' + PRIMARY]
            if not np.array_equal(cms.sum(0), shard['metrics'][PROTOCOL][PRIMARY]['confusion_matrix']):
                raise RuntimeError('Reference per-image sum differs.')
            for key, cm in zip(data['sample_keys'].tolist(), cms):
                if key in observed or key not in historical_images or not np.array_equal(cm, historical_images[key]):
                    raise RuntimeError('Historical P per-image confusion differs: ' + key)
                observed.add(key)
    if observed != set(historical_images) or len(observed) != 1669:
        raise RuntimeError('Historical per-image coverage differs.')
    report = dict(protocol='P', reference_merged=str(target), historical_merged=str(OLD / 'full/loveda/merged.json'),
        independent_text_cache=old['config']['text_cache'], exact_historical_per_image_confusions=True,
        global_sample_keys_sha256=sig['global_sample_keys_sha256'], images=len(observed), metric=metric,
        discrepancy='Main reference sliced D local text features; three P alias vectors differ from the independently cached historical P bank. Wide features and all candidate outputs are preserved.')
    save(root / 'reference_replay/loveda/verification.json', report)
    return report


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    state_path = root / 'reference_recovery_status.json'
    result_path = root / 'reference_recovery_results.json'
    if state_path.exists() or result_path.exists():
        raise RuntimeError('Existing reference recovery; preserve it.')
    failed = read_json(root / 'verification_results.json')
    if (not failed or failed['status'] != 'failed' or len(failed['failures']) != 1
            or next(iter(failed['failures'].values()))['error'] != 'Historical same-view original20 reference differs: loveda/P'):
        raise RuntimeError('Recovery is scoped only to the exact LoveDA P cache mismatch.')
    protocol = read_json(root / 'protocol.json')
    historical = read_json(OLD / 'full/loveda/merged.json')
    pending = list(range(8))
    active = {}
    try:
        while pending or active:
            for gpu, job in list(active.items()):
                if alive(job['session']):
                    continue
                row = read_json(Path(job['output']) / 'results.json')
                if (not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']):
                    raise RuntimeError('Reference worker failed: ' + job['session'] + '\n' + Path(job['log']).read_text(errors='replace')[-5000:])
                del active[gpu]
            for shard in list(pending):
                gpu = next((g for g in range(8) if g not in active and idle(g)), None)
                if gpu is None:
                    break
                active[gpu] = launch_reference(root, historical, shard, gpu)
                pending.remove(shard)
            save(state_path, dict(status='reference_replay', active=list(active.values()), pending=pending))
            if pending or active:
                time.sleep(15)
        reference = verify_reference(root, historical)
        for i in range(protocol['datasets']['loveda']['shards']):
            verify_job(dict(phase='full', output=str(root / 'full/loveda' / ('s' + str(i))), session='completed_candidate'))
        completed = dict(failed['completed'])
        completed['loveda'] = verify_dataset(root, 'loveda', protocol['datasets']['loveda'],
                                           preserve_existing=True, reference_override=reference)
        for d in protocol['order']:
            row = read_json(Path(completed[d]))
            if (not row.get('paired_scored_targets_equal') or not row.get('scalar_agreement_verified')
                    or not row['coverage_verified'] or row['processed_images'] != protocol['datasets'][d]['total_images']):
                raise RuntimeError('Unverified completed candidate: ' + d)
        save(result_path, dict(status='complete', completed=completed, failures={},
            candidate_inference_rerun=False, reference_only_replay=True, reference_verification=reference))
        save(state_path, dict(status='corrected_cost', active=[], pending=[]))
        complete_cost(root, prerequisite=result_path.name, state_prefix='natural_vip_cost_reference_recovery')
        save(state_path, dict(status='full_statistics', active=[], pending=[]))
        complete_statistics(root, main_result=result_path.name,
            cost_result='natural_vip_cost_reference_recovery_results.json')
        save(state_path, dict(status='complete', active=[], pending=[], completed=list(completed)))
    except Exception as error:
        save(state_path, dict(status='failed', error=str(error), active=list(active.values()), pending=pending))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
