"""Serial eight-domain fixed-image execution benchmark; preserve all prior outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DATASETS = ('vdd', 'potsdam', 'udd5', 'oem', 'loveda', 'vaihingen', 'landcoverai', 'flair1')
SETTINGS = {
    'vdd': ('VDD Release Version/VDD', 'grounded_vdd_official20.json'),
    'potsdam': ('Potsdam/preprocessed_RGB', 'grounded_potsdam20.json'),
    'udd5': ('UDD5/extracted/UDD/UDD5', 'hero_udd5_vip20.json'),
    'oem': ('OpenEarthMap_wo_xBD', 'hero_oem_vip20.json'),
    'loveda': ('/data/test/cafe-efa/data/processed/loveda/val', 'gar_llm_raw20_loveda_v1.json'),
    'vaihingen': ('Vaihingen/preprocessed_corrected_20261001', 'grounded_vaihingen20.json'),
    'landcoverai': ('LandCoverAI_v1_target_20260922/preprocessed_official512_gear29', 'gear_landcoverai_v1_20.json'),
    'flair1': ('FLAIR1_target_20260922', 'gear_flair1_main12_20.json')}


def main(args):
    root, tool = Path(args.root), Path(__file__).resolve().parents[1]
    if root.exists():
        raise RuntimeError('Preserve existing RS execution study, no relaunch.')
    root.mkdir(parents=True)
    completed = {}
    for dataset in DATASETS:
        data, vocab = SETTINGS[dataset]
        command = [sys.executable, '-u', str(tool / 'scripts/benchmark_sparse_rs_device.py'),
            '--dataset', dataset, '--dinov3-repo', str(tool / 'dinov3_hub'),
            '--checkpoint-dir', '/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO',
            '--upstream-root', '/data/test/code/ovss_cafe_ped_v2_20260919/third_party/VIP_official_5bd25ee',
            '--data-root', data if data.startswith('/') else '/data/test/datasets/' + data,
            '--vocabulary-config', str(tool / 'configs' / vocab), '--output-dir', str(root / dataset)]
        if args.geometry_execution:
            command.append('--geometry-execution')
        (root / 'suite_status.json').write_text(json.dumps(dict(status='running', active=dataset, completed=list(completed))) + '\n')
        with (root / (dataset + '.log')).open('x') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        result_path = root / dataset / 'results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        if process.returncode or not result or result['status'] != 'complete':
            (root / 'suite_status.json').write_text(json.dumps(dict(status='failed', active=dataset,
                completed=list(completed), returncode=process.returncode, log=str(root / (dataset + '.log')))) + '\n')
            raise RuntimeError('RS execution worker failed; preserve outputs/log: ' + dataset)
        completed[dataset] = dict(sample_key=result['sample_key'], timings=result['timings'])
        print('Completed ' + dataset, flush=True)
    final = dict(status='complete', completed=completed, unique_complete_images=8, serial_single_gpu=True)
    (root / 'suite_results.json').write_text(json.dumps(final, indent=2) + '\n')
    (root / 'suite_status.json').write_text(json.dumps(dict(status='complete', completed=list(completed))) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--geometry-execution', action='store_true')
    main(parser.parse_args())
