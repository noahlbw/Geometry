"""Summarize the fixed-input Geov2 target/gate ablation."""
import csv
import json
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'research' / 'geov2_gate_ablation_20261014'
DATASETS = ('vdd', 'potsdam', 'ade150', 'voc20')
METHODS = (
    'Geov2_VIPWide', 'Raw32', 'QuerySlot_Raw', 'DonorBefore',
    'Geov2_Full', 'Geov2_Target', 'Geov2_FixedG2', 'Geov2_AdaptiveGate',
    'Geov2_NoAlias_Canonical', 'Geov2_EqualFusion', 'Geov2_ShuffledQuery',
)
EXTERNAL_VIP = {'vdd': 52.5765, 'potsdam': 44.8995,
                'ade150': 29.4630, 'voc20': 92.5051}


def miou(cm):
    cm = np.asarray(cm, dtype=np.float64)
    den = cm.sum(1) + cm.sum(0) - np.diag(cm)
    valid = den > 0
    return float(np.mean(np.divide(np.diag(cm), den,
                                   out=np.zeros_like(den), where=valid)[valid]) * 100.)


def bootstrap(arrays, reps=3000, seed=20261014):
    rng = np.random.default_rng(seed)
    deltas = {m: [] for m in METHODS if m != 'Geov2_AdaptiveGate'}
    for _ in range(reps):
        sampled = {d: rng.integers(0, arrays[d][METHODS[0]].shape[0],
                                   arrays[d][METHODS[0]].shape[0]) for d in DATASETS}
        scores = {m: float(np.mean([miou(arrays[d][m][sampled[d]].sum(0))
                                     for d in DATASETS])) for m in METHODS}
        for m in deltas:
            deltas[m].append(scores[m] - scores['Geov2_AdaptiveGate'])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for m, v in deltas.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    records = {d: json.loads((root / d / 'results.json').read_text()) for d in DATASETS}
    arrays = {d: {m: np.load(root / d / 'per_image_confusions.npz')[m] for m in METHODS}
              for d in DATASETS}
    rows = []
    for d in DATASETS:
        for m in METHODS:
            score = records[d]['metrics'][m]['mean_iou_percent']
            rows.append({'dataset': d, 'method': m, 'images': records[d]['processed_images'],
                         'miou_percent': score})
    macro = {m: float(np.mean([next(r['miou_percent'] for r in rows
                                if r['dataset'] == d and r['method'] == m)
                               for d in DATASETS])) for m in METHODS}
    ci = bootstrap(arrays)
    summary = {
        'protocol': 'Four fixed 16-image development sets; shared backbone, alias bank, 24 slots, beta=.5, temperature=.1.',
        'methods': list(METHODS), 'rows': rows, 'macro_miou_percent': macro,
        'delta_vs_adaptive_gate_pp': {m: macro[m] - macro['Geov2_AdaptiveGate'] for m in METHODS},
        'paired_bootstrap_delta_ci95': ci,
        'external_vip_reference': EXTERNAL_VIP,
        'interpretation': {
            'target': 'Z* is the exact structure-anchored quadratic solution before the second-stage step.',
            'fixed_g2': 'Z=L+2(Z*-L), the declared over-relaxation control.',
            'adaptive': 'Z=L+g(x)(Z*-L), where g(x) uses only local/wide margins and structural support.',
            'equal_fusion': '0.5(L+W), a matched fusion control that bypasses the writer.',
            'final': 'The released Geov2 model uses adaptive structure writeback; fixed and equal fusion remain controls.'
        }
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / 'gate_ablation_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (root / 'gate_ablation_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    labels = {
        'Geov2_VIPWide': 'VIP-wide endpoint', 'Raw32': 'Raw32',
        'QuerySlot_Raw': 'QuerySlot', 'DonorBefore': 'DonorBefore',
        'Geov2_Full': 'Fixed writer g=1', 'Geov2_Target': 'Z* target',
        'Geov2_FixedG2': 'Fixed g=2', 'Geov2_AdaptiveGate': 'Adaptive gate',
        'Geov2_NoAlias_Canonical': 'Canonical only', 'Geov2_EqualFusion': 'Equal fusion',
        'Geov2_ShuffledQuery': 'Shuffled query'}
    lines = ['# Geov2 structure target and adaptive gate ablation (2026-10-14)', '',
             'All rows use the same 16-image development screen, backbone, aliases, background rule and 24-slot observer.', '',
             '| Dataset | ' + ' | '.join(labels[m] for m in METHODS) + ' |',
             '|---|' + '|'.join('---:' for _ in METHODS) + '|']
    for d in DATASETS:
        lines.append('| ' + d + ' | ' + ' | '.join(f"{next(r['miou_percent'] for r in rows if r['dataset']==d and r['method']==m):.4f}" for m in METHODS) + ' |')
    lines += ['', '| Equal-domain macro | ' + ' | '.join(f'{macro[m]:.4f}' for m in METHODS) + ' |', '',
              '## What is selected', '',
              'The final writer is `Z = L + g(x)(Z* - L)`. `Z*` is computed by the same structure-anchored solve used by the fixed writer; the gate is bounded to `[0.5, 2.0]` and uses no labels. `Geov2_FixedG2`, `Geov2_Target`, and `Geov2_EqualFusion` are controls.', '',
              '## External VIP reference', '', '| Dataset | Full-validation VIP mIoU |', '|---|---:|']
    lines += [f'| {d} | {EXTERNAL_VIP[d]:.4f} |' for d in DATASETS]
    lines += ['', 'This screen is a mechanism/ablation screen. It does not replace the frozen full-protocol results.']
    (root / 'GEOV2_GATE_ABLATION_20261014.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'macro_miou_percent': macro, 'output': str(root)}, indent=2))


if __name__ == '__main__':
    main()
