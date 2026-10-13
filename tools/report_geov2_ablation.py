"""Summarize the fixed-input Geov2 component ablation and draw its figure."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'research' / 'geov2_ablation_20261014'
DATASETS = ('vdd', 'potsdam', 'ade150', 'voc20')
METHODS = ('Geov2_VIPWide', 'Raw32', 'QuerySlot_Raw', 'DonorBefore',
           'Geov2_Full', 'Geov2_NoAlias_Canonical', 'Geov2_EqualFusion',
           'Geov2_ShuffledQuery')
EXTERNAL_VIP = {'vdd': 52.5765, 'potsdam': 44.8995,
                'ade150': 29.4630, 'voc20': 92.5051}


def miou(cm):
    cm = np.asarray(cm, dtype=np.float64)
    den = cm.sum(1) + cm.sum(0) - np.diag(cm)
    valid = den > 0
    return float(np.mean(np.divide(np.diag(cm), den, out=np.zeros_like(den), where=valid)[valid]) * 100.)


def bootstrap(arrays, method_names, reps=3000, seed=20261014):
    rng = np.random.default_rng(seed)
    values = {m: [] for m in method_names}
    deltas = {m: [] for m in method_names if m != 'Geov2_Full'}
    for _ in range(reps):
        sample = {d: rng.integers(0, arrays[d][method_names[0]].shape[0],
                                   arrays[d][method_names[0]].shape[0]) for d in DATASETS}
        scores = {m: float(np.mean([miou(arrays[d][m][sample[d]].sum(0)) for d in DATASETS]))
                  for m in method_names}
        for m, score in scores.items():
            values[m].append(score)
        for m in deltas:
            deltas[m].append(scores[m] - scores['Geov2_Full'])
    ci = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for m, v in values.items()}
    delta_ci = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                for m, v in deltas.items()}
    return ci, delta_ci


def main():
    results = {d: json.loads((ROOT / d / 'results.json').read_text()) for d in DATASETS}
    arrays = {d: {m: np.load(ROOT / d / 'per_image_confusions.npz')[m]
                  for m in METHODS} for d in DATASETS}
    rows = []
    for d in DATASETS:
        metric = results[d]['metrics']
        full = metric['Geov2_Full']['mean_iou_percent']
        for m in METHODS:
            rows.append(dict(dataset=d, method=m, images=results[d]['processed_images'],
                             miou_percent=metric[m]['mean_iou_percent'],
                             delta_vs_full_pp=metric[m]['mean_iou_percent'] - full,
                             delta_vs_vipwide_pp=metric[m]['mean_iou_percent'] - metric['Geov2_VIPWide']['mean_iou_percent']))
    macro = {}
    for m in METHODS:
        vals = [next(r['miou_percent'] for r in rows if r['dataset'] == d and r['method'] == m)
                for d in DATASETS]
        macro[m] = dict(miou_percent=float(np.mean(vals)), values=vals,
                        delta_vs_full_pp=float(np.mean(vals) - np.mean([next(r['miou_percent'] for r in rows if r['dataset'] == d and r['method'] == 'Geov2_Full') for d in DATASETS])))
    ci, delta_ci = bootstrap(arrays, METHODS)
    summary = dict(protocol='Four fixed development images sets; 16 images per dataset; 24 slots; beta=0.5; temperature=0.1.',
                    datasets=list(DATASETS), methods=list(METHODS), rows=rows, macro=macro,
                    bootstrap=dict(replicates=3000, seed=20261014, ci95=ci,
                                  delta_vs_full_ci95=delta_ci),
                    external_vip_reference=dict(source='research/TRAINING_FREE_BASELINES_FULL_20261011.md',
                        label='Strong verified local VIP / external VIP reference (full validation)', values=EXTERNAL_VIP,
                        note='This is a separate full-validation baseline. It is not the Geov2_VIPWide row and was not used to select the ablation.'),
                    interpretation=dict(
                        alias='Geov2_NoAlias_Canonical keeps one canonical phrase per class; the full row uses the frozen retained alias bank.',
                        query_slot='QuerySlot_Raw routes a local Geometry tile into the shared wide slot bank.',
                        donor_before='DonorBefore changes only the local response order for the original-strength path.',
                        raw32='Raw32 is the un-routed 32-slot whole-image memory control.',
                        equal_fusion='Geov2_EqualFusion is literal 0.5*(L+W) logit fusion and bypasses H.',
                        shuffle='Geov2_ShuffledQuery rolls each local query token sequence before routing; the slot bank and token multiset are unchanged.',
                        vip='Geov2_VIPWide is an internal endpoint. VIP is an external baseline, not a component of Geov2.'))
    (ROOT / 'ablation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (ROOT / 'ablation_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    labels = {'Geov2_VIPWide': 'Geov2 VIP-wide', 'Raw32': 'Raw32', 'QuerySlot_Raw': 'QuerySlot raw',
              'DonorBefore': 'DonorBefore', 'Geov2_Full': 'Geov2 full',
              'Geov2_NoAlias_Canonical': 'No alias', 'Geov2_EqualFusion': 'Equal fusion',
              'Geov2_ShuffledQuery': 'Shuffled query'}
    lines = ['# Geov2 component ablation (2026-10-14)', '',
             '四个数据集各取固定 16 张开发图，所有行共享 backbone、输入、文本 bank、背景规则和 24 槽配置。这里的 `Geov2_VIPWide` 是 Geov2 内部的 VIP 宽视野端点；官方 VIP 完整复现属于外部基线，单列在文末。', '',
             '## Same-input results', '',
             '| Dataset | ' + ' | '.join(labels[m] for m in METHODS) + ' |',
             '|---|' + '|'.join('---:' for _ in METHODS) + '|']
    for d in DATASETS:
        vals = [next(r['miou_percent'] for r in rows if r['dataset'] == d and r['method'] == m) for m in METHODS]
        lines.append('| ' + d + ' | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    lines += ['', '| Equal-domain macro | ' + ' | '.join(f"{macro[m]['miou_percent']:.4f}" for m in METHODS) + ' |',
              '', '## Component deltas', '',
              '| Variant | Δ vs Geov2 full (pp) | 95% paired bootstrap CI |', '|---|---:|---:|']
    for m in METHODS:
        lines.append(f"| {labels[m]} | {macro[m]['delta_vs_full_pp']:+.4f} | [{ci[m][0]:.4f}, {ci[m][1]:.4f}] |" if m == 'Geov2_Full' else
                     f"| {labels[m]} | {macro[m]['delta_vs_full_pp']:+.4f} | [{delta_ci[m][0]:+.4f}, {delta_ci[m][1]:+.4f}] |")
    lines += ['', '解释：', '',
              '- Alias：`No alias` 只保留每个类别的 canonical phrase，因此它测的是 alias bank 的贡献，不是把类别从文本输入中删除。',
              '- QuerySlot：`QuerySlot_Raw` 相对 `Raw32` 只增加局部 query 到共享 slot 的路由；`Geov2 full` 再加 DonorBefore。',
              '- DonorBefore：`DonorBefore` 与 `Geov2_VIPWide` 共享 VIP 宽视野，隔离局部分支变化；数值 strength 的配置仍使用原 Geometry 路径。',
              '- Equal fusion：逐点使用 `0.5(L+W)`，没有结构算子 H；因此它是融合控制，不是 Geov2 writer。',
              '- Shuffle：固定全图 slot bank，只滚动局部 query 的空间顺序；性能下降说明 QuerySlot 的提升依赖正确的局部位置对应。',
              '- VIP：官方 VIP 不是 Geov2 的一个组件。外部全验证参考来自 `research/TRAINING_FREE_BASELINES_FULL_20261011.md`，其数值不能与本 16 图子集直接混为同一列。', '',
              '## External VIP reference (full validation)', '',
              '| Dataset | External VIP reference mIoU |', '|---|---:|']
    for d in DATASETS: lines.append(f'| {d} | {EXTERNAL_VIP[d]:.4f} |')
    lines += ['', '本表用于部件归因，属于开发屏幕；不能替代 2026-10-09 冻结版全量主结果。原始逐图混淆矩阵和 bootstrap 数据见同目录。']
    (ROOT / 'GEOV2_ABLATION_20261014.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    try:
        import matplotlib.pyplot as plt
        x = np.arange(len(METHODS)); y = [macro[m]['miou_percent'] for m in METHODS]
        fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
        bars = ax.bar(x, y, color=['#7f8c8d', '#95a5a6', '#5dade2', '#48c9b0', '#2e86c1', '#c0392b', '#f5b041', '#884ea0'])
        ax.set_ylabel('mIoU (%)'); ax.set_xticks(x, [labels[m] for m in METHODS], rotation=25, ha='right')
        ax.set_title('Geov2 component ablation — equal-domain macro mIoU')
        ax.grid(axis='y', alpha=.25)
        for bar, value in zip(bars, y): ax.text(bar.get_x()+bar.get_width()/2, value+.25, f'{value:.2f}', ha='center', fontsize=8)
        fig.tight_layout(); fig.savefig(ROOT / 'component_ablation.png'); plt.close(fig)
    except Exception as e:
        (ROOT / 'plot_error.txt').write_text(str(e) + '\n')
    print(json.dumps({'macro': {m: round(macro[m]['miou_percent'], 4) for m in METHODS}, 'output': str(ROOT)}))


if __name__ == '__main__':
    main()
