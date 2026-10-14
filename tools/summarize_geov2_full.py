"""Write a readable report from already audited four-dataset gate results."""
import json
from pathlib import Path

from report_geov2_g0 import table


def main():
    root=Path('research/geov2_full_ablation_20261014')
    primary=json.loads((root/'audited_summary.json').read_text())
    supplement=json.loads((root/'supplement/audited_summary.json').read_text())
    cost=json.loads((root/'queryslot_writer_cost.json').read_text())
    datasets=list(primary['datasets'])
    def metrics(source,methods):
        return [[m]+[f"{source['datasets'][ds]['metrics'][m]['mean_iou_percent']:.4f}" for ds in datasets]+
            [f"{source['equal_dataset_macro_miou'][m]:.4f}"] for m in methods]
    fixed=primary['equal_dataset_macro_miou']['Geov2_FixedTarget']
    adaptive=primary['equal_dataset_macro_miou']['Geov2_AdaptiveGate']
    equal=primary['equal_dataset_macro_miou']['EqualFusion']
    lines=['# Geov2：四域全量 gate 候选验证（历史配置）','',
        f"固定结构写回平均 {fixed:.4f}%，等权融合 {equal:.4f}%，差 {fixed-equal:+.4f}pp。自适应为 {adaptive:.4f}%，相对固定仅 {adaptive-fixed:+.4f}pp；margin-only 的平均甚至略高。因此保留固定写回为主，自适应不称为普遍最优。",'',
        '**此实验与最新十域 g0 扫描使用不同的历史配置，必须分开引用。** Potsdam strength=3、VOC20 strength=1，故这两域的 DonorBefore 不执行；VOC20 文本 bank 为 official_segmentation。十域扫描统一 original，并沿用10/13各自最佳 bank，不能拼接两轮数值为一张完整模型表。','',
        '## 主消融：全量数据集 mIoU (%)','',
        table(['方法',*datasets,'等域平均'],metrics(primary,primary['datasets']['vdd']['methods'])),'',
        '## 自适应相对固定写回的配对区间','',
        table(['数据集','图像数','差值 pp','95% 配对区间 pp'],
            [[ds,primary['datasets'][ds]['global_sample_count'],
              f"{primary['datasets'][ds]['paired_intervals']['Geov2_FixedTarget']['difference_pp']:+.6f}",
              str([round(x,6) for x in primary['datasets'][ds]['paired_intervals']['Geov2_FixedTarget']['paired_dataset_miou_ci95_pp']])]
             for ds in datasets]),'',
        '每个区间为固定历史配置下的3000次图像配对重采样，重新累计 TP/union 计算数据集 mIoU，非逐图 IoU 均值。Potsdam、VOC20 的区间包含0；不宣称四域均有可靠提升。','',
        '## 匹配自适应写回的补充消融','',
        table(['方法',*datasets,'等域平均'],metrics(supplement,supplement['datasets']['vdd']['methods'])),'',
        'DoubleBaseStep 将旧 g0 翻倍，不是统一设 g0=2。QuadraticOptimum 才是所有域统一 g0=1。NoDonorAdaptive 对数值 strength 的两域不改实现，故不能据其接近0的差值否定 DonorBefore；NoRoutingAdaptive 取消 query 路由，局部路径与写回保持匹配。补充参考独立重读图像，CUDA聚类归约存在微小差异，不将两次参考的差值当作模块增益。','',
        '## 实际 QuerySlot 路径成本','',
        'A800、batch=1，4域各8张固定等距整图，每图/方法2次预热和5次同步重复。从CPU RGB tensor到CPU label map，含缩放、两个视觉角色、结构求解、文本读出、拼接和原图恢复；不含解码、初始化、离线文本编码和标签读取。表中为重复中位数和峰值 CUDA allocated。完整重复样本保留于 queryslot_writer_cost.json。','',
        table(['数据集','fixed ms/MiB','adaptive ms/MiB','equal_mean ms/MiB','adaptive_efficient ms/MiB'],
            [[r['dataset']]+[f"{r['methods'][m]['median_ms']:.3f} / {r['methods'][m]['peak_allocated_mib']:.2f}" for m in ('fixed','adaptive','equal_mean','adaptive_efficient')] for r in cost['records']]),'',
        'adaptive_efficient 与参考实现的抽样标签图存在非零像素差异，不能宣称逐位等价，也不能直接给它赋予参考自适应分支的全量 mIoU。部分重复有延迟长尾，应保留完整分布而非用均值与其他轮次中位数比较。旧 native-wide 候选的163ms/1632MiB不属于本 QuerySlot 模型。','',
        '## 写回公式及归属','',
        '本实验使用历史 H=(I+A^T A)^(-1)A^T A 的 FP64 求解，Z*=L+H(W−L)。固定输出 T=L+g0(Z*−L)；自适应输出 L+a(x)(T−L)，有效步长为g0·a(x)。只有g0=1时T才是原二次目标的最优解。gate 的 margin 是置信代理，不能等同于正确概率。','',
        'audited_summary.json 和 supplement/audited_summary.json 记录完整覆盖、汇总混淆矩阵及配对区间。原始逐图数据和实际执行代码在同目录分片及 implementation/ 中；可由 tools/report_geov2_full.py 重新审计。','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(root/'REPORT.md')


if __name__=='__main__':
    main()
