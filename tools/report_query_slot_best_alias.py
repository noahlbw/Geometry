"""Report completed full mIoU and isolated costs for the requested suite."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT/'research/query_slot_best_alias_20261013'
OLD = ROOT/'research/efficient_geometry_20261012'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol = read(SUITE/'protocol.json')
    completion = read(SUITE/'suite_results.json')
    if completion['status']!='complete':
        raise RuntimeError('Report only a completed suite.')
    rows = []
    aliases = []
    for d,entry in protocol['datasets'].items():
        full = read(SUITE/'merged'/f'{d}.json')
        cost = read(SUITE/('cost_recheck/cost' if d=='flair1' else 'cost')/f'{d}.json')
        old_cost = read(OLD/'cost'/f'{d}.json')['costs']['efficient']
        if not all(full[k] for k in ('coverage_verified','scored_target_counts_match','per_image_sums_verified','both_modules_active')):
            raise RuntimeError('Incomplete coverage/activation: '+d)
        config = read(SUITE/entry['config'])
        vocab = read(SUITE/'vocabularies'/f'{d}.json')
        counts = [len(c['synonyms']) for b in vocab['banks'].values() for c in b['classes']]
        aliases.append(dict(dataset=d,bank=config['profile']['bank'],alias_total=sum(counts),
            alias_min=min(counts),alias_max=max(counts),old_strength=entry['original_profile']['strength'],
            new_strength='original',residual_concepts=len((vocab.get('residual') or {}).get('names',[])),
            vocabulary=entry['alias_source']['vocabulary']))
        for p,metric in full['metrics'].items():
            current = cost['costs'][p]
            old_rows = [r for r in old_cost['images'] if r['protocol']==p]
            if not old_rows:
                raise RuntimeError('Historical cost protocol missing: '+d+'/'+p)
            import numpy as np
            historical_ms = float(np.median([t for row in old_rows for t in row['milliseconds']]))
            historical_memory = max(r['peak_allocated_mib'] for r in old_rows)
            baseline = entry['incumbent_metrics'][p]['mean_iou_percent']
            rows.append(dict(dataset=d,protocol=p,images=full['processed_images'],
                baseline_miou=baseline,query_slot_donor_before_miou=metric['mean_iou_percent'],
                delta_miou_pp=round(metric['mean_iou_percent']-baseline,4),
                median_ms=current['median_ms'],peak_allocated_mib=current['peak_allocated_mib'],
                peak_reserved_mib=current['peak_reserved_mib'],baseline_median_ms=historical_ms,
                baseline_peak_allocated_mib=historical_memory,
                latency_change_percent=(current['median_ms']/historical_ms-1)*100,
                memory_change_mib=current['peak_allocated_mib']-historical_memory,
                alias_bank=config['profile']['bank'],old_strength=entry['original_profile']['strength'],
                new_strength='original',both_modules_active=True))
    if len(rows)!=16:
        raise RuntimeError('Expected all sixteen protocols.')
    for name,data in [('RESULTS.csv',rows),('ALIAS_MANIFEST.csv',aliases)]:
        with (SUITE/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (SUITE/'RESULTS.json').write_text(json.dumps(dict(status='complete',rows=rows,aliases=aliases,
        source_images=completion['source_images'],protocol_image_evaluations=sum(r['images'] for r in rows)),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# QuerySlot＋DonorBefore：已有最佳 alias 的全量结果', '',
        '2026-10-13。全部 15 数据集、16 协议完成全量 mIoU 与独立成本测量。'
        '词表采用之前已完成带标签开发搜索的 retained 最佳版本；本轮没有再按新全量成绩挑选词表。', '',
        '为使 DonorBefore 在所有数据集实际执行，局部 strength 统一为 original。'
        '其中原先使用数值 strength 的协议同时发生局部路径变化；下表相对旧版的差值是整体配置差值，不能仅归因于宽视野替换。'
        'QuerySlot 固定 24 槽、beta=0.5、温度=0.1；PC60 保留 401 个残余概念。其余 g、词聚合、背景校准和视野策略沿用各自旧最佳配置。', '',
        '| 协议 | 图像 | 旧 Efficient mIoU % | QuerySlot＋DonorBefore % | Δpp | 耗时 ms | 峰值 allocated MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        label = r['dataset']+('/'+r['protocol'] if r['dataset']=='loveda' else '')
        lines.append(f"| {label} | {r['images']} | {r['baseline_miou']:.4f} | {r['query_slot_donor_before_miou']:.4f} | {r['delta_miou_pp']:+.4f} | {r['median_ms']:.2f} | {r['peak_allocated_mib']:.2f} |")
    lines += ['', '## 成本边界', '',
        'A800，batch=1。每个数据集使用旧成本清单相同的 7 张固定完整图；每图 2 次预热、7 次同步重复。'
        '报告所用成本记录在本轮全量任务不运行时逐项独立测量。耗时为这些重复的中位数；显存为 CUDA max_memory_allocated 的峰值。'
        '包含 CPU RGB 缩放、全部编码、读出、结构求解、原尺寸恢复和 CPU uint8 输出；排除图像解码、初始化、离线文本编码和标签读取。'
        '文本塔释放，两个视觉模型保留原数值角色并常驻。reserved 原始值另存在 CSV/JSON 中。', '',
        '与旧 Efficient 的耗时对照属于跨轮测量；运行状态可能影响绝对速度。LoveDA P/D 是独立读出，各自预测和计时。'
        '完整覆盖为 45,200 次数据集图像评测；含附加 LoveDA P 后为 46,869 次协议图像评测，来源之间存在共享图像。', '',
        'FLAIR 的初次成本与首批全量任务有约 30 秒重叠，原记录保留在 cost/flair1.json。'
        '本报告仅使用全量结束后 cost_recheck/cost/flair1.json 的独立重测。', '',
        '## alias 与局部路径', '',
        '| 数据集 | alias bank | alias 总数 | 每类范围 | 旧 strength | 新 strength | 残余概念 |',
        '|---|---|---:|---:|---|---|---:|']
    for a in aliases:
        lines.append(f"| {a['dataset']} | {a['bank']} | {a['alias_total']} | {a['alias_min']}–{a['alias_max']} | {a['old_strength']} | original | {a['residual_concepts']} |")
    lines += ['', '## 数据与实现', '',
        '所有合并记录核验唯一完整图像覆盖、各分片配置一致、标注像素计数与既有协议一致及逐图统计之和。'
        '推理本身不读取标签；标签仅由外层评测脚本计算混淆矩阵。', '',
        '- 汇总：RESULTS.csv、RESULTS.json；词表：ALIAS_MANIFEST.csv、vocabularies/。',
        '- 冻结配置：protocol.json、configs/；独立成本：cost/；全量指标：merged/。',
        '- 逐图 TP/target/predicted 充分统计：full/*.npz。',
        '- A800 独立源码与结果：results/query_slot_best_alias_20261013/。',
        '- 执行入口：DINOtool/scripts/eval_query_slot_best_alias.py、run_query_slot_best_alias.py。',
        '- 本实验独立保存；2026-10-09 论文主表继续使用其原冻结结果。']
    (SUITE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='complete',rows=len(rows),improved=sum(r['delta_miou_pp']>0 for r in rows),
        lower=sum(r['delta_miou_pp']<0 for r in rows),report=str(SUITE/'REPORT.md')),ensure_ascii=False))


if __name__=='__main__':
    main()
