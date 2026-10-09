"""Generate publication-facing tables from frozen aggregate evidence only."""
import argparse
import csv
import io
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(root):
    evidence = root / 'research/alias_finalization_20261009'
    summary = read(evidence / 'comparison_summary.json')
    final = read(evidence / 'final_model.json')
    protocol = read(evidence / 'protocol.json')
    if not summary['complete'] or not summary['cost_complete'] or final['ordinary_alias_policy']!='uniform':
        raise ValueError('Verified frozen Uniform release required.')
    lines = ['# 完整结果与 VIP 对比', '',
        '由冻结JSON自动生成，单位为mIoU百分比，Δ为百分点。最终版统一为Uniform；Retained保留旧普通局部软权重。自然普通前景两版原本相同。', '',
        '## 遥感：全量精度', '',
        '| 数据集/协议 | 图像 | VIP声明/蒸馏配置 | VIP原20 | 两种VIP较强值 | 最终版 | 对较强VIP Δ | Retained |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    records = []
    for d in protocol['order']:
        row = read(evidence / d / 'merged.json')
        for p, methods in row['metrics'].items():
            vip = final['vip_comparators'][d][p]
            declared = (vip.get('declared') or {}).get('miou')
            pool = (vip.get('pool20') or {}).get('miou')
            available = [v for v in (declared, pool) if v is not None]
            best = max(available) if available else None
            own = methods['Uniform']['mean_iou_percent']
            retained = methods['Retained']['mean_iou_percent']
            records.append(dict(dataset=d, protocol=p, family=protocol['datasets'][d]['family'],
                images=row['processed_images'], uniform_miou=own, retained_miou=retained,
                vip_declared_miou=declared, vip_pool20_miou=pool, stronger_measured_vip=best,
                delta_vs_declared=None if declared is None else own-declared,
                delta_vs_stronger=None if best is None else own-best,
                local_miou=methods['UniformLocal']['mean_iou_percent'],
                wide_miou=methods['UniformWide']['mean_iou_percent'],
                equal_mean_miou=methods['UniformMean']['mean_iou_percent']))
    number = lambda v: '—' if v is None else f'{v:.4f}'
    for r in records:
        if r['family']!='remote_sensing': continue
        lines.append('| '+r['dataset']+'/'+r['protocol']+f' | {r["images"]} | '+
            ' | '.join(number(r[k]) for k in ('vip_declared_miou','vip_pool20_miou','stronger_measured_vip',
                'uniform_miou','delta_vs_stronger','retained_miou'))+' |')
    lines += ['', f'八遥感等域均值（LoveDA D一次）：最终版 **{final["eight_rs_mean"]["Uniform"]:.4f}**；较强已测VIP **{final["eight_rs_stronger_measured_vip_mean"]:.4f}**；差 **{final["eight_rs_mean"]["Uniform"]-final["eight_rs_stronger_measured_vip_mean"]:+.4f}pp**。', '',
        '## 自然图像：全量精度', '',
        '| 协议 | 图像 | VIP声明配置 | 最终版 | Δpp |', '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        if r['family']=='natural':
            lines.append(f'| {r["dataset"]} | {r["images"]} | '+
                ' | '.join(number(r[k]) for k in ('vip_declared_miou','uniform_miou','delta_vs_declared'))+' |')
    lines += ['', 'PC59没有可用的完整VIP比较，不用pilot填补。VOC20忽略背景评分，不等同VOC21；PC59不等同PC60。', '',
        '## 同观测消融与软权重取舍', '',
        '| 协议 | 局部端点 | 宽视野端点 | 等权logit融合 | 结构耦合Uniform | Retained−Uniform |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append('| '+r['dataset']+'/'+r['protocol']+' | '+
            ' | '.join(number(r[k]) for k in ('local_miou','wide_miou','equal_mean_miou','uniform_miou'))+
            f' | {r["retained_miou"]-r["uniform_miou"]:+.4f} |')
    lines += ['', 'PC60端点仍保留protected residual所需的耦合前景竞争者，是最终读出端点而非完全孤立分支。结构耦合优于简单融合不等于总能胜过最强端点。', '',
        '## 独立进程整图计时与显存', '',
        '| 协议 | 最终版ms | p95 ms | Retained ms | VIP声明配置ms | 最终/VIP | 最终allocated MiB | VIP allocated MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    speed_rows=[]
    for d in protocol['order']:
        modes=summary['cost_summary'][d];own=modes['Uniform'];vip=modes['VIPOfficial']
        values=(own['median_ms'],own['p95_ms'],modes['Retained']['median_ms'],vip['median_ms'],
            own['median_ms']/vip['median_ms'],own['peak_allocated_mib'],vip['peak_allocated_mib'])
        lines.append('| '+d+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        speed_rows.append(dict(dataset=d,uniform_median_ms=values[0],uniform_p95_ms=values[1],
            retained_median_ms=values[2],vip_median_ms=values[3],uniform_vip_ratio=values[4],
            uniform_allocated_mib=values[5],vip_allocated_mib=values[6],
            uniform_reserved_mib=own['peak_reserved_mib'],vip_reserved_mib=vip['peak_reserved_mib']))
    lines += ['', 'A800；每协议7个固定完整图，每图预热后7次CUDA同步，单候选独立进程。包含缩放、全部视觉编码、语义头、词聚合、写回、拼接、原尺寸恢复和CPU预测；不含解码/初始化/文本编码。不是全量平均延迟。自然VIP用short336/cap2048；本模型short336/cap672；旧long448自然VIP计时排除。五个非官方遥感域的VIP计时为外部20词适配，并非蒸馏后短词计时。', '',
        '## 比较口径与结论', '',
        '- VIP均为本地实现并明确修复全mask attention行后的结果，不是认证的论文数字。VDD/Potsdam/Vaihingen使用官方短词；其余遥感域同时展示原20与论文规则蒸馏。',
        '- “较强VIP”只是两种已测设置的最大值，不代表所有可能VIP配置；不会据此切换我们的方法。',
        '- 完整模型比较的词库、模板、视野与额外残余概念预算不同，不能把整体差值单独归因Geometry。机制归因看同观测消融。',
        '- 遥感八域均超过较强已测VIP；自然尚未全面胜出。保留真实负结果。',
        '- 45,200是协议图像评测次数；VOC20/21和PC59/60共享来源。LandCover.ai替代未获得可评分标签的iSAID，Cityscapes未纳入。',
        '- 所有域参与过开发；配对统计条件于这些数据，不消除方法/词表选择偏倚。', '',
        '来源：[冻结模型](../research/alias_finalization_20261009/final_model.json)、[完整原报告](../research/CVPR_FINAL_MODEL_20261009.md)、[配对统计](../research/alias_finalization_20261009/statistics_full/PAIRED_UNCERTAINTY.md)。', '']
    docs=root/'docs';docs.mkdir(exist_ok=True)
    (docs/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (docs/'results.json').write_text(json.dumps(dict(accuracy=records,cost=speed_rows),indent=2)+'\n',encoding='utf-8')
    for name,rows in (('accuracy.csv',records),('speed_memory.csv',speed_rows)):
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
        (docs/name).write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    print(json.dumps(dict(table=str(docs/'RESULTS.md'),accuracy_protocols=len(records),cost_protocols=len(speed_rows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    export(parser.parse_args().root)
