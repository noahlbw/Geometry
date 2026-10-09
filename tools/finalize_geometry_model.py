"""Freeze one global alias policy only after complete accuracy/cost evidence.

This is a local reporting/release step. It never launches inference, edits
model rules, or chooses historical per-domain winners.
"""
import argparse
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / 'research/alias_finalization_20261009'
REPORT = WORKSPACE / 'research/CVPR_FINAL_MODEL_20261009.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def readiness():
    summary = read(ROOT / 'comparison_summary.json')
    stats_path = ROOT / 'statistics_full/paired_statistics.json'
    ready = (summary['complete'] and summary['cost_complete']
             and stats_path.exists() and summary['decision']['status'] == 'frozen')
    return ready, summary, stats_path


def comparator(path, dataset, protocol, method, current):
    """Read only completed local comparators; absent full protocols stay absent."""
    if not path.exists():
        return None
    old = read(path)
    keys = old.get('sample_keys', old.get('signature', {}).get('sample_keys'))
    if (old.get('status') != 'complete' or not old.get('coverage_verified')
            or old['processed_images'] != current['processed_images']
            or old['total_images'] != current['total_images']
            or keys is None or len(set(keys)) != len(keys)
            or set(keys) != set(current['signature']['sample_keys'])):
        raise ValueError('Comparator coverage differs: ' + str(path))
    group = old['metrics'][protocol]
    metric = group[method] if method in group else group
    signature = old.get('source_signature', old.get('signature', {}))
    checkpoints = signature.get('checkpoint_manifest', signature.get('checkpoints'))
    if checkpoints != current['signature']['checkpoints']:
        raise ValueError('Comparator checkpoint identity differs: ' + str(path))
    # The old ADE metadata contains "bed " at scored ID7; strip surrounding
    # display whitespace only. Semantic names and ordinal IDs are not remapped.
    if [c['name'].strip() for c in metric['per_class']] != [n.strip() for n in current['signature']['classes'][protocol]]:
        raise ValueError('Comparator scored class order differs: ' + str(path))
    targets = np.asarray(current['metrics'][protocol]['Retained']['confusion_matrix']).sum(1)
    if not np.array_equal(np.asarray(metric['confusion_matrix']).sum(1), targets):
        raise ValueError('Comparator scored targets differ: ' + str(path))
    return dict(miou=metric['mean_iou_percent'], source=str(path.relative_to(WORKSPACE)),
                method=method if method in group else 'direct_protocol_metric',
                checkpoint_identity_verified=True, scored_class_order_verified=True,
                scored_name_normalization='Surrounding display whitespace only; no semantic or ID remapping.',
                extra_metrics={k:metric[k] for k in ('foreground_mean_iou_percent','non_residual_mean_iou_percent') if k in metric})


def vip_results(dataset, protocol, entry, current):
    research = WORKSPACE / 'research'
    if entry['family'] == 'remote_sensing':
        if dataset == 'vaihingen':
            pool_path = research / 'shared_rival_soft20_full_20261005/vip_all20_vaihingen.json'
        else:
            parent = ('vip_official20_distillation_full_20261003' if dataset in ('vdd', 'potsdam')
                      else 'vip_paper_distillation_full_20261003')
            pool_path = research / parent / dataset / 'merged.json'
        declared_path = research / 'vip_paper_distillation_full_20261003' / dataset / 'merged.json'
        return dict(pool20=comparator(pool_path, dataset, protocol, 'VIP_All20', current),
                    declared=comparator(declared_path, dataset, protocol, 'VIP_Distilled', current))
    path = (research / 'context60_official_finite_20261007/merged.json' if dataset == 'context60'
            else research / 'natural_text_adaptation_20261003' / dataset / 'full_merged.json')
    return dict(declared=comparator(path, dataset, protocol, 'VIP_Official_Finite', current))


def finalize():
    ready, summary, stats_path = readiness()
    if not ready:
        raise RuntimeError('Complete all15 accuracy protocols, corrected cost modes and full paired statistics first.')
    destination = ROOT / 'final_model.json'
    errors_path = ROOT / 'class_competition.json'
    if destination.exists() or REPORT.exists() or errors_path.exists():
        raise RuntimeError('Existing final release; preserve it.')
    protocol = read(ROOT / 'protocol.json')
    stats = read(stats_path)
    if set(stats['domains']) != set(protocol['order']) or stats.get('inference_rerun') is not False:
        raise ValueError('Full frozen paired analysis required.')
    rows, comparisons, count = {}, {}, 0
    for d in protocol['order']:
        entry = protocol['datasets'][d]
        row = read(ROOT / d / 'merged.json')
        if (row['status'] != 'complete' or not row['coverage_verified']
                or not row.get('paired_scored_targets_equal') or not row.get('scalar_agreement_verified')
                or row['processed_images'] != entry['total_images']
                or row['total_images'] != entry['total_images']
                or row['signature']['implementation'] != protocol['implementation']):
            raise ValueError('Unverified final protocol: ' + d)
        keys = row['signature']['sample_keys']
        if len(set(keys)) != len(keys) or len(keys) != entry['total_images']:
            raise ValueError('Duplicate/incomplete final coverage.')
        rows[d] = row
        comparisons[d] = {p: vip_results(d, p, entry, row) for p in row['metrics']}
        count += len(keys)
    if count != protocol['expected_total_images']:
        raise ValueError('Full manifest count differs.')
    gate = protocol['decision']
    value = lambda d: summary['outcomes'][d]['D' if d == 'loveda' else d]['delta_pp']
    primary = [value(d) for d in gate['primary_panel']]
    passes = (np.mean(primary) >= gate['mean_delta_floor_pp']
              and min(primary) >= gate['worst_delta_floor_pp']
              and min(value(d) for d in protocol['order']) >= gate['transfer_worst_delta_floor_pp'])
    policy = 'uniform' if passes else 'retained'
    if summary['decision']['ordinary_alias_policy'] != policy:
        raise ValueError('Collected decision does not match frozen gate.')
    method = policy.capitalize()
    manifest = dict(implementation=protocol['implementation'], ordinary_alias_policy=policy,
        factory='dinotool.taxonomy_inference.TaxonomyInference.for_finalization',
        frozen_input_manifest='protocol.json', datasets=protocol['order'],
        factory_arguments='Explicit family/background/residual_features from frozen protocol; local_background=retained. No new per-domain arm selection.',
        evidence='comparison_summary.json and statistics_full/paired_statistics.json',
        budget=protocol['budget'], expected_protocol_images=count,
        semantic_screening_contribution=False, default_incumbent_api_changed=False,
        conditional_alias='LME response responsibility and inherited image salience remain. These do not certify semantic reliability.',
        ordinary_weighting='Disabled globally for simplification.' if passes else 'Legacy empirical coverage-soft weighting retained globally; not a new reliability estimator.',
        provenance=protocol['source_label_development'], vip_comparators=comparisons,
        limitations=['No untouched validation from rerunning developed domains.',
            'Nearest bounded SCLIP control belongs to fixed patch-only2, not this task route.',
            'VIP observer and salience are attributed; quadratic solve is classical.',
            'No universal positive transfer or all-domain SOTA claim.'])
    manifest['vocabulary_budget'] = {}
    for d, entry in protocol['datasets'].items():
        bank_names = ('semantic_segmentation',) if entry['family']=='natural' else ('original_imagenet','focused20')
        manifest['vocabulary_budget'][d] = {name:[len(c['synonyms']) for c in entry['banks'][name]['classes']]
            for name in bank_names}
    manifest['supplemental_reference_recovery'] = {
        d:row['reference_recovery'] for d,row in rows.items() if 'reference_recovery' in row}
    errors = {}
    for d, row in rows.items():
        errors[d] = {}
        for p, groups in row['metrics'].items():
            primary_cm = np.asarray(groups[method]['confusion_matrix'], np.int64)
            targets = primary_cm.sum(1)
            predicted = primary_cm.sum(0)
            tp = primary_cm.diagonal()
            names = row['signature']['classes'][p]
            classes = []
            for i, name in enumerate(names):
                changes = {}
                for control in ('UniformLocal', 'UniformWide', 'UniformMean'):
                    cm = np.asarray(groups[control]['confusion_matrix'], np.int64)
                    delta_tp = int(tp[i] - cm[i, i])
                    changes[control] = dict(delta_tp=delta_tp, delta_fn=-delta_tp,
                        delta_fp=int(predicted[i] - cm[:, i].sum()) - delta_tp,
                        delta_iou_pp=(None if groups[method]['per_class'][i]['iou_percent'] is None
                            or groups[control]['per_class'][i]['iou_percent'] is None else
                            groups[method]['per_class'][i]['iou_percent'] - groups[control]['per_class'][i]['iou_percent']))
                classes.append(dict(name=name, tp=int(tp[i]), fp=int(predicted[i]-tp[i]),
                    fn=int(targets[i]-tp[i]), target_pixels=int(targets[i]), predicted_pixels=int(predicted[i]),
                    precision_percent=None if not predicted[i] else float(100*tp[i]/predicted[i]),
                    recall_percent=None if not targets[i] else float(100*tp[i]/targets[i]),
                    iou_percent=groups[method]['per_class'][i]['iou_percent'], changes=changes))
            errors[d][p] = dict(classes=classes,
                extra_metrics={k:v for k,v in groups[method].items() if k not in
                    ('mean_iou_percent','pixel_accuracy_percent','per_class','confusion_matrix')},
                endpoint_note=protocol['controls'])
    manifest['class_competition']='class_competition.json'
    lines = ['# 最终 Geometry 耦合模型：冻结结果与论文边界', '',
        f'统一 ordinary alias policy：`{policy}`；全部15协议、{count}次协议图像评测及修正后的独立成本完成。VOC20/21、PC59/60共享来源，不能称为同数量独立图像。', '',
        '终版入口为 `TaxonomyInference.for_finalization`，必须显式传入本清单词库和任务信息；历史默认入口不变。这个结论冻结一个整体配置，不拼接历史单域最佳分数。', '',
        '结构是 Geometry 局部读取＋明确归属的 VIP 宽视野观测＋Geometry 支持的修正写回。额外词权重不是核心创新；局部 LME 和宽视野 salience 已使词贡献依赖响应，但没有证明它们能判断语义正确性。', '',
        '```mermaid', 'flowchart LR',
        '  X[RGB图像] --> G[最多4窗：Geometry局部读取]',
        '  X --> V[最多4窗：VIP来源宽视野观测]',
        '  T[冻结词库与模板] --> G', '  T --> V',
        '  G --> L[局部分数L]', '  G --> A[视觉关系A与写回H]',
        '  V --> W[宽视野分数W]', '  L --> Z[Z=L+gH乘以W-L]',
        '  W --> Z', '  A --> Z', '  Z --> Y[概率拼接与原尺寸预测]', '```', '',
        '局部长边896；遥感宽视野长边448，自然宽视野短边336/长边上限672。无fine编码、无按原图尺寸无限增长的滑窗数量。', '',
        '词库不是所有协议固定20：遥感保留冻结的20词库与既定选库规则，自然使用已开发的变长词库（普通前景通常1–18词，VOC20为3–18词），VOC21/COCOObject残余类另含更多表达，PC60还使用额外401残余概念。逐类词数见final_model.json的vocabulary_budget，词表/模板/选路仍由冻结protocol.json给定。变长支持是输入能力，不是已验证的词表稳健性创新。', '',
        '## 完整结果', '',
        '| 协议 | 图像 | Retained | Uniform | 最终版 | 同观测局部 | 同观测等权融合 | VIP声明/蒸馏配置 | VIP原20参考 | 最终−声明VIP pp |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d, row in rows.items():
        for p, groups in row['metrics'].items():
            vals = [groups[m]['mean_iou_percent'] for m in ('Retained','Uniform',method,'UniformLocal','UniformMean')]
            vips = comparisons[d][p]
            v = lambda key: '未完成全量比较' if vips.get(key) is None else f'{vips[key]["miou"]:.4f}'
            delta='未完成全量比较' if vips.get('declared') is None else f'{groups[method]["mean_iou_percent"]-vips["declared"]["miou"]:+.4f}'
            lines.append(f'| {d}/{p} | {row["processed_images"]} | ' + ' | '.join(f'{x:.4f}' for x in vals)
                         + ' | ' + v('declared') + ' | ' + (v('pool20') if 'pool20' in vips else '未测同池全量') + ' | ' + delta + ' |')
    rs = [d for d in protocol['order'] if protocol['datasets'][d]['family']=='remote_sensing']
    means = {m:float(np.mean([rows[d]['metrics']['D' if d=='loveda' else d][m]['mean_iou_percent'] for d in rs]))
             for m in (method,'UniformLocal','UniformMean')}
    manifest['eight_rs_mean']=means
    lines += ['', '八遥感均值（LoveDA D一次）：'+json.dumps(means,ensure_ascii=False)+'。', '',
        'VIP列均为本地有限行数值修复后的测量，不是论文认证复现。遥感三域为官方短词，其余五域为蒸馏外部词库；VIP20是原20适配。Vaihingen采用修正输入。自然缺失的完整比较保留为空；不得引用pilot填补。整体比较的词库/信息预算不同，不能单独归因Geometry。', '',
        '## 冻结判据与配对机制证据', '',
        json.dumps(summary['decision'],ensure_ascii=False), '',
        '全局判据按15个主入口计算，LoveDA取D；P单列，不将这个门槛写成P的额外非劣保证。统计区间条件于已开发数据，未调整方法选择偏倚；文件名分组不是认证独立采集单位。Uniform耦合与Uniform端点/融合是匹配的普通权重对照；Retained相对它们同时包含旧软权重差异，应与Retained−Uniform增量分开解释。', '',
        '全部候选推理结果复用。LoveDA P的主队列历史参考由D词库切片，而独立历史P缓存有3个局部alias向量不同；只补算历史P Reference20SameViews，1669张逐图混淆矩阵与历史精确一致。修正仅进入派生verified_merged.json的参考项；候选Retained/Uniform及原始参考/逐图文件保留，补算来源见reference_recovery字段。D参考本来精确一致。', '',
        '| 面板 | 比较 | Δpp | 95%配对区间 |',
        '| --- | --- | ---: | --- |']
    for panel, pairs in stats['panel_means'].items():
        for pair, outcome in pairs.items():
            lo,hi=outcome['ci95_pp']
            lines.append(f'| {panel} | {pair} | {outcome["delta_pp"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    vip_rs={}
    for d in rs:
        p='D' if d=='loveda' else d
        available=[v['miou'] for v in comparisons[d][p].values() if v is not None]
        if len(available)==2:vip_rs[d]=max(available)
    if len(vip_rs)==8:
        stronger_mean=float(np.mean(list(vip_rs.values())))
        manifest['eight_rs_stronger_measured_vip_mean']=stronger_mean
        lines += ['', f'八遥感较强已测VIP均值{stronger_mean:.4f}；最终版均值差{means[method]-stronger_mean:+.4f}pp。较强参考取两种已完成VIP协议的最大值，不能称最优可能VIP，也不据此切换我们的方法。']
    lines += ['', '## 前景与非残余指标', '',
        '沿用各协议已保存的指标定义，不由类别名称另造评分口径；缺失的VIP附加指标不填估计。', '',
        '| 协议 | 指标 | 最终版 | 局部 | 宽视野 | 等权融合 | 声明VIP |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for d,row in rows.items():
        for p,groups in row['metrics'].items():
            for key in ('foreground_mean_iou_percent','non_residual_mean_iou_percent'):
                if key not in groups[method]:continue
                display=lambda v:'未记录' if v is None else f'{v:.4f}'
                values=[groups[m].get(key) for m in (method,'UniformLocal','UniformWide','UniformMean')]
                vip=(comparisons[d][p].get('declared') or {}).get('extra_metrics',{}).get(key)
                lines.append(f'| {d}/{p} | {key} | '+' | '.join(display(v) for v in (*values,vip))+' |')
    lines += ['',
        '## 独立整图成本', '',
        '| 协议 | 最终版中位ms | p95 ms | VIP声明配置ms | 比率 | allocated MiB | reserved MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in protocol['order']:
        costs=summary['cost_summary'][d]; chosen=costs[method]; vip=costs['VIPOfficial']
        lines.append(f'| {d} | {chosen["median_ms"]:.2f} | {chosen["p95_ms"]:.2f} | {vip["median_ms"]:.2f} | '
                     f'{chosen["median_ms"]/vip["median_ms"]:.3f}× | {chosen["peak_allocated_mib"]:.2f} | {chosen["peak_reserved_mib"]:.2f} |')
    lines += ['', '每协议7个确定性完整图、各7次同步重复、独立进程；包括缩放/全部编码/词聚合/写回/原尺寸恢复/CPU输出，解码、初始化和文本编码另记。不是全量平均延迟。自然VIP采用short336/cap2048，旧long448尺度控制不参与比率。五个非官方遥感域成本是外部20词适配，非蒸馏短词计时。移除权重不保证可测的加速，全部原始记录保留。', '',
        '## 论文主张', '',
        '1. 主张视觉关系组织读取及结构写回的同信息增量，用同观测等权融合、局部端点、最近算子和配对区间支持；不宣称每域都正向迁移。',
        '2. 撤回当前像素级坏词识别/条件语义可靠性的贡献。若保留旧降权，只称经验启发式；若移除，其微小精度代价仍如实报告。可变词数是输入能力，未经词表扰动检验不能升级为稳健性贡献。',
        '3. Geometry原质量保持与patch-only strength1/2/3不同，不混用守恒性质。g=1的二次闭式解是经典线性代数，g=2外推不拥有同一个目标保证。H可含负元素，不等同语义正确性或非负传播。',
        '4. 最近匹配证据：固定20261005 patch-only2相对SCLIP coupled八域平均+0.3075pp、6/8胜，是DINO.text算子适配；不自动覆盖本轮任务路由。原Geometry近邻优势尚未稳定证明。',
        '5. 仍需最终配置的最近方法对照、真正新来源/官方测试、尺度/边界机制证据及文献优先权核查，才足以支撑强CVPR投稿。training-free指不更新网络权重；开发过的词表/参数及PC60额外401残余概念必须公开。', '',
        '投稿下一步按证据优先级进行：先固定本清单并完成最终任务路由的最近读出匹配；再用相同观测预算隔离Geometry关系/写回的作用，并补空间尺度与边界证据；随后一次冻结用于真正新来源或官方测试。只有出现能区分正确覆盖与竞争误激活的新独立证据，才重开词级可靠性研究。当前不以第三个模块数量作为论文完整性的标准。', '',
        '配对区间见 `alias_finalization_20261009/statistics_full/PAIRED_UNCERTAINTY.md`；词级失败归因、Introduction/方法和评测草稿见 `CVPR_FINALIZATION_EXECUTION_20261009.md`；完整逐类结果见 `ALIAS_FINALIZATION_20261009.md`。', '',
        '科学定案与本轮模型冻结完成，不代表已达到全部域SOTA或获得CVPR录用保证。', '',
        '## 类别竞争与负迁移', '',
        '完整逐类IoU、precision/recall、预测/目标像素及相对局部/宽视野/等权融合的TP/FP/FN变化保存于 `class_competition.json`。以下列每协议相对局部和宽视野IoU下降最大的三个类，包括没有下降的情形。类别名不能代替实例尺寸或边界诊断，混淆总数不能重建逐像素转移。PC60端点保留protected residual，属于最终读出端点而非孤立分支。', '',
        '| 协议 | 比较端点 | 类别 | 最终IoU | Δpp | ΔTP | ΔFP | ΔFN |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for d, protocols in errors.items():
        for p, field in protocols.items():
            for control in ('UniformLocal','UniformWide'):
                valid=[c for c in field['classes'] if c['changes'][control]['delta_iou_pp'] is not None]
                for c in sorted(valid,key=lambda c:c['changes'][control]['delta_iou_pp'])[:3]:
                    v=c['changes'][control]
                    lines.append(f'| {d}/{p} | {control} | {c["name"]} | {c["iou_percent"]:.4f} | {v["delta_iou_pp"]:+.4f} | '
                        f'{v["delta_tp"]:+d} | {v["delta_fp"]:+d} | {v["delta_fn"]:+d} |')
    errors_path.write_text(json.dumps(errors,indent=2)+'\n',encoding='utf-8')
    destination.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(policy=policy,manifest=str(destination),report=str(REPORT),eight_rs_mean=means)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.check_only:
        ready,summary,stats=readiness()
        print(json.dumps(dict(ready=ready,accuracy_complete=summary['complete'],cost_complete=summary['cost_complete'],
            verified_protocol_images=summary['verified_protocol_images'],full_statistics_exists=stats.exists())))
    else:
        finalize()
