"""Audit full coverage and report the opt-in Geov2 wide/readout experiments."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS=('vdd','potsdam','ade150','voc20','voc21','coco_object81')
LABELS=dict(vdd='VDD',potsdam='Potsdam',ade150='ADE150',voc20='VOC20',voc21='VOC21',coco_object81='COCO-Object81')


def read(path):
    return json.loads(path.read_text()) if path.exists() and path.stat().st_size else None


def score(tp,pred,target):
    union=pred+target-tp
    return np.nanmean(np.divide(tp,union,out=np.full_like(union,np.nan,dtype=float),where=union>0),axis=-1)*100


def merge(root,variant,dataset):
    records=[]
    for path in (root/'full'/variant).glob(dataset+'_s*/results.json'):
        r=read(path)
        if r and r.get('status')=='complete':records.append((path,r))
    if not records:return None
    count=records[0][1]['num_shards']
    if len(records)!=count:return None
    assert {r['shard_index'] for _,r in records}==set(range(count))
    reference=records[0][1]
    expected=reference['expected_full_keys']
    assert len(expected)==reference['config']['total_images']
    samples={};matrix=np.zeros_like(reference['metrics']['confusion_matrix'],dtype=np.int64)
    for path,r in records:
        assert r['cache_identity']==reference['cache_identity']
        assert r['profile']==reference['profile'] and r['wide_config']==reference['wide_config']
        assert r['expected_full_keys']==expected
        assert r['sample_keys']==expected[r['shard_index']::count]
        assert r['processed_images']==len(r['sample_keys'])
        stat=np.load(path.parent/'statistics.npz')
        assert stat['sample_keys'].tolist()==r['sample_keys']
        cm=np.asarray(r['metrics']['confusion_matrix'],dtype=np.int64)
        assert np.array_equal(stat['tp'].sum(0),cm.diagonal())
        assert np.array_equal(stat['predicted'].sum(0),cm.sum(0))
        assert np.array_equal(stat['target'].sum(0),cm.sum(1))
        for i,k in enumerate(r['sample_keys']):
            assert k not in samples
            samples[k]=tuple(stat[name][i] for name in ('tp','predicted','target'))
        matrix+=cm
    assert set(samples)==set(expected)
    arrays=tuple(np.stack([samples[k][i] for k in expected]) for i in range(3))
    result=dict(status='complete',dataset=dataset,variant=variant,images=len(expected),shards=count,
                mean_iou_percent=float(score(matrix.diagonal(),matrix.sum(0),matrix.sum(1))),
                profile=reference['profile'],wide_config=reference['wide_config'],
                cache_identity=reference['cache_identity'],class_names=reference['class_names'],
                confusion_matrix=matrix.tolist(),unique_full_coverage_verified=True)
    out=root/'merged'/variant;out.mkdir(parents=True,exist_ok=True)
    (out/(dataset+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result,arrays,expected


def interval(a,b,repeats=2000):
    assert np.array_equal(a[2],b[2])
    rng=np.random.default_rng(20261015)
    n=len(a[0]);values=[]
    for start in range(0,repeats,100):
        weight=rng.multinomial(n,np.full(n,1/n),size=min(100,repeats-start)).astype(float)
        sa=score(*(weight@x.astype(float) for x in a))
        sb=score(*(weight@x.astype(float) for x in b))
        values.extend(sb-sa)
    return np.percentile(values,[2.5,97.5]).tolist()


def main(args):
    root=Path(args.root)
    variants=('baseline','shared_canvas','shared_slots48','optimized24','optimized48','ridge01','ridge03','ridge1')
    full={v:{d:merge(root,v,d) for d in DATASETS} for v in variants}
    costs={v:{d:read(root/'stable_cost'/v/(d+'_s0')/'results.json') for d in DATASETS} for v in variants}
    rows=[]
    for d in DATASETS:
        if not full['baseline'][d] or not full[args.variant][d]:continue
        old,oa,ok=full['baseline'][d];new,na,nk=full[args.variant][d]
        assert ok==nk and old['cache_identity']==new['cache_identity']
        lo,hi=interval(oa,na)
        row=dict(dataset=d,images=new['images'],baseline_miou=old['mean_iou_percent'],
                 optimized_miou=new['mean_iou_percent'],delta_pp=new['mean_iou_percent']-old['mean_iou_percent'],
                 paired_95_low=lo,paired_95_high=hi,gain=new['profile']['coupling'])
        cb,cn=costs['baseline'][d],costs[args.variant][d]
        if cb and cn:
            assert [r['key'] for r in cb['images']]==[r['key'] for r in cn['images']]
            assert cb['cpu_threads']==cn['cpu_threads']==1
            row.update(baseline_ms=cb['median_ms'],optimized_ms=cn['median_ms'],
                baseline_allocated_mib=cb['peak_allocated_mib'],optimized_allocated_mib=cn['peak_allocated_mib'],
                speedup=cb['median_ms']/cn['median_ms'],
                memory_saving_percent=(1-cn['peak_allocated_mib']/cb['peak_allocated_mib'])*100,
                optimized_initialization_peak_mib=cn['initialization_peak_allocated_mib'])
        rows.append(row)
    complete=len(rows)==len(DATASETS) and all('speedup' in r for r in rows)
    payload=dict(status='complete' if complete else 'in_progress',variant=args.variant,rows=rows,
                 bootstrap_images=2000,local_path='DonorBefore strength=original',
                 selection='Labeled development screen; fixed words/calibration inherited. No masks used in inference.')
    (root/'SUMMARY.json').write_text(json.dumps(payload,indent=2)+'\n')
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (root/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if args.variant == 'shared_canvas':
        implementation = 'shared_canvas 共享 local FP32 backbone 的参数存储，宽视野保留独立的历史 FP16 head、24 槽构造、局部 raw query 路由和全图类别画布；局部 DonorBefore 与原 H 矩阵写回保持原实现。局部和宽视野仍各自编码自己的 RGB，参数共享不等于特征复用。'
        writer = '结构写回仍使用原实现 `Z=L+g0 H(W-L)`，`H=(I+A^T A)^-1 A^T A`，没有在该候选中换成因子化求解。运行显存减少来自释放重复 backbone；初始化仍先加载两个模型，因此另报启动峰值。'
        numerical = '共享 FP32 backbone 改变了宽视野历史 FP16 权重/输入舍入，不能声称宽视野或最终预测逐像素等价。局部 DonorBefore 计算路径未修改；其特征在已有 smoke 中逐元素相等。'
    else:
        implementation = 'optimized48 只保留一份冻结视觉模型，局部和宽视野各自编码各自的 RGB；共享参数不等于特征复用。宽视野的球面槽构造使用 gather/scatter，避免 CUDA 标量索引的主机同步。局部 raw query 仍用于选择 48 个语义槽，宽视野 class logits 直接在各局部 patch 网格读出，不再先铺满全图类别分数画布。其他候选的具体设置以 wide_config 为准。'
        writer = '结构写回使用同一 A、同一目标与固定步长：`Z=L+g0 H(W-L)`，`H=(I+A^T A)^-1 A^T A`。保存 Cholesky 因子 R 后，直接计算 `H D = D - solve(R R^T,D)`，把求解的右端数从 N 减为类别数 C。最终 optimized48 使用 FP64 因子求解；早期 FP32 候选另行保留。'
        numerical = '直接 patch 读出取消了旧画布的二次插值和重叠混合，共享 FP32 视觉参数也改变了宽视野的历史 FP16 舍入。因此整个候选是模型/执行联合优化，不能声称全部最终预测与旧版逐像素相同；局部 DonorBefore 单独的特征等价已检查。'
    lines=['# Geov2 宽视野与耦合读出优化（2026-10-15）','',
        f'状态：{"全量及成本测量完成" if complete else "实验仍在运行；以下仅列已完成的记录"}。当前报告候选：`{args.variant}`。','',
        '## 实验约束与模型','',
        '局部 DonorBefore 的权重、FP32 存储、RGB 预处理、512 窗口、视觉关系 A 和 head 投影路径保持原实现。语言端沿用 10/13 的冻结 Alias、模板、温度与背景规则。本轮优化宽视野和耦合执行；没有训练，也没有在推理时读取标签。','',
        '可复现基准是当前 QuerySlot-24 + DonorBefore，使用 YAML 的原有逐域 g0。旧 Efficient 使用了不同宽视野和局部配置，作为历史精度/成本参考，不能把两者混为一条配对基准。','',
        implementation,'',writer,'',numerical,'',
        '## 全量精度','',
        '| 协议 | 张数 | 原 QuerySlot-24 | 当前候选 | Δ pp | 配对 95% 区间 pp | g0 |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        lines.append(f"| {LABELS[r['dataset']]} | {r['images']} | {r['baseline_miou']:.4f} | {r['optimized_miou']:.4f} | {r['delta_pp']:+.4f} | [{r['paired_95_low']:+.4f}, {r['paired_95_high']:+.4f}] | {r['gain']:g} |")
    lines+=['','每行全量唯一覆盖已核验；COCO 两个分片的类别混淆矩阵相加后再算 mIoU。区间使用 2,000 次配对图像 bootstrap，不能当作独立测试集的保证。VOC20 与 VOC21 共用同一图像集合。','',
        '## 成本','',
        '| 协议 | 原耗时 ms | 当前耗时 ms | 加速比 | 原 allocated MiB | 当前 allocated MiB | 显存减少 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if 'speedup' in r:lines.append(f"| {LABELS[r['dataset']]} | {r['baseline_ms']:.2f} | {r['optimized_ms']:.2f} | {r['speedup']:.3f}× | {r['baseline_allocated_mib']:.2f} | {r['optimized_allocated_mib']:.2f} | {r['memory_saving_percent']:.1f}% |")
    lines+=['','成本边界：CPU RGB tensor → 原尺寸 CPU uint8 标签，包含预处理、GPU 推理、拼接及标签回传；不包含文件解码、初始化、文本编码。每域 7 张固定图，每张 2 次 warmup、7 次同步重复，汇总全部 49 次的中位数；显存列为 torch peak allocated，reserved 在原始 JSON 中。最终配对成本测量使用各进程 1 个 CPU 线程，并在其他全量任务结束后进行。早期线程竞争期间的 cost/cpu1 结果保留，但不用于最终速度结论。','',
        '启动峰值另外记录在 `initialization_peak_allocated_mib`，不把释放重复模型后的运行显存冒充启动显存。','',
        '## 开发筛选及负结果','',
        '前四域使用各 16 张预声明开发图，VOC21/COCO 各 64 张等距样本。类别覆盖较少，筛选 mIoU 不能与全量值直接比较。所有已完成候选如下。','',
        '| 候选 | VDD | Potsdam | ADE150 | VOC20 | VOC21 | COCO-Object81 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for folder in sorted((root/'screen').glob('*')):
        if not folder.is_dir():continue
        vals=[read(folder/(d+'_s0')/'results.json') for d in DATASETS]
        lines.append('| '+folder.name+' | '+' | '.join(f"{r['metrics']['mean_iou_percent']:.4f}" if r else '—' for r in vals)+' |')
    lines+=['','原生语义头与缩小宽视野分辨率明显损害部分开发域，未直接推广。低正则的回归式槽读出在 VOC20 上退化；后续仅在预设 ridge={0.1,0.3,1.0} 菜单中检查正则化能否消除该问题，开发选择必须披露，不能声称无需适配。','',
        '## VOC21 步长对照与旧版参考','']
    gain1=read(root/'gain1_voc21_results.json')
    if gain1:lines.append(f"optimized48 的全量 VOC21、g0=1 为 **{gain1['metrics']['mean_iou_percent']:.4f}%**；这是耦合步长对照，不能归因于宽视野改动本身。对应当前旧 QuerySlot 在既有全量 g0 扫描中的 g0=1 为 75.2094%。")
    lines+=['','历史 Efficient 参考：VOC21 75.8586%，COCO-Object81 50.3999%，VDD 60.4352%。必须逐项比较；小幅波动和区间覆盖零时按精度持平解释，不写“全面提升”或“全面 SOTA”。','',
        '## 运行方式','',
        '在原 infer/evaluate 参数基础上，速度优先使用 `--wide-branch query_slot --efficient --wide-readout shared_canvas --cpu-threads 1`；显存优先候选使用 `--wide-readout optimized48`。默认 `--wide-readout incumbent` 保持旧路径。需要已有冻结词表缓存；YAML 的 local strength 必须为 original。支持本轮 retained/union_max 背景规则；PC60 的 residual 读出和 LoveDA P 独立前景读出尚未纳入本轮验证。','',
        '原始证据：`full/`、`screen/`、`stable_cost/`、`merged/`、`VERIFICATION.json`、`assignment_benchmark.json`；`source/` 保存代码快照。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=payload['status'],variant=args.variant,completed_full=len(rows),complete_cost=sum('speedup' in r for r in rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='research/geov2_wide_optimized_20261015')
    p.add_argument('--variant',default='optimized48')
    main(p.parse_args())
