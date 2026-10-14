"""Audit all 40 UDD5 images, summarize fixed-step ablations and paired CIs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAME='geov2_udd5_fixed05_ablation_20261014'
WORKSPACE=Path(__file__).resolve().parents[1]

DESCRIPTIONS={
 'Full':'31词＋DonorBefore＋QuerySlot24＋H 写回',
 'NoDonorBefore':'仅关闭 DonorBefore，退回原 Geometry 局部头',
 'NoQueryRouting':'仅关闭 query 路由（beta=0），保留24槽与采样顺序',
 'NoDonorNoRouting':'同时关闭 DonorBefore 和 query 路由',
 'ShuffledQuery':'raw query 沿token序列循环移位512位，其余不变',
 'CanonicalForeground':'前景每类仅类名，保留20个背景概念，共24词',
 'CanonicalAll':'每类仅类名，背景也缩为 other，共5词',
 'UnscreenedAliases':'同模板未筛选候选池，每类20词，共100词',
 'WideCountNormalized':'保留31词，仅宽视野减去 log(每类词数)/tau',
 'EqualLogitFusion':'同一 L、W，使用0.5L＋0.5W，关闭H写回',
 'LocalOnly':'同一观测的局部端点 L',
 'WideOnly':'同一观测的宽视野端点 W（已受局部query条件化）',
 'Raw24CoordinateGrid':'DonorBefore＋24槽整图坐标读出，非query路由',
 'Raw32CoordinateGrid':'DonorBefore＋32槽整图坐标读出，非query路由',
 'LegacyGeometryVIPWide':'原Geometry＋旧VIP crop宽视野＋H',
 'LegacyDonorVIPWide':'DonorBefore＋旧VIP crop宽视野＋H',
}


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def scores(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    union=cm.sum(-1)+cm.sum(-2)-tp
    iou=np.divide(tp,union,out=np.full(tp.shape,np.nan),where=union>0)*100
    return np.nanmean(iou,axis=-1),iou


def main():
    root=WORKSPACE/'research'/NAME
    protocol=json.loads((root/'protocol.json').read_text())
    expected=json.loads((WORKSPACE/'research/geov2_g0_sweep_20261014/expected_keys.json').read_text())['udd5']
    previous=json.loads((WORKSPACE/'research/geov2_g0_sweep_20261014/full/udd5_s0/results.json').read_text())
    arrays,keys,first=[],[],None
    for shard in range(protocol['shards']):
        folder=root/'full'/f's{shard}'
        row=json.loads((folder/'results.json').read_text())
        with np.load(folder/'confusions.npz',allow_pickle=False) as z:
            array=z['confusions'].copy()
            local_keys=z['sample_keys'].tolist()
        if (row['status']!='complete' or row['sample_keys']!=expected[shard::protocol['shards']]
            or local_keys!=row['sample_keys'] or array.shape!=(len(local_keys),16,5,5)
            or row['shard_index']!=shard or row['num_shards']!=protocol['shards']
            or row['global_sample_count']!=40 or row['processed_images']!=len(local_keys)
            or row['g0']!=.5 or row['profile']['strength']!='original'
            or row['adaptive_gain_enabled'] or not row['masks_used_only_for_scoring']):
            raise ValueError('Invalid shard: '+str(folder))
        if not np.all(array.sum(-1)==array[:,:1].sum(-1)):
            raise ValueError('Target support differs across methods.')
        for i,name in enumerate(row['methods']):
            cm=array[:,i].sum(0)
            if not np.array_equal(cm,row['metrics'][name]['confusion_matrix']):
                raise ValueError('Per-image statistics do not sum to recorded confusion.')
        if first is not None:
            for field in ('methods','profile','config','alias_counts','cache_identity','class_names','arms'):
                if row[field]!=first[field]:
                    raise ValueError('Shard metadata differs: '+field)
        first=first or row
        arrays.append(array)
        keys+=local_keys
    if len(keys)!=40 or len(set(keys))!=40 or set(keys)!=set(expected):
        raise ValueError('Missing or duplicate validation images.')
    all_cm=np.concatenate(arrays,axis=0)
    order=[keys.index(key) for key in expected]
    all_cm=all_cm[order]
    methods=first['methods']
    total=all_cm.sum(0)
    if not np.array_equal(total[0].sum(1),np.asarray(previous['metrics']['retained__g0.5']['confusion_matrix']).sum(1)):
        raise ValueError('Validation target supports differ from previous sweep.')
    values,per_class=scores(total)
    old=previous['metrics']['retained__g0.5']['mean_iou_percent']
    if abs(values[0]-old)>.01:
        raise ValueError('Full is not a close reproduction of the frozen configuration.')
    rng=np.random.default_rng(20261014)
    # Paired image bootstrap: the same resampled image indices for every arm.
    draws=rng.integers(0,40,size=(10000,40))
    weights=np.stack([np.bincount(draw,minlength=40) for draw in draws])
    boot,_=scores(np.einsum('bn,nmij->bmij',weights,all_cm))
    full_index=methods.index('Full')
    rows=[]
    for i,name in enumerate(methods):
        differences=boot[:,full_index]-boot[:,i]
        row=dict(method=name,description=DESCRIPTIONS[name],mean_iou_percent=float(values[i]),
            full_minus_arm_pp=float(values[full_index]-values[i]),
            paired_95ci_pp=np.quantile(differences,[.025,.975]).tolist(),
            class_iou_percent=per_class[i].tolist(),confusion_matrix=total[i].tolist())
        if name in protocol['cost_arms']:
            cost=json.loads((root/'cost'/name/'results.json').read_text())
            if (cost['status']!='complete' or cost['methods']!=[name]
                or [r['key'] for r in cost['images']]!=protocol['cost_keys']
                or any(len(r['milliseconds'])!=7 for r in cost['images'])
                or cost['target_masks_loaded'] or cost['g0']!=.5):
                raise ValueError('Invalid independent cost: '+name)
            row.update({k:cost[k] for k in ('median_ms','peak_allocated_mib','peak_reserved_mib')})
        rows.append(row)
    lookup={r['method']:r for r in rows}
    def effect(a,b):
        x,y=methods.index(a),methods.index(b)
        return dict(a=a,b=b,difference_pp=float(values[x]-values[y]),
                    paired_95ci_pp=np.quantile(boot[:,x]-boot[:,y],[.025,.975]).tolist())
    interaction=(boot[:,methods.index('Full')]-boot[:,methods.index('NoQueryRouting')]
                 -boot[:,methods.index('NoDonorBefore')]+boot[:,methods.index('NoDonorNoRouting')])
    effects=dict(donor_with_routing=effect('Full','NoDonorBefore'),
        donor_without_routing=effect('NoQueryRouting','NoDonorNoRouting'),
        routing_with_donor=effect('Full','NoQueryRouting'),
        routing_without_donor=effect('NoDonorBefore','NoDonorNoRouting'),
        query_correspondence=effect('Full','ShuffledQuery'),
        foreground_expansion=effect('Full','CanonicalForeground'),
        kev_screening=effect('Full','UnscreenedAliases'),
        structure_vs_mean=effect('Full','EqualLogitFusion'),
        capacity32_vs24=effect('Raw32CoordinateGrid','Raw24CoordinateGrid'),
        donor_routing_interaction=dict(difference_pp=float(
            lookup['Full']['mean_iou_percent']-lookup['NoQueryRouting']['mean_iou_percent']
            -lookup['NoDonorBefore']['mean_iou_percent']+lookup['NoDonorNoRouting']['mean_iou_percent']),
            paired_95ci_pp=np.quantile(interaction,[.025,.975]).tolist()))
    controls=json.loads((root/'writer_controls/results.json').read_text())
    with np.load(root/'writer_controls/confusions.npz',allow_pickle=False) as z:
        control_cm=z['confusions'].copy()
        control_keys=z['sample_keys'].tolist()
    if (controls['status']!='complete' or control_keys!=expected or controls['sample_keys']!=expected
        or controls['g0']!=.5 or not controls['observations_shared']
        or controls['profile']!=first['profile']
        or not controls['masks_used_only_for_scoring']):
        raise ValueError('Writer controls are incomplete or mismatched.')
    if not np.all(control_cm.sum(-1)==control_cm[:,:1].sum(-1)) or not np.array_equal(control_cm[:,0].sum((0,2)),total[0].sum(1)):
        raise ValueError('Writer control target support differs.')
    control_values,_=scores(control_cm.sum(0))
    control_boot,_=scores(np.einsum('bn,nmij->bmij',weights,control_cm))
    control_rows=[]
    for i,name in enumerate(controls['methods']):
        if not np.array_equal(control_cm[:,i].sum(0),controls['metrics'][name]['confusion_matrix']):
            raise ValueError('Writer control statistics do not sum correctly.')
        control_rows.append(dict(method=name,mean_iou_percent=float(control_values[i]),
            full_minus_arm_pp=float(control_values[0]-control_values[i]),
            paired_95ci_pp=np.quantile(control_boot[:,0]-control_boot[:,i],[.025,.975]).tolist()))
    if abs(control_values[0]-values[0])>.01:
        raise ValueError('Supplemental full reproduction differs.')
    summary=dict(status='complete',dataset='udd5',g0=.5,images=40,methods=methods,class_names=first['class_names'],
        sample_keys=expected,profile=first['profile'],config=first['config'],alias_counts=first['alias_counts'],
        rows=rows,effects=effects,previous_full_miou=old,reproduction_delta_pp=float(values[0]-old),
        bootstrap=dict(unit='image',paired=True,replicates=10000,seed=20261014,
                       note='Descriptive percentile intervals conditional on this inspected 40-image split; no multiple-comparison correction.'),
        writer_controls=dict(rows=control_rows,diagnostics=controls['diagnostics'],
            average_valid_h_diagonal=controls['diagnostics']['valid_h_trace']/controls['diagnostics']['patches'],
            rms_h_delta_over_delta=float(np.sqrt(controls['diagnostics']['correction_energy']/controls['diagnostics']['difference_energy']))),
        all_shards_audited=True,all_per_image_sums_verified=True)
    dump(root/'SUMMARY.json',summary)
    np.savez_compressed(root/'per_image_confusions.npz',sample_keys=np.asarray(expected),
        methods=np.asarray(methods),class_names=np.asarray(first['class_names']),confusions=all_cm)
    with (root/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['method','mIoU_percent','Full_minus_arm_pp','CI_low_pp','CI_high_pp','median_ms','peak_allocated_MiB','peak_reserved_MiB',*first['class_names']])
        for r in rows:
            writer.writerow([r['method'],r['mean_iou_percent'],r['full_minus_arm_pp'],*r['paired_95ci_pp'],
                *[r.get(k,'') for k in ('median_ms','peak_allocated_mib','peak_reserved_mib')],*r['class_iou_percent']])
    plots(root,lookup,summary['writer_controls'])
    report(root,summary,lookup)
    print(json.dumps({name:round(lookup[name]['mean_iou_percent'],4) for name in methods},indent=2))


def plots(root,lookup,writer_controls):
    panels=[('Visual components',['NoDonorBefore','NoQueryRouting','NoDonorNoRouting','ShuffledQuery']),
            ('Same-observation writer',['LocalOnly','WideOnly','EqualLogitFusion']),
            ('Language interface',['CanonicalForeground','CanonicalAll','UnscreenedAliases','WideCountNormalized'])]
    labels={'NoDonorBefore':'Remove DonorBefore','NoQueryRouting':'Disable query routing',
            'NoDonorNoRouting':'Disable both','ShuffledQuery':'Shift local queries',
            'LocalOnly':'Local endpoint','WideOnly':'Wide endpoint','EqualLogitFusion':'Equal logit fusion',
            'CanonicalForeground':'Canonical foreground','CanonicalAll':'Canonical all',
            'UnscreenedAliases':'Unscreened 100 aliases','WideCountNormalized':'Normalize wide alias count'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,(title,methods) in zip(axes,panels):
        for y,name in enumerate(methods):
            r=lookup[name]
            value=r['full_minus_arm_pp']
            low,high=r['paired_95ci_pp']
            ax.plot([low,high],[y,y],color='#6d7681',lw=2)
            ax.scatter([value],[y],c='#147d92' if value>=0 else '#c65b3d',s=45,zorder=3)
            ax.annotate(f'{value:+.2f}',(value,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.axvline(0,color='black',lw=.8,ls='--')
        ax.set_yticks(range(len(methods)),[labels[x] for x in methods],fontsize=9)
        ax.set_ylim(len(methods)-.5,-.6)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('Full minus control (mIoU points)')
        ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('UDD5 full validation (40 images), fixed g0 = 0.5\nPaired image bootstrap: 95% descriptive intervals',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(root/('component_effects.'+ext),dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
    labels={'IsotropicHalfH':'H replaced by 0.5 I','DiagonalH':'Only diagonal of H','SpatialH':'Spatial relation only'}
    for y,r in enumerate(writer_controls['rows'][2:]):
        low,high=r['paired_95ci_pp']
        value=r['full_minus_arm_pp']
        ax.plot([low,high],[y,y],lw=2,color='#6d7681')
        ax.scatter([value],[y],color='#147d92',s=45,zorder=3)
        ax.annotate(f'{value:+.3f} pp',(value,y),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_yticks(range(3),list(labels.values()))
    ax.set_ylim(2.5,-.6)
    ax.axvline(0,color='black',lw=.8,ls='--')
    ax.set_xlabel('Full minus control (mIoU points)')
    ax.set_title('Writer mechanism: identical L/W, fixed g0 = 0.5\nUDD5, 40 images; paired 95% intervals',fontsize=11)
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):
        fig.savefig(root/('writer_controls.'+ext),dpi=200)
    plt.close(fig)


def report(root,summary,lookup):
    full=lookup['Full']
    lines=['# UDD5：固定 g0=0.5 的完整部件消融','',
        '2026-10-14。完整验证集40张、5类，16个预声明对照。所有使用H的行固定g0=0.5；没有自适应gate，也没有针对消融重新选择词表、温度、阈值或背景偏置。',
        '',f"**Full：{full['mean_iou_percent']:.4f}% mIoU，{full['median_ms']:.2f} ms，峰值 allocated {full['peak_allocated_mib']:.2f} MiB。**",
        f"相对前轮55.2567% 的复现差为 {summary['reproduction_delta_pp']:+.6f}pp。所有40张唯一覆盖、分片配置和逐图混淆矩阵之和均已核验。",'',
        '## 1. 完整模型与主消融','',
        'Full = 冻结31词＋局部DonorBefore＋宽视野24槽QuerySlot（beta=0.5，query温度0.1）＋Z=L+0.5H(W−L)。',
        'H=(I+AᵀA)⁻¹AᵀA；局部strength=original，使DonorBefore确实执行。Full宽视野没有VIP proxy self-correction或salience；词表的VIP历史来源、冻结dino.txt骨干及接口兼容不等于把官方VIP作为模型部件。','',
        '| 配置 | mIoU % | Full−该行 pp | 配对95%区间 pp | 耗时 ms | allocated MiB |',
        '| --- | ---: | ---: | --- | ---: | ---: |']
    primary=['Full','NoDonorBefore','NoQueryRouting','NoDonorNoRouting','ShuffledQuery','CanonicalForeground',
        'UnscreenedAliases','WideCountNormalized','EqualLogitFusion','LocalOnly','WideOnly']
    for name in primary:
        r=lookup[name]
        lo,hi=r['paired_95ci_pp']
        lines.append(f"| {name} | {r['mean_iou_percent']:.4f} | {r['full_minus_arm_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "+
            (f"{r['median_ms']:.2f} | {r['peak_allocated_mib']:.2f} |" if 'median_ms' in r else '— | — |'))
    lines += ['', '差值为Full减去对照；负数表示该对照更好。区间为10,000次配对图像bootstrap的百分位区间，不把像素当独立样本。40张较少，区间只描述这份已查看验证集的不确定性，不代表独立测试或跨数据集保证。','',
              '![component effects](component_effects.png)','', '## 2. 如何解释各部件','']
    lines += [
        f"主表的观察是：结构写回相对等权融合提高{lookup['EqualLogitFusion']['full_minus_arm_pp']:.4f}pp，保留背景不变的前景扩词提高{lookup['CanonicalForeground']['full_minus_arm_pp']:.4f}pp。冻结Kev词表相对未筛候选池平均提高{lookup['UnscreenedAliases']['full_minus_arm_pp']:.4f}pp，但其配对区间跨过0，不能据此称为稳定筛词收益。",
        f"但Full比关闭路由低{-lookup['NoQueryRouting']['full_minus_arm_pp']:.4f}pp；Raw32坐标宽视野与旧VIP宽视野＋DonorBefore也分别达到{lookup['Raw32CoordinateGrid']['mean_iou_percent']:.4f}%和{lookup['LegacyDonorVIPWide']['mean_iou_percent']:.4f}%。因此不能写成QuerySlot在UDD5上是必需的精度模块，也不能仅凭打乱控制下降就宣称路由超过坐标基线。保留路由需要其他数据集或效率收益支持。",'']
    effect_labels=[('donor_with_routing','保留query路由时，DonorBefore的条件收益'),
        ('routing_with_donor','保留DonorBefore时，query路由的条件收益'),
        ('query_correspondence','正确局部query对应相对循环移位控制'),
        ('foreground_expansion','保留同样背景时，入选前景扩词的净效果'),
        ('kev_screening','相同模板和背景下，Kev筛选31词相对未筛100词'),
        ('structure_vs_mean','同一L/W下，结构写回相对等权logit融合')]
    for key,label in effect_labels:
        r=summary['effects'][key]
        lo,hi=r['paired_95ci_pp']
        lines.append(f"- {label}：{r['difference_pp']:+.4f}pp，95%区间[{lo:+.4f}, {hi:+.4f}]。")
    interaction=summary['effects']['donor_routing_interaction']
    lines += ['',f"DonorBefore×路由的差分交互为 {interaction['difference_pp']:+.4f}pp。部件贡献是条件收益，不能把各项删除差值直接相加当作总增益。",'',
        'NoQueryRouting使用beta=0：24槽、宽视野head、局部裁窗、坐标特征采样、采样后归一化与计分顺序都保留，只不混入query选出的槽值。ShuffledQuery使用固定循环移位512个token（在32×32网格上相当于16行位移），保留query多重集，破坏对应位置；它是位置对应控制，不是随机多seed实验。',
        f"逐类来看，DonorBefore使五类IoU全部提高；query路由相对关闭路由使road IoU变化{lookup['Full']['class_iou_percent'][2]-lookup['NoQueryRouting']['class_iou_percent'][2]:+.4f}pp、building变化{lookup['Full']['class_iou_percent'][1]-lookup['NoQueryRouting']['class_iou_percent'][1]:+.4f}pp，是其均值下降的主要来源。等权融合vehicle IoU为{lookup['EqualLogitFusion']['class_iou_percent'][3]:.4f}%，Full为{lookup['Full']['class_iou_percent'][3]:.4f}%。",
        '', '## 3. Alias的控制与限制','',
        '当前Kev词数为[7,1,2,1,20]；CanonicalForeground为[1,1,1,1,20]；CanonicalAll为[1,1,1,1,1]；UnscreenedAliases为[20,20,20,20,20]。全部使用同一个seg_template及相同冻结文本编码，背景20词在前景扩词和筛词对照中完全不变。',
        '局部Alias用温度0.07的log-mean-exp，之后除以固定局部校准温度0.05；宽视野使用40倍相似度和tau=1的log-sum-exp，没有除词数，也没有salience。WideCountNormalized只去掉宽视野log(词数)/tau偏置，不删词。',
        '因此Full相对未筛词池的变化包含语义筛选与词数先验的共同效果；不能全部解释成Kev理解语义的收益。CanonicalAll会同时删掉19个背景概念，不能当作纯前景同义词消融。Kev阈值0.6和当前词表曾经用带标签开发数据选择，本轮只是冻结使用，不能称为完全无标签开发。','',
        '## 4. 历史宽视野与容量参考','',
        '| 配置 | mIoU % | 耗时 ms | allocated MiB |',
        '| --- | ---: | ---: | ---: |']
    for name in ('Raw24CoordinateGrid','Raw32CoordinateGrid','LegacyGeometryVIPWide','LegacyDonorVIPWide','CanonicalAll'):
        r=lookup[name]
        lines.append(f"| {name} | {r['mean_iou_percent']:.4f} | {r['median_ms']:.2f} | {r['peak_allocated_mib']:.2f} |")
    lines += ['',
        'Raw24/Raw32先在宽视野整图坐标网格计分，再向局部采样分数；Full和NoQueryRouting先采样/混合特征再计分，非线性Alias聚合与归一化不一定与采样交换。所以不能把Full−Raw32全归因于query路由；主路由结论应看Full−NoQueryRouting。Raw24与Raw32之间则只改变槽容量。',
        'LegacyGeometryVIPWide与LegacyDonorVIPWide是历史模型内部宽视野替换对照：long448、crop336/stride224、相同31词与g0=0.5。它们含有局部分支和结构写回，绝不是官方VIP完整模型。VIP原生滑窗stride112且有自己的配置，UDD5也没有官方配置；本轮没有把旧VIP宽视野冒充官方VIP外部基线。','',
        '## 5. 逐类结果','',
        '| 配置 | '+' | '.join(summary['class_names'])+' |',
        '| --- | '+' | '.join(['---:']*5)+' |']
    for name in primary:
        r=lookup[name]
        lines.append('| '+name+' | '+' | '.join(f'{x:.4f}' for x in r['class_iou_percent'])+' |')
    lines += ['', '## 6. 结构与修正幅度的同观测补充对照','',
        '这组控制在看过主消融表后另行预声明并运行，协议保存在writer_controls_protocol.json。使用同一L/W和g0=0.5：IsotropicHalfH将H替换为0.5I（即0.75L+0.25W），DiagonalH只保留H对角项，SpatialH用纯空间Gaussian关系代替视觉＋空间A来构造H。它们不重选步长或语言参数。',
        '0.5I仅是预声明的固定减幅对照，并非证明它与H谱界或能量完全匹配。Full和EqualLogitFusion在这次重新编码的观测上重新计分，比较应使用此补充表内部差值。','',
        '| 写回 | mIoU % | Full−该行 pp | 配对95%区间 pp |',
        '| --- | ---: | ---: | --- |']
    for r in summary['writer_controls']['rows']:
        lo,hi=r['paired_95ci_pp']
        lines.append(f"| {r['method']} | {r['mean_iou_percent']:.4f} | {r['full_minus_arm_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['',
        f"全量有效patch的平均H对角为{summary['writer_controls']['average_valid_h_diagonal']:.6f}；修正与差分的RMS比 ||H(W−L)||/||W−L||={summary['writer_controls']['rms_h_delta_over_delta']:.6f}。这些量仅作无标签机制诊断，不用于选参。",
        'Full相对固定减幅融合提高0.5740pp且区间不含0，相对仅对角H提高3.3950pp且区间不含0，支持跨位置结构写回超越单纯固定减幅。Full相对纯空间H提高0.5148pp，但区间包含0，因此视觉关系相对空间先验的额外贡献仍需更多数据；不宣称已经证明其不可替代。','',
        '![writer controls](writer_controls.png)','',
        '## 7. 成本和数值边界','',
        'A800，batch=1；所有成本配置在同一张GPU1顺序执行，完整精度作业结束后才开始。每个配置独立新进程，只保留该行语言bank；同一7张固定图，每图2次预热、7次CUDA同步重复。耗时为49次中位数，显存为CUDA max_memory_allocated峰值；reserved另存于JSON/CSV。',
        '计时包含CPU float RGB输入后的缩放、全部视觉编码、分支读出、H求解（如使用）、原尺寸恢复和CPU uint8输出；排除解码、模型初始化、离线文本编码及标签。等权融合的独立成本省去不用的H求解。两个视觉模型按现有部署常驻，没有以多消融共驻显存冒充单模型成本。端点只评精度，未作为独立部署优化后计时。',
        'NoQueryRouting/NoDonorNoRouting在当前函数中以beta=0关闭输出路由，仍计算未混入的槽路由中间量；它们的时延是该控制实现的实测成本，不代表彻底删除路由计算后的最小成本。三项补充H控制只做同观测精度评测，未作为部署模型单独计时。',
        '顺序成本存在运行状态波动；例如Raw24/Raw32的槽数差不能据本轮数毫秒差值推断理论加速，去H的等权融合也没有测得更低中位延迟。小幅时延差只报告实测值，不宣称稳定速度收益。',
        'CUDA槽聚类的index_add累加有微小重复编码差异；smoke检查在同一观测下与FrozenGeometry.predict逐像素一致，并保留重新编码的像素差异记录。Full全量与前轮复现差见开头，不将这种小数值波动解释为部件贡献。',
        '', '## 8. 论文用途','',
        'UDD5适合承担一套完整部件消融，证明这些机制在本数据集上的条件效果。论文若要声称部件对遥感和自然图像普遍有效，应至少补一个自然协议的关键行；无需把16行全部重跑所有数据集。十协议统一g0主表仍应保留，避免仅凭UDD5这一效果较强协议宣称普遍优势。',
        '', '复现入口：[评测脚本](../../DINOtool/scripts/eval_geov2_udd5_fixed05.py)、[H控制脚本](../../DINOtool/scripts/eval_geov2_udd5_writer_controls.py)、[统计和作图](../../tools/report_udd5_fixed05_ablation.py)。',
        '在已配置项目和模型依赖的环境中，将protocol.json中的数据/检查点/文本cache路径改为自己的路径，设置PYTHONPATH包含DINOtool、DINOtool/scripts和固定DINO_Soars依赖；运行评测脚本 --experiment 本目录 --mode full。使用 --mode cost --arm Full 运行独立成本。四分片及顺序成本的实际A800命令也保留于full.sh和cost.sh；输出目录存在时脚本拒绝覆盖。',
        '数据：[SUMMARY.json](SUMMARY.json)、[RESULTS.csv](RESULTS.csv)、[逐图混淆矩阵](per_image_confusions.npz)、[冻结协议](protocol.json)、full/与cost/。','']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
