"""Audit the frozen VOC21 replication and publish paired image-level effects."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[1]
NAME='geov2_voc21_fixed05_ablation_20261014'
DESCRIPTIONS={
    'Full':'冻结Alias＋DonorBefore＋QuerySlot24＋g0=0.5结构写回',
    'NoDonorBefore':'只关闭DonorBefore，其他设置相同',
    'NoQueryRouting':'只设beta=0，保留24槽及先采样特征后计分顺序',
    'NoDonorNoRouting':'同时关闭DonorBefore与query路由',
    'ShuffledQuery':'局部raw query沿token序列循环移位512位',
    'CanonicalForeground':'每个前景只保留类名，背景56词不变',
    'WideCountNormalized':'仅宽视野减去log(每类词数)/tau',
    'EqualLogitFusion':'相同L/W的0.5L＋0.5W，不使用H',
    'LocalOnly':'相同观测的局部端点L',
    'WideOnly':'相同观测的宽视野端点W，仍受局部query条件化',
    'Raw24CoordinateGrid':'DonorBefore＋24槽整图坐标网格先计分后采样',
    'Raw32CoordinateGrid':'DonorBefore＋32槽整图坐标网格先计分后采样',
    'LegacyDonorVIPWide':'DonorBefore＋内部旧VIP宽视野替换，非官方完整VIP',
}


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def scores(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    union=cm.sum(-1)+cm.sum(-2)-tp
    iou=np.divide(tp,union,out=np.full(tp.shape,np.nan),where=union>0)*100
    return np.nanmean(iou,axis=-1),iou


def paired_bootstrap(cm,replicates=10000):
    # IoU needs only per-image TP and union; no B x N x M x C x C tensor.
    n,m,c,_=cm.shape
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    union=cm.sum(-1)+cm.sum(-2)-tp
    stats=np.stack((tp,union),axis=-1).reshape(n,-1).astype(np.float64)
    rng=np.random.default_rng(20261014)
    result=np.empty((replicates,m),dtype=np.float64)
    for start in range(0,replicates,100):
        count=min(100,replicates-start)
        draws=rng.integers(0,n,size=(count,n))
        weights=np.stack([np.bincount(draw,minlength=n) for draw in draws]).astype(np.float64)
        total=(weights@stats).reshape(count,m,c,2)
        iou=np.divide(total[...,0],total[...,1],out=np.full((count,m,c),np.nan),where=total[...,1]>0)
        result[start:start+count]=np.nanmean(iou,axis=-1)*100
    return result


def main():
    root=WORKSPACE/'research'/NAME
    protocol=json.loads((root/'protocol.json').read_text())
    expected=json.loads((WORKSPACE/'research/geov2_g0_sweep_20261014/expected_keys.json').read_text())['voc21']
    old=json.loads((WORKSPACE/'research/geov2_g0_sweep_20261014/voc21_audited.json').read_text())
    old_support=np.zeros(21,dtype=np.int64)
    for shard in range(2):
        previous=json.loads((WORKSPACE/f'research/geov2_g0_sweep_20261014/full/voc21_s{shard}/results.json').read_text())
        old_support+=np.asarray(previous['metrics']['retained__g0.5']['confusion_matrix']).sum(1)
    arrays,keys,first=[],[],None
    for shard in range(protocol['shards']):
        folder=root/'full'/f's{shard}'
        row=json.loads((folder/'results.json').read_text())
        with np.load(folder/'confusions.npz',allow_pickle=False) as z:
            cm=z['confusions'].copy()
            shard_keys=z['sample_keys'].tolist()
        if (row['status']!='complete' or row['sample_keys']!=expected[shard::protocol['shards']]
            or shard_keys!=row['sample_keys'] or cm.shape!=(len(shard_keys),len(protocol['methods']),21,21)
            or row['shard_index']!=shard or row['num_shards']!=protocol['shards']
            or row['global_sample_count']!=1449 or row['processed_images']!=len(shard_keys)
            or row['g0']!=.5 or row['profile']['strength']!='original'
            or row['profile']['coupling']!=.5 or row['profile']['bank']!='official_segmentation'
            or row['adaptive_gain_enabled'] or not row['masks_used_only_for_scoring']
            or row['methods']!=protocol['methods']):
            raise ValueError('Invalid full shard: '+str(folder))
        if not np.all(cm.sum(-1)==cm[:,:1].sum(-1)):
            raise ValueError('Target support differs across methods.')
        for i,method in enumerate(row['methods']):
            if not np.array_equal(cm[:,i].sum(0),row['metrics'][method]['confusion_matrix']):
                raise ValueError('Per-image sums differ: '+method)
        if first is not None:
            for field in ('methods','profile','config','alias_counts','cache_identity','class_names','arms'):
                if row[field]!=first[field]:
                    raise ValueError('Metadata differs across shards: '+field)
        first=first or row
        arrays.append(cm)
        keys+=shard_keys
    if len(keys)!=1449 or len(set(keys))!=1449 or set(keys)!=set(expected):
        raise ValueError('Missing or duplicated validation images.')
    all_cm=np.concatenate(arrays)
    position={key:i for i,key in enumerate(keys)}
    all_cm=all_cm[[position[key] for key in expected]]
    methods=first['methods']
    total=all_cm.sum(0)
    if not np.array_equal(total[0].sum(1),old_support):
        raise ValueError('Target support differs from the frozen previous sweep.')
    values,per_class=scores(total)
    old_value=old['metrics']['retained__g0.5']
    if abs(old_value-protocol['expected_full_miou'])>1e-9 or abs(values[0]-old_value)>.02:
        raise ValueError('Frozen Full is not closely reproduced.')
    smoke=json.loads((root/'smoke/check/prediction_check.json').read_text())
    if smoke['same_observation_frozen_vs_full_pixels']!=0 or smoke['reencoded_changed_fraction']>.001:
        raise ValueError('Smoke prediction consistency failed.')
    print('Full coverage and exact per-image sum audit passed.',flush=True)
    boot=paired_bootstrap(all_cm)
    rows=[]
    for i,method in enumerate(methods):
        row=dict(method=method,description=DESCRIPTIONS[method],mean_iou_percent=float(values[i]),
                 full_minus_arm_pp=float(values[0]-values[i]),
                 paired_95ci_pp=np.quantile(boot[:,0]-boot[:,i],[.025,.975]).tolist(),
                 class_iou_percent=per_class[i].tolist(),confusion_matrix=total[i].tolist())
        if method in protocol['cost_arms']:
            cost=json.loads((root/'cost'/method/'results.json').read_text())
            if (cost['status']!='complete' or cost['methods']!=[method]
                or [r['key'] for r in cost['images']]!=protocol['cost_keys']
                or any(len(r['milliseconds'])!=7 for r in cost['images'])
                or cost['target_masks_loaded'] or cost['g0']!=.5
                or cost['profile']!=first['profile'] or cost['config']!=first['config']
                or cost['cache_identity']!=first['cache_identity']):
                raise ValueError('Invalid independent cost record: '+method)
            row.update({key:cost[key] for key in ('median_ms','peak_allocated_mib','peak_reserved_mib')})
        rows.append(row)
    lookup={row['method']:row for row in rows}
    def effect(a,b):
        x,y=methods.index(a),methods.index(b)
        return dict(a=a,b=b,difference_pp=float(values[x]-values[y]),
            paired_95ci_pp=np.quantile(boot[:,x]-boot[:,y],[.025,.975]).tolist(),
            classes_improved=int(np.sum(per_class[x]>per_class[y])),
            classes_worsened=int(np.sum(per_class[x]<per_class[y])))
    interaction=boot[:,0]-boot[:,methods.index('NoQueryRouting')]-boot[:,methods.index('NoDonorBefore')]+boot[:,methods.index('NoDonorNoRouting')]
    effects=dict(routing_with_donor=effect('Full','NoQueryRouting'),
        routing_without_donor=effect('NoDonorBefore','NoDonorNoRouting'),
        donor_with_routing=effect('Full','NoDonorBefore'),
        donor_without_routing=effect('NoQueryRouting','NoDonorNoRouting'),
        query_correspondence=effect('Full','ShuffledQuery'),
        foreground_expansion=effect('Full','CanonicalForeground'),
        structure_vs_mean=effect('Full','EqualLogitFusion'),
        full_vs_raw24=effect('Full','Raw24CoordinateGrid'),
        full_vs_raw32=effect('Full','Raw32CoordinateGrid'),
        capacity32_vs24=effect('Raw32CoordinateGrid','Raw24CoordinateGrid'),
        donor_routing_interaction=dict(difference_pp=float(values[0]-values[methods.index('NoQueryRouting')]-values[methods.index('NoDonorBefore')]+values[methods.index('NoDonorNoRouting')]),
            paired_95ci_pp=np.quantile(interaction,[.025,.975]).tolist()))
    udd=json.loads((WORKSPACE/'research/geov2_udd5_fixed05_ablation_20261014/SUMMARY.json').read_text(encoding='utf-8'))
    summary=dict(status='complete',dataset='voc21',g0=.5,images=1449,methods=methods,
        class_names=first['class_names'],sample_keys=expected,profile=first['profile'],config=first['config'],
        alias_counts=first['alias_counts'],rows=rows,effects=effects,
        previous_full_miou=old_value,reproduction_delta_pp=float(values[0]-old_value),
        smoke=smoke,bootstrap=dict(unit='image',paired=True,replicates=10000,seed=20261014,
            note='Descriptive percentile intervals conditional on previously inspected validation data. No multiple-comparison correction; primary contrast is Full minus NoQueryRouting.'),
        udd5_reference=dict(images=40,g0=.5,effects=udd['effects']),
        all_shards_audited=True,all_per_image_sums_verified=True)
    dump(root/'SUMMARY.json',summary)
    np.savez_compressed(root/'per_image_confusions.npz',sample_keys=np.asarray(expected),
        methods=np.asarray(methods),class_names=np.asarray(first['class_names']),confusions=all_cm)
    with (root/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file)
        writer.writerow(['method','mIoU_percent','Full_minus_arm_pp','CI_low_pp','CI_high_pp',
            'median_ms','peak_allocated_MiB','peak_reserved_MiB',*first['class_names']])
        for row in rows:
            writer.writerow([row['method'],row['mean_iou_percent'],row['full_minus_arm_pp'],*row['paired_95ci_pp'],
                *[row.get(key,'') for key in ('median_ms','peak_allocated_mib','peak_reserved_mib')],*row['class_iou_percent']])
    plots(root,lookup,summary)
    report(root,lookup,summary)
    print(json.dumps({name:round(lookup[name]['mean_iou_percent'],4) for name in methods},indent=2))


def plots(root,lookup,summary):
    methods=['NoQueryRouting','NoDonorBefore','NoDonorNoRouting','ShuffledQuery',
             'Raw24CoordinateGrid','Raw32CoordinateGrid','EqualLogitFusion']
    labels=['Disable query routing (matched)','Remove DonorBefore','Remove both','Shift local queries',
            'Raw24 coordinate grid','Raw32 coordinate grid','Equal logit fusion']
    fig,ax=plt.subplots(figsize=(8.6,5),layout='constrained')
    for y,name in enumerate(methods):
        row=lookup[name]
        low,high=row['paired_95ci_pp']
        value=row['full_minus_arm_pp']
        ax.plot([low,high],[y,y],color='#687784',lw=2)
        ax.scatter([value],[y],c='#147d92' if value>=0 else '#c65b3d',s=42,zorder=3)
        ax.annotate(f'{value:+.3f}',(value,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(0,color='black',lw=.8,ls='--')
    ax.set_yticks(range(len(methods)),labels,fontsize=9)
    ax.set_ylim(len(methods)-.5,-.6)
    ax.set_xlabel('Full minus control (mIoU points)')
    ax.set_title('VOC21: all 1449 validation images, fixed g0 = 0.5\nPaired image bootstrap: 95% descriptive intervals',fontsize=11)
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):
        fig.savefig(root/('component_effects.'+ext),dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,3.3),layout='constrained')
    for y,(name,effects) in enumerate([('UDD5 (40 images)',summary['udd5_reference']['effects']),('VOC21 (1449 images)',summary['effects'])]):
        row=effects['routing_with_donor']
        low,high=row['paired_95ci_pp']
        value=row['difference_pp']
        ax.plot([low,high],[y,y],color='#687784',lw=2)
        ax.scatter([value],[y],c='#147d92' if value>=0 else '#c65b3d',s=48,zorder=3)
        ax.annotate(f'{value:+.4f} pp',(value,y),xytext=(0,11),textcoords='offset points',ha='center')
    ax.set_yticks([0,1],['UDD5 (40 images)','VOC21 (1449 images)'])
    ax.set_ylim(1.5,-.6)
    ax.axvline(0,color='black',lw=.8,ls='--')
    ax.set_xlabel('Query routing on minus off (mIoU points)')
    ax.set_title('Matched routing control; g0 = 0.5 in both datasets\nFrozen dataset-specific aliases and calibration retained',fontsize=11)
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):
        fig.savefig(root/('routing_across_datasets.'+ext),dpi=200)
    plt.close(fig)


def report(root,lookup,summary):
    full=lookup['Full']
    query=summary['effects']['routing_with_donor']
    low,high=query['paired_95ci_pp']
    conclusion=('本轮支持QuerySlot在VOC21当前配置下有净收益。' if low>0 else
                '本轮显示QuerySlot在VOC21当前配置下有净损失。' if high<0 else
                '本轮配对区间跨0，尚不足以确认QuerySlot在VOC21当前配置下有稳定净收益。')
    lines=['# VOC21：固定 g0=0.5 的 QuerySlot 复核与部件消融','',
        f"**Full：{full['mean_iou_percent']:.4f}% mIoU；关闭query路由：{lookup['NoQueryRouting']['mean_iou_percent']:.4f}%；净差 {query['difference_pp']:+.4f}pp，配对95%区间 [{low:+.4f}, {high:+.4f}]。**",'',
        conclusion,'',
        '2026-10-14。UDD5之后事先选择VOC21作为自然图像复核，完整验证集1,449张、21类；13项对照，不筛图、不在本轮改alias、beta、g0或背景校准。所有结构写回行固定g0=0.5，无自适应gate。UDD5负向路由结果一并保留。','',
        '## 1. 本轮完整模型与协议','',
        'Full = 冻结official_segmentation词表＋局部DonorBefore＋整图24槽QuerySlot（raw query，beta=0.5、温度0.1）＋Z=L+0.5H(W−L)。H=(I+AᵀA)⁻¹AᵀA。局部strength=original，确保DonorBefore实际执行。',
        '词表共119词，其中背景56词。VOC21这份词表来自官方分割词表及历史开发配置，不能称作本轮Kev筛词结果。CanonicalForeground保留同一背景56词，仅将20个前景类别各缩为类名，共76词。WideCountNormalized只修改宽视野按类聚合时的词数项。',
        '自然宽视野沿用natural_short336_cap672；局部温度0.07、宽视野tau=1、背景union_max、bias=-0.25、threshold=0.15均不变。原10/13配置coupling=0.75，本轮显式冻结为0.5，其余历史任务参数仍不是全数据集统一配置。',
        'Full宽视野没有VIP proxy self-correction或salience。LegacyDonorVIPWide只是将内部宽视野替换成旧VIP来源分支，不能当作官方完整VIP对比。骨干及文本来源的归属与模型部件应分开说明。','',
        '## 2. 全量结果','',
        '| 配置 | mIoU % | Full−该行 pp | 配对95%区间 pp | 中位耗时 ms | allocated MiB |',
        '| --- | ---: | ---: | --- | ---: | ---: |']
    for name in summary['methods']:
        row=lookup[name]
        lo,hi=row['paired_95ci_pp']
        cost=f"{row['median_ms']:.2f} | {row['peak_allocated_mib']:.2f}" if 'median_ms' in row else '— | —'
        lines.append(f"| {name} | {row['mean_iou_percent']:.4f} | {row['full_minus_arm_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {cost} |")
    lines += ['', '差值为Full减对照，负数表示对照更好。mIoU由全量合并混淆矩阵计算，不是逐图mIoU平均。区间使用10,000次配对图像bootstrap，同一次抽样对所有方法使用相同图像权重，不把像素当独立样本；它们是已查看验证集上的描述性区间，无多重比较校正。','',
        '![component effects](component_effects.png)','',
        '## 3. QuerySlot究竟是否有效','',
        '需要区分三个不同问题：正确query是否比错误query好、加入路由是否比关闭路由好、QuerySlot实现是否比Raw坐标网格好。只有第二个是本轮主要的路由净贡献检验。','',
        '| 检验 | 差值 pp | 配对95%区间 pp |', '| --- | ---: | --- |']
    names={'routing_with_donor':'Full−NoQueryRouting（主检验）',
        'routing_without_donor':'NoDonorBefore−NoDonorNoRouting',
        'query_correspondence':'Full−ShuffledQuery',
        'full_vs_raw24':'Full−Raw24CoordinateGrid',
        'full_vs_raw32':'Full−Raw32CoordinateGrid',
        'capacity32_vs24':'Raw32CoordinateGrid−Raw24CoordinateGrid',
        'donor_with_routing':'Full−NoDonorBefore',
        'donor_without_routing':'NoQueryRouting−NoDonorNoRouting',
        'donor_routing_interaction':'DonorBefore×routing的2×2交互'}
    for key,label in names.items():
        row=summary['effects'][key]
        lo,hi=row['paired_95ci_pp']
        lines.append(f"| {label} | {row['difference_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['',
        'NoQueryRouting保留同一24槽、同一局部query来源、同一整图特征采样顺序，仅beta=0使路由值不混入预测。ShuffledQuery保留局部query多重集合，按512位循环移位打破query与输出位置对应；正确比打乱好只能证明空间对应有作用，不能替代关闭路由对照。',
        'Raw24/Raw32先在整图网格计分再采样；Full与NoQueryRouting先在局部位置采样特征、归一化，再计分。采样、归一化和非线性类聚合不能随意交换，因此Full−Raw24/32包含采样顺序与容量差异，不能全部归因于query路由。','',
        f"Query路由使21类中{query['classes_improved']}类IoU提高、{query['classes_worsened']}类下降。按变化排序的前三类与后三类如下；其余类别完整列于RESULTS.csv。",'',
        '| 类别 | 路由开启−关闭的类别IoU pp |', '| --- | ---: |']
    class_differences=np.asarray(full['class_iou_percent'])-np.asarray(lookup['NoQueryRouting']['class_iou_percent'])
    order=np.argsort(-class_differences)
    for i in list(order[:3])+list(order[-3:]):
        lines.append(f"| {summary['class_names'][i]} | {class_differences[i]:+.4f} |")
    lines += ['',
        '![routing across datasets](routing_across_datasets.png)','',
        '## 4. 为什么之前说过QuerySlot有用','',
        '历史正向证据存在，但适用配置必须写清楚：','',
        '- [10/13十域开发筛查](../QUERY_SLOT_WIDE_20261013.md)：每域16图，32槽QuerySlot_Raw 53.7964%对Memory32_Raw 53.6474%，宏平均+0.1490pp，配对95%区间[-0.3161,+0.5832]；QuerySlot＋DonorBefore为53.9444%，相对Raw32 +0.2970pp，区间[-0.1740,+0.7757]。后一个差值还混合DonorBefore变化。',
        '- [历史全量VOC20消融](../geov2_full_ablation_20261014/REPORT.md)：QuerySlot_Raw 92.1652%对Raw24Slots 90.9828%，+1.1824pp；旧adaptive配置92.2111%对NoRoutingAdaptive 91.2746%，+0.9365pp。但局部strength=1、DonorBefore不执行，且包含自适应写回，不等于本轮固定g0、strength=original配置。',
        '- [UDD5固定0.5全量](../geov2_udd5_fixed05_ablation_20261014/REPORT_ZH.md)：Full 55.2574%对关闭路由55.5846%，−0.3273pp；正确query显著优于打乱query，但没有证明加入路由优于不加入。','',
        '因此“以前有正向证据”成立，“每个域、每套参数都必然获益”不成立。本轮不以换数据集为由删除UDD5结果，也不把VOC20旧配置的收益搬到当前VOC21。','',
        '## 5. Alias、写回与效率的独立解释','']
    for key,label in [('foreground_expansion','保留背景不变的前景扩词'),('structure_vs_mean','固定结构写回相对同观测等权融合')]:
        row=summary['effects'][key]
        lo,hi=row['paired_95ci_pp']
        lines.append(f"{label}：{row['difference_pp']:+.4f}pp，配对95%区间[{lo:+.4f}, {hi:+.4f}]。")
    raw=lookup['Raw32CoordinateGrid']
    lines += ['',f"Full相对Raw32坐标读出增加{full['median_ms']-raw['median_ms']:.2f}ms（{(full['median_ms']/raw['median_ms']-1)*100:.2f}%）和{full['peak_allocated_mib']-raw['peak_allocated_mib']:.2f}MiB allocated；相应mIoU提高{full['mean_iou_percent']-raw['mean_iou_percent']:.4f}pp，但该精度差还包含采样顺序变化。"]
    lines += ['',
        '不能把Alias收益自动归为Kev筛词收益：本轮没有为VOC21定义与当前词表同来源、同模板、同背景的未筛候选池。词数归一化控制测试聚合规则，不等价于词表筛选。',
        '这两份全量消融可分别支撑机制，但不能宣称每个部件在每个域都提高：UDD5固定写回胜过等权融合，而VOC21主要展示query路由的收益。VOC21当前固定0.5写回低于等权融合，说明统一步长在自然图像上仍有取舍。上一轮同配置g0扫描中，VOC21的g0=0.75为74.6881%、g0=1为75.2094%；这些只作已查看参数敏感性的历史参考，不替换本轮预先固定0.5的消融结果，也不算独立泛化证据。',
        '耗时为A800上7张预先固定图、每张2次预热＋7次同步计时的合并中位数。每个arm独立进程，在精度作业结束后顺序运行；边界为CPU float RGB输入到原尺寸CPU uint8类ID，排除解码、初始化、离线文本编码和mask。显存为PyTorch峰值allocated，reserved另见CSV/JSON，不能与nvidia-smi驻留值混用。',
        'NoQueryRouting仍计算未混入的路由中间量，其成本用于控制当前执行路径，不代表彻底删除query计算后的最优速度。当前脚本不以全量精度评测墙钟时间替代单图延迟。','',
        '## 6. 复现、来源和核验','',
        f"Full相对上一轮完整g0扫描的{summary['previous_full_miou']:.6f}%复现差为{summary['reproduction_delta_pp']:+.6f}pp。所有1,449张唯一覆盖、7个分片的元数据和逐图混淆矩阵之和均已核验，并与旧全量标签支持数一致。",
        f"同观测实验接口与FrozenGeometry.predict差异为0像素；重新编码首图有{summary['smoke']['reencoded_vs_full_pixels']}个像素变化（{summary['smoke']['reencoded_changed_fraction']:.6%}），记录CUDA槽归约顺序产生的浮点波动。",'',
        '历史Alias、温度和背景参数用过带标签开发集，且此前查看过全量验证；本轮推理不读取标签，标签只在全部预测完成后用于评分。冻结复核不等于独立测试，不能声明无标签开发选择。',
        '入口：[通用评测](../../DINOtool/scripts/eval_geov2_fixed05_components.py)、[审计统计和作图](../../tools/report_voc21_fixed05_ablation.py)。配置和实际词表位于[configs/voc21.yaml](configs/voc21.yaml)、[vocabularies/voc21.json](vocabularies/voc21.json)。YAML保留历史coupling=0.75，本轮覆盖值以protocol.json与结果profile中的0.5为准。',
        '在已配置DINOtool及固定上游依赖的环境中，将protocol.json中的数据、checkpoint及文本cache路径换成本机路径，设置PYTHONPATH包含DINOtool、DINOtool/scripts及DINO_Soars依赖。运行eval_geov2_fixed05_components.py --experiment 本目录 --mode full；成本使用--mode cost --arm Full。实际七分片及顺序成本命令见full.sh、cost.sh。输出已存在时脚本拒绝覆盖。',
        '证据：[SUMMARY.json](SUMMARY.json)、[RESULTS.csv](RESULTS.csv)、[逐图混淆矩阵](per_image_confusions.npz)、[冻结协议](protocol.json)、full/及cost/；历史对照来源保留在reference/。权重、图像、mask、二进制文本缓存与SSH配置不发布。','']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
