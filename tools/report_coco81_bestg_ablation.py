"""Audit component-wise gain search; separate matched and retuned evidence."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[1]
NAME='geov2_coco81_bestg_ablation_20261014'


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def score(tp,predicted,target):
    union=predicted+target-tp
    iou=np.divide(tp,union,out=np.full(tp.shape,np.nan,dtype=np.float64),where=union>0)*100
    return np.nanmean(iou,axis=-1),iou


def grid_cells(protocol):
    grid=[dict(key=m+'__g'+format(g,'g'),method=m,g0=g) for m in protocol['component_arms'] for g in protocol['gains']]
    return grid+[dict(key=m,method=m,g0=None) for m in protocol['endpoint_arms']]


def bootstrap(tp,predicted,target,indices):
    t=tp[:,indices]
    union=predicted[:,indices]+target[:,None]-t
    n,m,c=t.shape
    stats=np.stack((t,union),axis=-1).reshape(n,-1).astype(np.float64)
    rng=np.random.default_rng(20261014)
    boot=np.empty((5000,m))
    for start in range(0,len(boot),50):
        count=min(50,len(boot)-start)
        draws=rng.integers(0,n,size=(count,n))
        weights=np.stack([np.bincount(draw,minlength=n) for draw in draws]).astype(np.float64)
        total=(weights@stats).reshape(count,m,c,2)
        iou=np.divide(total[...,0],total[...,1],out=np.full((count,m,c),np.nan),where=total[...,1]>0)
        boot[start:start+count]=np.nanmean(iou,axis=-1)*100
    return boot


def cross_dataset(root):
    """Post-hoc equal-domain summary of every completed matched .5 study."""
    methods=['Full','NoQueryRouting','NoDonorBefore','CanonicalForeground','EqualLogitFusion']
    rng=np.random.default_rng(20261014)
    observed=[]
    resampled=[]
    sources=[]
    for dataset,folder in [('udd5','geov2_udd5_fixed05_ablation_20261014'),
                           ('voc21','geov2_voc21_fixed05_ablation_20261014'),
                           ('coco_object81',NAME)]:
        path=WORKSPACE/'research'/folder
        if dataset=='coco_object81':
            with np.load(path/'per_image_statistics.npz',allow_pickle=False) as z:
                keys=z['methods'].tolist()
                indices=[keys.index(m if m=='EqualLogitFusion' else m+'__g0.5') for m in methods]
                tp=z['tp'][:,indices].copy()
                predicted=z['predicted'][:,indices].copy()
                target=z['target'].copy()
        else:
            with np.load(path/'per_image_confusions.npz',allow_pickle=False) as z:
                keys=z['methods'].tolist()
                cm=z['confusions'][:,[keys.index(m) for m in methods]]
            tp=np.diagonal(cm,axis1=-2,axis2=-1).copy()
            predicted=cm.sum(-2)
            target=cm[:,0].sum(-1)
        values,_=score(tp.sum(0),predicted.sum(0),target.sum(0))
        n,m,c=tp.shape
        stats=np.stack((tp,predicted+target[:,None]-tp),axis=-1).reshape(n,-1).astype(np.float64)
        boot=np.empty((5000,m))
        for start in range(0,len(boot),50):
            count=min(50,len(boot)-start)
            draws=rng.integers(0,n,size=(count,n))
            weights=np.stack([np.bincount(draw,minlength=n) for draw in draws]).astype(np.float64)
            total=(weights@stats).reshape(count,m,c,2)
            iou=np.divide(total[...,0],total[...,1],out=np.full((count,m,c),np.nan),where=total[...,1]>0)
            boot[start:start+count]=np.nanmean(iou,axis=-1)*100
        observed.append(values)
        resampled.append(boot)
        sources.append(dict(dataset=dataset,images=n,g0=.5,folder=folder,mious=dict(zip(methods,values.tolist()))))
    values=np.mean(observed,axis=0)
    boot=np.mean(resampled,axis=0)
    rows=[]
    for i,method in enumerate(methods):
        rows.append(dict(method=method,equal_domain_mean_miou_percent=float(values[i]),
            full_minus_arm_pp=float(values[0]-values[i]),
            paired_95ci_pp=np.quantile(boot[:,0]-boot[:,i],[.025,.975]).tolist()))
    result=dict(status='complete',post_hoc=True,g0=.5,domains=3,sources=sources,rows=rows,
        bootstrap=dict(replicates=5000,seed=20261014,unit='image within each dataset',paired=True,
            aggregation='Recompute pooled mIoU within each dataset; average datasets equally. Resample each dataset independently; use identical image weights across methods within each dataset.'),
        limitations='Post-hoc summary of all three completed matched .5 ablations, including negative individual results; not a domain sample, independent test, each-domain claim, or per-dataset-best-g0 table. VOC21 remains the earlier fixed-.5 study although its historical best gain is 1. Historical label selection remains.' )
    dump(root/'CROSS_DATASET_COMPONENTS.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def precision(root):
    p=json.loads((root/'protocol.json').read_text())
    grid=grid_cells(p)
    expected=json.loads((WORKSPACE/'research/geov2_g0_sweep_20261014/expected_keys.json').read_text())['coco_object81']
    old=json.loads((WORKSPACE/'research/geov2_g0_sweep_20261014/coco_object81_audited.json').read_text())
    arrays,keys,records=[],[],[]
    first=None
    for shard in range(p['shards']):
        folder=root/'full'/f's{shard}'
        row=json.loads((folder/'results.json').read_text())
        with np.load(folder/'statistics.npz',allow_pickle=False) as z:
            a={k:z[k].copy() for k in ('tp','predicted','target')}
            local_keys=z['sample_keys'].tolist()
        if (row['status']!='complete' or local_keys!=expected[shard::p['shards']]
            or row['sample_keys']!=local_keys or row['cells']!=grid
            or a['tp'].shape!=(len(local_keys),len(grid),81) or a['predicted'].shape!=a['tp'].shape
            or a['target'].shape!=(len(local_keys),81)
            or row['processed_images']!=len(local_keys) or row['global_sample_count']!=5000
            or row['shard_index']!=shard or row['num_shards']!=p['shards']
            or row['adaptive_gain_enabled'] or not row['masks_used_only_for_scoring']
            or row['profile']['strength']!='original' or row['gains']!=p['gains']):
            raise ValueError('Invalid shard: '+str(folder))
        if (np.any(a['tp']>a['predicted']) or np.any(a['tp']>a['target'][:,None])
            or not np.all(a['predicted'].sum(-1)==a['target'].sum(-1)[:,None])):
            raise ValueError('Invalid image-level supports.')
        for i,cell in enumerate(grid):
            cm=np.asarray(row['metrics'][cell['key']]['confusion_matrix'])
            if (not np.array_equal(a['tp'][:,i].sum(0),cm.diagonal())
                or not np.array_equal(a['predicted'][:,i].sum(0),cm.sum(0))
                or not np.array_equal(a['target'].sum(0),cm.sum(1))):
                raise ValueError('Image sums do not equal recorded confusion: '+cell['key'])
        if first is not None:
            for field in ('cells','methods','profile','config','alias_counts','cache_identity','class_names','gains'):
                if row[field]!=first[field]:
                    raise ValueError('Shard metadata differs: '+field)
        first=first or row
        arrays.append(a)
        keys+=local_keys
        records.append(row)
    if len(keys)!=5000 or len(set(keys))!=5000 or set(keys)!=set(expected):
        raise ValueError('Full coverage differs.')
    lookup={k:i for i,k in enumerate(keys)}
    order=[lookup[k] for k in expected]
    all_stats={k:np.concatenate([a[k] for a in arrays])[order] for k in ('tp','predicted','target')}
    old_support=np.zeros(81,dtype=np.int64)
    for path in (WORKSPACE/'research/geov2_g0_sweep_20261014/full').glob('coco_object81_s*/results.json'):
        row=json.loads(path.read_text())
        old_support+=np.asarray(row['metrics']['retained__g0.5']['confusion_matrix']).sum(1)
    if not np.array_equal(all_stats['target'].sum(0),old_support):
        raise ValueError('Previous full target support differs.')
    tp=all_stats['tp'].sum(0)
    predicted=all_stats['predicted'].sum(0)
    target=all_stats['target'].sum(0)
    values,class_iou=score(tp,predicted,target)
    indices={cell['key']:i for i,cell in enumerate(grid)}
    reference=values[indices['Full__g'+format(p['reference_g0'],'g')]]
    if abs(reference-p['expected_full_miou'])>.02:
        raise ValueError('Frozen full reference is not reproduced.')
    if abs(values[indices['UniformBlend__g1']]-values[indices['EqualLogitFusion']])>1e-7:
        raise ValueError('Uniform blend alpha=.5 differs from equal fusion.')
    smoke=json.loads((root/'smoke/check/prediction_check.json').read_text())
    if smoke['same_observation_frozen_vs_full_pixels'] or smoke['uniform_g1_vs_equal_pixels']:
        raise ValueError('Smoke parity failed.')
    best={}
    for method in p['component_arms']:
        candidates=[indices[method+'__g'+format(g,'g')] for g in p['gains']]
        best[method]=max(candidates,key=lambda i:values[i])
    full=best['Full']
    gain=grid[full]['g0']
    matched={m:indices[m+'__g'+format(gain,'g')] for m in p['component_arms']}
    matched.update({m:indices[m] for m in p['endpoint_arms']})
    selected=sorted(set([*best.values(),*matched.values()]))
    print('Full coverage, image sums, previous supports and parity audited. Computing paired intervals.',flush=True)
    boot=bootstrap(all_stats['tp'],all_stats['predicted'],all_stats['target'],selected)
    boot_indices={i:j for j,i in enumerate(selected)}
    def comparison(index):
        difference=boot[:,boot_indices[full]]-boot[:,boot_indices[index]]
        return dict(key=grid[index]['key'],method=grid[index]['method'],g0=grid[index]['g0'],
            mean_iou_percent=float(values[index]),full_minus_arm_pp=float(values[full]-values[index]),
            paired_95ci_pp=np.quantile(difference,[.025,.975]).tolist(),
            classes_improved=int(np.sum(class_iou[full]>class_iou[index])),
            class_iou_percent=class_iou[index].tolist())
    matched_rows=[comparison(i) for i in matched.values()]
    retuned_rows=[comparison(i) for i in best.values()]
    by_matched={r['method']:r for r in matched_rows}
    by_retuned={r['method']:r for r in retuned_rows}
    def supported(row):
        return row['full_minus_arm_pp']>0 and row['paired_95ci_pp'][0]>0
    primary=dict(query_routing=by_matched['NoQueryRouting'],donor_before=by_matched['NoDonorBefore'],
        foreground_aliases=by_matched['CanonicalForeground'],writer_vs_equal=by_matched['EqualLogitFusion'])
    strict=dict(query_routing=by_retuned['NoQueryRouting'],donor_before=by_retuned['NoDonorBefore'],
        foreground_aliases=by_retuned['CanonicalForeground'],writer_vs_retuned_uniform=by_retuned['UniformBlend'])
    summary=dict(status='precision_complete',dataset='coco_object81',images=5000,gains=p['gains'],
        full_best_g0=gain,full_miou_percent=float(values[full]),matched_rows=matched_rows,retuned_rows=retuned_rows,
        primary_effects=primary,retuned_effects=strict,
        all_four_supported_matched=all(supported(r) for r in primary.values()),
        all_four_supported_after_gain_retuning=all(supported(r) for r in strict.values()),
        grid_rows=[dict(**cell,mean_iou_percent=float(values[i])) for i,cell in enumerate(grid)],
        profile=first['profile'],config=first['config'],alias_counts=first['alias_counts'],
        cache_identity=first['cache_identity'],class_names=first['class_names'],sample_keys=expected,
        reproduction_delta_pp=float(reference-p['expected_full_miou']),smoke=smoke,
        bootstrap=dict(unit='image',paired=True,replicates=5000,seed=20261014,
            note='Conditional descriptive intervals at gains chosen on this already-inspected validation set. No nested gain selection or multiple-comparison correction; not independent test evidence.'),
        all_shards_audited=True,all_per_image_sums_verified=True)
    dump(root/'SUMMARY.json',summary)
    dump(root/'selection.json',dict(full_best_g0=gain,full_miou_percent=float(values[full]),
        selection_rule='Maximize pooled full-validation mIoU over declared gain grid; ties use smaller gain.',
        labeled_validation_used_for_selection=True,control_best_g0={m:grid[i]['g0'] for m,i in best.items()}))
    np.savez_compressed(root/'per_image_statistics.npz',sample_keys=np.asarray(expected),
        methods=np.asarray([c['key'] for c in grid]),class_names=np.asarray(first['class_names']),**all_stats)
    with (root/'GAIN_SWEEP.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['method',*[format(g,'g') for g in p['gains']],'best_g0','best_mIoU_percent'])
        for method in p['component_arms']:
            writer.writerow([method,*[values[indices[method+'__g'+format(g,'g')]] for g in p['gains']],grid[best[method]]['g0'],values[best[method]]])
    print(json.dumps(dict(full_best_g0=gain,matched={r['method']:r['mean_iou_percent'] for r in matched_rows},
        all_four_supported_matched=summary['all_four_supported_matched'],
        all_four_supported_after_gain_retuning=summary['all_four_supported_after_gain_retuning']),indent=2))


def finalize(root):
    p=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'SUMMARY.json').read_text(encoding='utf-8'))
    if not summary['all_shards_audited'] or not summary['all_per_image_sums_verified']:
        raise ValueError('Precision must be audited first.')
    costs={}
    for method in p['cost_arms']:
        row=json.loads((root/'cost'/method/'results.json').read_text())
        if (row['status']!='complete' or row['arm']!=method or row['g0']!=summary['full_best_g0']
            or row['target_masks_loaded'] or row['config']!=summary['config']
            or row['cache_identity']!=summary['cache_identity']
            or [r['key'] for r in row['images']]!=p['cost_keys']
            or any(len(r['milliseconds'])!=7 for r in row['images'])):
            raise ValueError('Independent cost mismatch: '+method)
        costs[method]={k:row[k] for k in ('median_ms','peak_allocated_mib','peak_reserved_mib')}
    summary.update(status='complete',costs=costs)
    dump(root/'SUMMARY.json',summary)
    with (root/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['comparison','method','g0','mIoU_percent','Full_minus_arm_pp','CI_low','CI_high','median_ms','peak_allocated_MiB'])
        for kind in ('matched','retuned'):
            for row in summary[kind+'_rows']:
                cost=costs.get(row['method'],{}) if row['g0']==summary['full_best_g0'] else {}
                writer.writerow([kind,row['method'],row['g0'],row['mean_iou_percent'],row['full_minus_arm_pp'],
                    *row['paired_95ci_pp'],cost.get('median_ms',''),cost.get('peak_allocated_mib','')])
    plots(root,summary)
    report(root,summary)
    print(json.dumps(dict(full_best_g0=summary['full_best_g0'],full_miou_percent=summary['full_miou_percent'],
        full_cost=costs['Full'],all_four_supported_matched=summary['all_four_supported_matched'],
        all_four_supported_after_gain_retuning=summary['all_four_supported_after_gain_retuning']),indent=2))


def plots(root,s):
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    methods=['NoQueryRouting','NoDonorBefore','CanonicalForeground','UniformBlend']
    labels=['Disable routing','Remove DonorBefore','Canonical foreground','Uniform blend']
    for ax,kind in zip(axes,('matched','retuned')):
        rows={r['method']:r for r in s[kind+'_rows']}
        active=[m for m in methods if m in rows]
        for y,method in enumerate(active):
            row=rows[method]
            value=row['full_minus_arm_pp']
            low,high=row['paired_95ci_pp']
            ax.plot([low,high],[y,y],color='#687784',lw=2)
            ax.scatter([value],[y],color='#147d92' if value>=0 else '#c65b3d',s=42,zorder=3)
            ax.annotate(f'{value:+.3f}',(value,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.axvline(0,color='black',ls='--',lw=.8)
        ax.set_yticks(range(len(active)),[labels[methods.index(m)] for m in active],fontsize=9)
        ax.set_ylim(len(active)-.5,-.6)
        ax.set_xlabel('Full minus control (mIoU points)')
        ax.set_title('All controls at Full-selected g0' if kind=='matched' else 'Each control selects its own best g0',fontsize=10)
        ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"COCO-Object81: all 5000 validation images, Full-selected g0 = {s['full_best_g0']:g}\n95% paired image intervals conditional on selected gains",fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(root/('component_effects.'+ext),dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.5),layout='constrained')
    for method in ('Full','NoQueryRouting','NoDonorBefore','CanonicalForeground','UniformBlend'):
        rows=[r for r in s['grid_rows'] if r['method']==method]
        ax.plot([r['g0'] for r in rows],[r['mean_iou_percent'] for r in rows],marker='o',ms=4,label=method)
    ax.set_xlabel('g0 (uniform blend uses alpha = g0 / 2)')
    ax.set_ylabel('Pooled mIoU (%)')
    ax.set_title('COCO-Object81: identical declared gain search')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):
        fig.savefig(root/('gain_curves.'+ext),dpi=200)
    plt.close(fig)


def report(root,s):
    full=s['full_miou_percent']
    cost=s['costs']['Full']
    point_positive=all(r['full_minus_arm_pp']>0 for r in s['primary_effects'].values())
    verdict='四个主部件在同g0的消融中均有正向配对区间。' if s['all_four_supported_matched'] else ('四个主部件的点估计均为正，但部分配对区间跨0，尚不能确认每个部件都有稳定净收益。' if point_positive else '同g0消融未能支持所有主部件都有效。')
    strict='对照各自重选g0后，四个主部件仍均有正向配对区间。' if s['all_four_supported_after_gain_retuning'] else '允许各对照独立重选g0后，不能声称每个部件都优于其替代方案。'
    alias_count=sum(s['alias_counts']['retained'])
    canonical_count=sum(s['alias_counts']['canonical_fg'])
    lines=['# COCO-Object81：最佳g0与各部件全量消融','',
        f"**完整模型最佳g0={s['full_best_g0']:g}，mIoU={full:.4f}%，A800中位耗时{cost['median_ms']:.2f}ms，峰值allocated {cost['peak_allocated_mib']:.2f}MiB。**",'',
        verdict+' '+strict,'',
        '完整验证集5,000张、81类；12种耦合配置各扫描g0∈{0,0.25,0.5,0.75,1,1.5,2,3}，再加等权融合与两个端点，共99项配置。未改变历史alias、beta、温度、背景bias或阈值。本轮选择COCO以使用独立于VOC的图像来源及多物体场景，不对新数据集做轮番筛选后只报告获胜者。','',
        'g0由全量验证标签选出，是开发分析，不是测试集上无标签自动选择。推理读取冻结g0，不读取标签。区间为5,000次配对图像bootstrap，条件于已经选好的g0，没有嵌套选择或多重比较校正；不能将其视为独立泛化证据。','',
        '## 1. 完整模型及对照定义','',
        'Full = 冻结official_imagenet词表＋局部DonorBefore（strength=original）＋raw-query QuerySlot24（beta=0.5，温度0.1）＋Z=L+g0H(W−L)，H=(I+AᵀA)⁻¹AᵀA。两路共享类别映射；QuerySlot以局部raw token从整图槽库读上下文，H限制修正写回。',
        f"当前词表共{alias_count}词，背景{s['alias_counts']['retained'][0]}词；前景仅类名的控制共{canonical_count}词，背景完全保留。消融删除的是前景扩展词，保留同一模板、类别映射与背景校准。",
        'NoQueryRouting仅beta=0，保留相同槽数及先采样特征再计分的顺序；NoDonorBefore只关闭局部响应顺序干预；CanonicalForeground只删除前景扩展词，背景词表不变。ShuffledQuery将局部query循环移位512位，保留token多重集合。',
        'LocalOnly与WideOnly只改变同观测读出。WideOnly中的W仍受局部query条件化，不能当成删除局部编码器后的独立整图模型。',
        'UniformBlend将H换成0.5I，对应Z=(1−alpha)L+alphaW，alpha=g0/2；其8档包含等权融合与alpha=1.5的超松弛。DiagonalH只保留H的对角线，检验跨位置关系的作用。它们共享Full的L/W，并各自扫描同一g0网格。Raw24/32还改变采样与计分顺序，不能把其差值全归于query路由。LegacyDonorVIPWide只替换内部宽视野分支，不是官方完整VIP基线。','',
        '## 2. 在完整模型最佳g0下删除部件','',
        '| 配置 | g0 | mIoU % | Full−对照 pp | 配对95%区间 pp |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in s['matched_rows']:
        lo,hi=row['paired_95ci_pp']
        gain='—' if row['g0'] is None else format(row['g0'],'g')
        lines.append(f"| {row['method']} | {gain} | {row['mean_iou_percent']:.4f} | {row['full_minus_arm_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '## 3. 允许删除部件后的对照也重选最佳g0','',
        '这张表检查收益是否主要来自只给完整模型选好步长。各行都按同一8档网格最大化整个验证集累计混淆矩阵的mIoU；不是逐图选参数。负差或区间跨0的项均保留。','',
        '| 配置 | 各自最佳g0 | 最佳mIoU % | Full最佳−对照最佳 pp | 配对95%区间 pp |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in s['retuned_rows']:
        lo,hi=row['paired_95ci_pp']
        lines.append(f"| {row['method']} | {row['g0']:g} | {row['mean_iou_percent']:.4f} | {row['full_minus_arm_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '![component effects](component_effects.png)','',
        '![gain curves](gain_curves.png)','',
        '## 4. 结果如何写进论文','',
        verdict,strict,
        '可以报告对应配置、数据集与选择协议下的部件贡献。不能因为希望得到完整故事，就将重选g0后更好的对照、UDD5的路由结果或VOC21固定0.5低于等权融合的结果删掉。单域正向消融也不意味着该部件对所有域必需。',
        '本轮Alias控制检验保留背景不变的前景扩词，不单独证明Kev筛词有效。当前official_imagenet bank的来源和实际词表一起公开；如果没有同来源、同模板、同背景的未筛候选池，不把扩词收益称为筛词收益。','',
    ]
    uniform=next(r for r in s['retuned_rows'] if r['method']=='UniformBlend')
    lo,hi=uniform['paired_95ci_pp']
    lines += [f"结构本身的净证据应看最佳普通加权融合对照：Full比其高{uniform['full_minus_arm_pp']:.4f}pp，区间[{lo:+.4f}, {hi:+.4f}]，其最佳alpha={uniform['g0']/2:g}。Full相对等权融合的大差值还包含融合步长的影响，不能全部归于结构滤波。",'']
    cross_path=root/'CROSS_DATASET_COMPONENTS.json'
    if cross_path.exists():
        cross=json.loads(cross_path.read_text(encoding='utf-8'))
        lines += ['### 已完成三域的事后平均分析','',
            '为检查部件的平均贡献，纳入目前全部三份完整匹配消融：UDD5（40图）、VOC21（1,449图）、COCO（5,000图），不删除任何域或任何负向单域结果。三份都使用已完成的g0=0.5记录，按数据集等权平均，独立在域内重采样图像且各方法保持配对。',
            '这是看到COCO结果后的事后分析；不是预注册主检验、独立测试、逐域最佳g0表或每域均改善的证明。VOC21沿用此前固定0.5消融，未改成其历史最优1。区间只描述这三份固定记录的图像不确定性，不包括数据集抽样及历史参数选择的不确定性。','',
            '| 删除或替换部件 | Full−对照的等域平均 pp | 配对95%区间 pp |',
            '| --- | ---: | --- |']
        for row in cross['rows'][1:]:
            low,high=row['paired_95ci_pp']
            lines.append(f"| {row['method']} | {row['full_minus_arm_pp']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        lines += ['', '三域平均四项区间均高于0，支持这三份记录上的平均贡献为正；COCO单域QuerySlot与DonorBefore区间跨0的结论仍保留。原始来源与重采样定义见[CROSS_DATASET_COMPONENTS.json](CROSS_DATASET_COMPONENTS.json)。','']
    lines += ['## 5. 独立成本','',
        '| 配置 | g0 | 中位耗时 ms | 峰值allocated MiB | 峰值reserved MiB |',
        '| --- | ---: | ---: | ---: | ---: |']
    for method,c in s['costs'].items():
        lines.append(f"| {method} | {s['full_best_g0']:g} | {c['median_ms']:.2f} | {c['peak_allocated_mib']:.2f} | {c['peak_reserved_mib']:.2f} |")
    lines += ['',
        '每项独立进程，在全量精度作业结束后顺序运行。7张预先固定图，每图2次预热和7次同步重复，表内是49次的合并中位数。边界为CPU float RGB到原尺寸CPU uint8类ID，排除图像解码、初始化、离线文本编码和mask。allocated与reserved都不是nvidia-smi驻留显存；NoQueryRouting仍计算不混入输出的路由中间量，成本不是彻底移除该计算的优化下限。99配置扫描的墙钟耗时不代表单次推理延迟。','',
        '## 6. 核验和复现','',
        f"全量5,000张唯一覆盖，7分片配置一致，每图TP/predicted/target之和与每项完整混淆矩阵逐一一致，标签支持数与旧全量扫描相同。Full在参考g0=0.5下的旧结果复现差为{s['reproduction_delta_pp']:+.6f}pp；接口同观测差0像素，UniformBlend(g0=1)与等权融合差0像素。",
        '配置与词表位于configs/、vocabularies/；[冻结协议](protocol.json)声明搜索空间，[selection.json](selection.json)记录标签选择及各行选出的g0。历史alias和背景校准使用过开发标签，且验证集此前已查看；本轮不是独立测试。',
        '代码：[评测](../../DINOtool/scripts/eval_geov2_component_gain_sweep.py)、[审计、配对统计与作图](../../tools/report_coco81_bestg_ablation.py)。在配置好DINOtool及固定上游依赖的环境中替换protocol内路径，设置PYTHONPATH包含DINOtool、DINOtool/scripts、固定DINO_Soars依赖，按full.sh运行分片；按cost.sh测量独立成本。统计脚本先--stage precision，--stage aggregate生成事后三域汇总，最后--stage finalize。已有评测输出拒绝覆盖。',
        '证据：[SUMMARY.json](SUMMARY.json)、[全步长表](GAIN_SWEEP.csv)、[消融与成本CSV](RESULTS.csv)、[完整逐图统计](per_image_statistics.npz)、full/、cost/及实际运行source/。只发布统计，不发布图像、mask、权重、二进制文本cache或SSH管理配置。','']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('precision','aggregate','finalize'),required=True)
    args=p.parse_args()
    root=WORKSPACE/'research'/NAME
    {'precision':precision,'aggregate':cross_dataset,'finalize':finalize}[args.stage](root)
