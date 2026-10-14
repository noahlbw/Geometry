"""Audit fixed-gain full splits and compare global, held-out and alias choices.

All dataset mIoUs use summed pixel counts. Equal-dataset means deliberately
give each of the ten declared protocols the same weight.
"""
import argparse
import json
from pathlib import Path

import numpy as np

LABELS = dict(vdd='VDD', potsdam='Potsdam', udd5='UDD5', oem='OEM',
              vaihingen='Vaihingen', context60='PC60', ade150='ADE150',
              voc20='VOC20', voc21='VOC21', coco_object81='COCO-Object81')


def miou(tp, predicted, target):
    union = predicted+target-tp
    return float(np.nanmean(np.divide(tp, union, out=np.full(tp.shape, np.nan), where=union>0))*100)


def audit(root, protocol, expected, ds):
    record = protocol['datasets'][ds]
    total = record['total_images']
    shards = 8 if total>=5000 else 4 if total>=2000 else 2 if total>=1000 else 1
    rows, arrays = [], []
    for shard in range(shards):
        folder = root/'full'/f'{ds}_s{shard}'
        row = json.loads((folder/'results.json').read_text())
        keys = expected[ds][shard::shards]
        if (row['status']!='complete' or row['dataset']!=ds
                or row['shard_index']!=shard or row['num_shards']!=shards
                or row['global_sample_count']!=total or row['sample_keys']!=keys
                or row['processed_images']!=len(keys) or not row['both_modules_active']
                or not row['masks_used_only_for_scoring']):
            raise ValueError('Incomplete or mismatched validation shard: '+str(folder))
        if row['profile']!=record['frozen_profile'] or row['profile']['strength']!='original':
            raise ValueError('Frozen profile mismatch: '+ds)
        if row['gains']!=protocol['gains'] or row['alias_modes']!=protocol['alias_modes']:
            raise ValueError('Scan specification mismatch: '+ds)
        with np.load(folder/'statistics.npz', allow_pickle=False) as archive:
            a = {key:archive[key] for key in archive.files}
        if a['sample_keys'].tolist()!=keys:
            raise ValueError('Statistics/image key mismatch: '+ds)
        if len(row['methods'])!=30 or a['tp'].shape!=(len(keys), 30, len(row['class_names'])):
            raise ValueError('Invalid statistics axes: '+ds)
        if a['target'].shape!=(len(keys), len(row['class_names'])):
            raise ValueError('Invalid target axis: '+ds)
        for index, method in enumerate(row['methods']):
            cm = np.asarray(row['metrics'][method]['confusion_matrix'], dtype=np.int64)
            if not (np.array_equal(a['tp'][:,index].sum(0), cm.diagonal())
                    and np.array_equal(a['predicted'][:,index].sum(0), cm.sum(0))
                    and np.array_equal(a['target'].sum(0), cm.sum(1))):
                raise ValueError('Statistics/matrix support mismatch: '+ds+' '+method)
            value = miou(cm.diagonal(), cm.sum(0), cm.sum(1))
            if abs(value-row['metrics'][method]['mean_iou_percent'])>1e-9:
                raise ValueError('Recorded mIoU mismatch: '+ds+' '+method)
        for field in ('methods', 'config', 'cache_identity', 'class_names', 'alias_counts', 'residual_concepts'):
            if rows and row[field]!=rows[0][field]:
                raise ValueError('Inconsistent shard metadata: '+ds+' '+field)
        rows.append(row)
        arrays.append(a)
    keys = [key for row in rows for key in row['sample_keys']]
    if len(keys)!=total or len(set(keys))!=total or set(keys)!=set(expected[ds]):
        raise ValueError('Full split coverage mismatch: '+ds)
    methods = rows[0]['methods']
    lookup = {key:i for i,key in enumerate(keys)}
    order = np.asarray([lookup[key] for key in expected[ds]])
    tp = np.concatenate([a['tp'] for a in arrays])[order]
    pred = np.concatenate([a['predicted'] for a in arrays])[order]
    target = np.concatenate([a['target'] for a in arrays])[order]
    zero = methods.index('retained__g0')
    other = methods.index('count_normalized__g0')
    if not np.array_equal(tp[:,zero],tp[:,other]) or not np.array_equal(pred[:,zero],pred[:,other]):
        raise ValueError('Count normalization changed the local endpoint: '+ds)
    if ds=='context60' and rows[0]['residual_concepts']!=401:
        raise ValueError('PC60 residual vocabulary changed.')
    metrics = {m:miou(tp[:,i].sum(0), pred[:,i].sum(0), target.sum(0)) for i,m in enumerate(methods)}
    diagnostics = {mode:{field:sum(row['diagnostics'][mode][field] for row in rows)
        for field in rows[0]['diagnostics'][mode]} for mode in protocol['alias_modes']}
    for values in diagnostics.values():
        values['correction_to_displacement_energy'] = values['correction_energy']/max(values['displacement_energy'],1e-30)
    counts = np.asarray(rows[0]['alias_counts'], dtype=float)
    return dict(dataset=ds, images=total, methods=methods, metrics=metrics,
        class_names=rows[0]['class_names'], alias_counts=counts.tolist(),
        alias_mean=float(counts.mean()), alias_count_cv=float(counts.std()/counts.mean()),
        residual_concepts=rows[0]['residual_concepts'], diagnostics=diagnostics,
        audit=dict(exact_authoritative_split_order=True, all_shards_complete=True,
            per_image_sufficient_statistics_verified=True, identical_target_support=True,
            count_normalized_local_endpoint_identical=True)), (tp,pred,target)


def bootstrap_difference(arrays, results, expected, reference, control, repetitions):
    """Conditional paired dataset-mIoU intervals, preserving shared VOC images."""
    draws = []
    dataset_draws = {}
    for ds,(tp,pred,target) in arrays.items():
        methods = results[ds]['methods']
        indices = [methods.index(reference),methods.index(control)]
        sums = tp[:,indices].astype(float)
        unions = pred[:,indices].astype(float)+target[:,None].astype(float)-sums
        n = len(tp)
        # Reinitialize the VOC generator identically for the shared split.
        seed = 20261014 if ds in ('voc20','voc21') else 20261014+list(results).index(ds)+1
        rng = np.random.default_rng(seed)
        if ds=='voc21' and expected['voc20']!=expected['voc21']:
            raise ValueError('Shared VOC bootstrap requires identical authoritative key order.')
        delta=[]
        for start in range(0,repetitions,100):
            weights=rng.multinomial(n,np.full(n,1/n),size=min(100,repetitions-start)).astype(float)
            boot_tp=(weights @ sums.reshape(n,-1)).reshape(-1,2,sums.shape[-1])
            boot_union=(weights @ unions.reshape(n,-1)).reshape(boot_tp.shape)
            values=np.nanmean(np.divide(boot_tp,boot_union,out=np.full_like(boot_tp,np.nan),where=boot_union>0),axis=-1)*100
            delta.extend((values[:,0]-values[:,1]).tolist())
        draws.append(delta)
        dataset_draws[ds]=dict(difference_pp=results[ds]['metrics'][reference]-results[ds]['metrics'][control],
            ci95_pp=np.quantile(delta,[.025,.975]).tolist())
    macro=np.asarray(draws).mean(0)
    return dict(reference=reference,control=control,repetitions=repetitions,
        macro_difference_pp=float(np.mean([r['difference_pp'] for r in dataset_draws.values()])),
        macro_ci95_pp=np.quantile(macro,[.025,.975]).tolist(),datasets=dataset_draws,
        note='Conditional on inspected gain candidates and existing development profiles; no correction for selecting the best g0. VOC20/VOC21 use paired shared image resamples. Other domains resampled independently.')


def rank(values):
    result = np.empty(len(values), dtype=float)
    for value in set(values):
        result[np.asarray(values)==value] = np.flatnonzero(np.sort(values)==value).mean()
    return result


def spearman(x,y):
    x,y = rank(x),rank(y)
    if np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def make_figures(root, protocol, results, analyses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gains = np.asarray(protocol['gains'])
    modes = protocol['alias_modes']
    datasets = list(results)
    colors = dict(retained='#1565c0', canonical='#b8600d', count_normalized='#278343')
    figure, axes = plt.subplots(2,5, figsize=(17,7), layout='constrained')
    for ax,ds in zip(axes.flat,datasets):
        for mode in modes:
            values = [results[ds]['metrics'][mode+'__g'+format(g,'g')] for g in gains]
            ax.plot(gains, values, marker='o', ms=3, color=colors[mode], label=mode)
        ax.axhline(results[ds]['metrics']['retained__equal_mean'], color='#555', ls='--', lw=1, label='equal mean (retained)')
        ax.set(title=LABELS[ds], xlabel=r'$g_0$', ylabel='mIoU (%)')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=4)
    for suffix in ('png','pdf'):
        figure.savefig(root/('g0_dataset_curves.'+suffix), dpi=180)
    plt.close(figure)
    figure,axes = plt.subplots(1,2, figsize=(12,4.5), layout='constrained')
    for mode in modes:
        axes[0].plot(gains, analyses[mode]['macro_curve'], marker='o', label=mode, color=colors[mode])
    retained = analyses['retained']
    for group,color in (('remote_sensing','#4064a6'),('natural','#a9603d')):
        axes[1].plot(gains, retained['groups'][group]['curve'], marker='o', color=color, label=group.replace('_',' '))
    for ax in axes:
        ax.set(xlabel=r'$g_0$', ylabel='Equal-protocol mean mIoU (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title('Alias interface')
    axes[1].set_title('Dataset family (retained aliases)')
    for suffix in ('png','pdf'):
        figure.savefig(root/('g0_macro_curves.'+suffix), dpi=180)
    plt.close(figure)


def main(args):
    root = Path(args.root)
    protocol = json.loads((root/'protocol.json').read_text())
    expected = json.loads((root/'expected_keys.json').read_text())
    gains = protocol['gains']
    datasets = list(protocol['datasets'])
    results,arrays = {},{}
    for ds in datasets:
        result, arrays[ds] = audit(root, protocol, expected, ds)
        results[ds] = result
        (root/(ds+'_audited.json')).write_text(json.dumps(result, indent=2)+'\n')
        print('Audited',ds,result['images'],flush=True)
    analyses = {}
    for mode in protocol['alias_modes']:
        values = np.asarray([[results[ds]['metrics'][mode+'__g'+format(g,'g')] for g in gains] for ds in datasets])
        macro = values.mean(0)
        selected = int(macro.argmax())  # Declared ascending gain menu breaks ties toward smaller g.
        oracle = values.max(1)
        lodo, grouped = [],[]
        for index,ds in enumerate(datasets):
            others = [i for i in range(len(datasets)) if i!=index]
            chosen = int(values[others].mean(0).argmax())
            lodo.append(dict(dataset=ds,g0=gains[chosen],miou=float(values[index,chosen]),
                regret_pp=float(oracle[index]-values[index,chosen])))
        groups = [[ds] for ds in datasets if ds not in ('voc20','voc21')]+[['voc20','voc21']]
        for heldout in groups:
            test = [datasets.index(ds) for ds in heldout]
            train = [i for i in range(len(datasets)) if i not in test]
            chosen = int(values[train].mean(0).argmax())
            grouped.extend(dict(dataset=datasets[i],heldout_group=heldout,g0=gains[chosen],
                miou=float(values[i,chosen]),regret_pp=float(oracle[i]-values[i,chosen])) for i in test)
        grouped_macro=np.asarray([values[[datasets.index(ds) for ds in group]].mean(0)
            for group in groups]).mean(0)
        grouped_best=int(grouped_macro.argmax())
        families = {}
        for family,name in [('remote_sensing','remote_sensing'),('natural','natural')]:
            indices = [i for i,ds in enumerate(datasets) if protocol['datasets'][ds]['family']==family]
            # Some historic manifests use 'rs' for the remote-sensing family.
            if family=='remote_sensing' and not indices:
                indices=[i for i,ds in enumerate(datasets) if protocol['datasets'][ds]['family']!='natural']
            curve = values[indices].mean(0)
            chosen = int(curve.argmax())
            families[name]=dict(datasets=[datasets[i] for i in indices],curve=curve.tolist(),
                best_g0=gains[chosen],best_macro=float(curve[chosen]))
        analyses[mode] = dict(macro_curve=macro.tolist(), best_global_g0=gains[selected],
            best_global_macro=float(macro[selected]), per_dataset_oracle_macro=float(oracle.mean()),
            global_regret_pp=float(oracle.mean()-macro[selected]),
            per_dataset={ds:dict(best_g0=gains[int(values[i].argmax())],best_miou=float(oracle[i]),
                global_miou=float(values[i,selected]),global_regret_pp=float(oracle[i]-values[i,selected]))
                for i,ds in enumerate(datasets)},
            lodo=lodo,lodo_macro=float(np.mean([r['miou'] for r in lodo])),
            grouped_lodo=grouped,grouped_lodo_macro=float(np.mean([r['miou'] for r in grouped])),
            nine_group_macro_curve=grouped_macro.tolist(),nine_group_best_g0=gains[grouped_best],
            nine_group_best_macro=float(grouped_macro[grouped_best]),
            groups=families)
    retained = analyses['retained']
    best_g = retained['best_global_g0']
    fusion_losses=[ds for ds in datasets if retained['per_dataset'][ds]['global_miou']<results[ds]['metrics']['retained__equal_mean']]
    best_gs = [retained['per_dataset'][ds]['best_g0'] for ds in datasets]
    covariates = dict(
        wide_minus_local_pp=[results[ds]['metrics']['retained__wide']-results[ds]['metrics']['retained__g0'] for ds in datasets],
        mean_alias_count=[results[ds]['alias_mean'] for ds in datasets],
        alias_count_cv=[results[ds]['alias_count_cv'] for ds in datasets],
        correction_energy_ratio=[results[ds]['diagnostics']['retained']['correction_to_displacement_energy'] for ds in datasets])
    correlations = {key:spearman(best_gs,value) for key,value in covariates.items()}
    controls = {mode:{key:float(np.mean([results[ds]['metrics'][mode+'__'+key] for ds in datasets]))
        for key in ('g0','wide','equal_mean')} for mode in protocol['alias_modes']}
    reproduction = {}
    reference_root = root.parent/'query_slot_best_alias_20261013'/'merged'
    for ds in datasets:
        path = reference_root/(ds+'.json')
        if not path.exists():
            continue
        old = json.loads(path.read_text())
        if set(old['sample_keys'])!=set(expected[ds]):
            raise ValueError('Historical reference split differs: '+ds)
        old_cm = np.asarray(old['metrics'][ds]['confusion_matrix'],dtype=np.int64)
        if not np.array_equal(old_cm.sum(1),arrays[ds][2].sum(0)):
            raise ValueError('Historical target support differs: '+ds)
        g = protocol['datasets'][ds]['frozen_profile']['coupling']
        value = results[ds]['metrics']['retained__g'+format(g,'g')]
        old_value = miou(old_cm.diagonal(),old_cm.sum(0),old_cm.sum(1))
        reproduction[ds] = dict(g0=g,historical_miou=old_value,sweep_miou=value,
            difference_pp=value-old_value,identical_split_and_target_support=True)
    runnerup = int(np.argsort(-np.asarray(retained['macro_curve']),kind='stable')[1])
    reference = 'retained__g'+format(best_g,'g')
    intervals = {control:bootstrap_difference(arrays,results,expected,reference,control,args.bootstrap)
        for control in ('retained__g'+format(gains[runnerup],'g'),'retained__equal_mean')}
    summary = dict(status='complete', datasets=results, analyses=analyses, macro_controls=controls,
        retained_global_vs_equal_fusion=dict(wins=len(datasets)-len(fusion_losses),
            protocols=len(datasets),lower_protocols=fusion_losses,
            mean_difference_pp=retained['best_global_macro']-controls['retained']['equal_mean']),
        historical_reproduction=reproduction,
        paired_conditional_intervals=intervals,
        descriptive_correlations=dict(n=10, spearman_optimal_g0=correlations, covariates=covariates),
        protocol_images=sum(r['images'] for r in results.values()),
        interpretation='Best values are development selections, not independent-test estimates. LODO only holds out g0 selection: aliases, calibration and other frozen task settings previously used labels. VOC20/VOC21 share images.',
        averaging='Equal-protocol mean of aggregate dataset mIoU; never mean image IoU.',
        gain_tie_break='Smallest gain in the ascending frozen menu.')
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    shared_configs=root/'configs_shared_g0'
    shared_configs.mkdir(exist_ok=True)
    for ds in datasets:
        config=json.loads((root/'configs'/(ds+'.yaml')).read_text())
        config['profile']['coupling']=best_g
        config['provenance']['writeback_selection']=dict(
            experiment='geov2_g0_sweep_20261014',g0=best_g,
            criterion='Highest equal-ten-protocol full-validation mIoU in the predeclared grid',
            labeled_selection=True,other_task_specific_settings_unchanged=True)
        (shared_configs/(ds+'.yaml')).write_text(json.dumps(config,indent=2)+'\n')
    lines=['# Geov2 fixed structure writeback: ten-protocol g0 sweep','',
        'Full model: 2026-10-13 retained best Alias + QuerySlot (24 slots) + DonorBefore, with original local strength on every dataset. PC60 replaces LandCover.ai. All other task-specific calibration stays frozen.','',
        'Writer: Z = L + g0 H(W-L). g0=0 is the local endpoint; g0=1 is the unrelaxed quadratic optimum. The wide endpoint and literal equal-logit mean are separate controls. Each image shares visual observations across all 30 arms; masks are used only after inference.','',
        f"{summary['protocol_images']:,} protocol images, ten complete splits. VOC20/VOC21 share 1,449 images. Each dataset mIoU comes from its summed confusion matrix; means below weight protocols equally.",'',
        '## Retained Alias: full split mIoU (%)','',
        table(['Dataset','N']+[format(g,'g') for g in gains]+['Equal mean','Wide','Best g0'],
            [[LABELS[ds],results[ds]['images']]+[f"{results[ds]['metrics']['retained__g'+format(g,'g')]:.4f}" for g in gains]+
             [f"{results[ds]['metrics']['retained__equal_mean']:.4f}",f"{results[ds]['metrics']['retained__wide']:.4f}",retained['per_dataset'][ds]['best_g0']] for ds in datasets]+
            [['Equal-protocol mean','—']+[f'{v:.4f}' for v in retained['macro_curve']]+[f"{controls['retained']['equal_mean']:.4f}",f"{controls['retained']['wide']:.4f}",best_g]]),'',
        f"Best shared g0 = **{best_g:g}**, mean **{retained['best_global_macro']:.4f}%**. Per-dataset selection reference = {retained['per_dataset_oracle_macro']:.4f}%; shared-step cost = {retained['global_regret_pp']:.4f} pp. These best values use validation labels for analysis; inference itself has no labels.",'',
        f"The shared step exceeds equal fusion on {len(datasets)-len(fusion_losses)}/10 protocols; it is lower on {', '.join(LABELS[ds] for ds in fusion_losses)}. The positive macro difference is not an all-protocol improvement.",'',
        table(['Paired comparison (shared g0)','Mean difference (pp)','Conditional 95% interval (pp)'],
            [[reference+' vs '+control,f"{row['macro_difference_pp']:.4f}",f"[{row['macro_ci95_pp'][0]:.4f}, {row['macro_ci95_pp'][1]:.4f}]"] for control,row in intervals.items()]),'',
        'Intervals resample images within each dataset and pair shared VOC20/VOC21 images. They are conditional on the inspected candidates; selection of the winning g0 is not accounted for.','',
        '## Held-out g0 selection','',
        table(['Dataset','Best g0','Shared-step loss (pp)','LODO g0','LODO mIoU','Grouped LODO g0','Grouped LODO mIoU'],
            [[LABELS[ds],retained['per_dataset'][ds]['best_g0'],f"{retained['per_dataset'][ds]['global_regret_pp']:.4f}",
              next(r['g0'] for r in retained['lodo'] if r['dataset']==ds),f"{next(r['miou'] for r in retained['lodo'] if r['dataset']==ds):.4f}",
              next(r['g0'] for r in retained['grouped_lodo'] if r['dataset']==ds),f"{next(r['miou'] for r in retained['grouped_lodo'] if r['dataset']==ds):.4f}"] for ds in datasets]),'',
        f"LODO mean = {retained['lodo_macro']:.4f}%; grouped LODO mean = {retained['grouped_lodo_macro']:.4f}%. Grouped selection holds VOC20/VOC21 out together. Only g0 selection is held out: previously selected aliases/calibration are still task-specific.",'',
        f"Weighting the VOC20/VOC21 pair as one group gives nine-group best g0={retained['nine_group_best_g0']:g}, mean={retained['nine_group_best_macro']:.4f}%. This is a sensitivity analysis; the declared primary remains ten equally weighted protocols.",'',
        '## Alias interfaces','',
        'retained keeps all aliases. canonical keeps each class’s first canonical phrase and the same cached templates. count_normalized keeps all aliases and existing scores, subtracting log(alias_count)/tau only from wide class LSE. The current QuerySlot CompactContextScorer has salience disabled; the incumbent VIP salience is not used here. Local class LME already normalizes alias count. PC60 keeps all 401 residual concepts and residual_max in every arm. This control isolates the explicit wide count offset, not every statistical effect of having more words.','',
        table(['Interface','Best shared g0','Best shared mean','Mean at retained shared g0','Per-dataset reference'],
            [[mode,analyses[mode]['best_global_g0'],f"{analyses[mode]['best_global_macro']:.4f}",
              f"{analyses[mode]['macro_curve'][gains.index(best_g)]:.4f}",f"{analyses[mode]['per_dataset_oracle_macro']:.4f}"] for mode in protocol['alias_modes']]),'',
        table(['Dataset','Mean alias count','Count CV','retained best g0','canonical best g0','normalized best g0'],
            [[LABELS[ds],f"{results[ds]['alias_mean']:.2f}",f"{results[ds]['alias_count_cv']:.3f}"]+
              [analyses[mode]['per_dataset'][ds]['best_g0'] for mode in protocol['alias_modes']] for ds in datasets]),'',
        '## Dataset family and descriptive relationships','',
        table(['Family','Best shared g0','Mean mIoU'],[[name,group['best_g0'],f"{group['best_macro']:.4f}"] for name,group in retained['groups'].items()]),'',
        table(['Covariate','Spearman correlation with retained optimal g0'],[[name,'undefined' if value is None else f'{value:.3f}'] for name,value in correlations.items()]),'',
        'Ten correlated protocols are insufficient to infer causality or establish a reliable label-free g0 selector. Endpoint advantage uses labels and is explanatory only. Canonical changes both phrase diversity and count; its calibration was not re-tuned. A single g0 does not imply a globally fixed full model because temperatures, vocabularies and background rules remain task-specific.','',
        '## Figures','',
        '![Dataset curves](g0_dataset_curves.png)','',
        '![Protocol means](g0_macro_curves.png)','',
        '## Reproduce and audit','',
        table(['Dataset','Original g0','10/13 mIoU','Current mIoU','Difference (pp)'],
            [[LABELS[ds],r['g0'],f"{r['historical_miou']:.4f}",f"{r['sweep_miou']:.4f}",f"{r['difference_pp']:+.4f}"] for ds,r in reproduction.items()]),'',
        'Historical references use the same full splits and target support. Tiny differences can arise from CUDA reductions and slot clustering; current paired arms share one set of observations.','',
        'Run tools/collect_geov2_g0.py, then python tools/report_geov2_g0.py. protocol.json records the predeclared scan; expected_keys.json records authoritative full split order. full/*/statistics.npz stores per-image TP, predicted and target supports for each arm, sufficient to recompute dataset mIoU. The audit verifies full split keys, all 29 shards, aggregate matrices, class supports and the unchanged normalized local endpoint. Evaluated scripts are retained under source/scripts/. configs_shared_g0/ contains opt-in deployment profiles for the selected scalar, with labeled-selection provenance; use --wide-branch query_slot --efficient --writeback coupled with the frozen inference/evaluation entry points. No latency claim is derived from the 30-arm sweep.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    worst = max(datasets,key=lambda ds:retained['per_dataset'][ds]['global_regret_pp'])
    changed = [LABELS[ds] for ds in datasets if retained['per_dataset'][ds]['best_g0']
        !=analyses['count_normalized']['per_dataset'][ds]['best_g0']]
    old_mean=float(np.mean([results[ds]['metrics']['retained__g'+format(protocol['datasets'][ds]['frozen_profile']['coupling'],'g')] for ds in datasets]))
    zh=['# Geov2：十协议固定结构写回步长扫描','',
        f"全量完成，共 {summary['protocol_images']:,} 次协议图像评测。PC60 已替换 LandCover.ai。模型固定为 10/13 最佳 Alias + QuerySlot（24 槽）+ DonorBefore，所有局部 strength=original。",'',
        f"**统一最佳 g0={best_g:g}，十协议平均 mIoU={retained['best_global_macro']:.4f}%。** 每个数据集分别选最佳步长的参考上界为 {retained['per_dataset_oracle_macro']:.4f}%，统一步长少 {retained['global_regret_pp']:.4f}pp。旧逐域步长在本轮配对观测上的平均为 {old_mean:.4f}%。",'',
        f"统一步长的最大单域损失出现在 {LABELS[worst]}，相对该域最优少 {retained['per_dataset'][worst]['global_regret_pp']:.4f}pp。选择统一步长应同时看这个损失，不能只看平均值。",'',
        f"统一步长比等权融合的十协议平均高 {retained['best_global_macro']-controls['retained']['equal_mean']:.4f}pp，胜出 {len(datasets)-len(fusion_losses)}/10 个协议；{ '、'.join(LABELS[ds] for ds in fusion_losses)} 仍低于等权融合，不能写成全部协议均超过融合。",'',
        table(['数据集','0','0.25','0.5','0.75','1','1.5','2','3','等权融合','逐域最佳g0'],
            [[LABELS[ds]]+[f"{results[ds]['metrics']['retained__g'+format(g,'g')]:.4f}" for g in gains]+
             [f"{results[ds]['metrics']['retained__equal_mean']:.4f}",retained['per_dataset'][ds]['best_g0']] for ds in datasets]+
            [['十协议平均']+[f'{v:.4f}' for v in retained['macro_curve']]+[f"{controls['retained']['equal_mean']:.4f}",best_g]]),'',
        f"留一数据集选 g0 的平均为 {retained['lodo_macro']:.4f}%；将 VOC20/VOC21 共同留出时为 {retained['grouped_lodo_macro']:.4f}%。VOC20/VOC21 共用 1,449 张图，不能当作两份独立来源。逐域最优和全域最优都属于开发分析；留一只隔离 g0 的选择，不隔离历史 Alias、温度和背景校准的标签使用。",'',
        f"把 VOC20/VOC21 的平均作为一个组计权，九组平均的最佳 g0={retained['nine_group_best_g0']:g}，mIoU={retained['nine_group_best_macro']:.4f}%。该项仅检查共源计权是否影响结论，主结果仍为十协议等权平均。",'',
        '## 与数据集和 Alias 的关系','',
        table(['接口','统一最佳g0','最佳平均mIoU','固定为retained最佳g0的平均'],
            [[mode,analyses[mode]['best_global_g0'],f"{analyses[mode]['best_global_macro']:.4f}",f"{analyses[mode]['macro_curve'][gains.index(best_g)]:.4f}"] for mode in protocol['alias_modes']]),'',
        'count_normalized 仅去掉宽视野 class LSE 中的 log(alias_count)/tau，所有短语保留；canonical 每类只留第一条短语，模板不变。PC60 的 401 残余概念和 residual_max 全程保留。', '',
        '去掉显式词数偏置后，最佳 g0 改变的协议：'+('、'.join(changed) if changed else '无')+'。这说明步长与当前语言接口存在交互；不能据十个协议推断一个通用自动选择规则。canonical 同时改变语义覆盖和词数，其结果也受未重新校准的温度和背景规则影响。','',
        '局部 LME 已按词数归一化，当前 QuerySlot 宽视野没有启用 VIP salience。不同类词数不等会在宽视野造成 class bias；结构修正再将其按 g0 写回，因此 Alias、两路分数尺度和背景规则都可能改变最佳步长。','',
        f"在这十个协议上，宽视野相对局部的端点优势与最佳 g0 的 Spearman 相关为 {correlations['wide_minus_local_pp']:.3f}，而平均 Alias 数的相关为 {correlations['mean_alias_count']:.3f}。这支持把步长看作两路可靠性与刻度的取舍，不能仅按词数或遥感/自然类别决定；端点优势用到了标签，只能用于解释，不能作为无标签推理规则。",'',
        '## 使用建议与论文口径','',
        f"统一配置的主要候选是 g0={best_g:g}；逐域最佳仅作为开发适配上界，不能称为无需调参的统一方法。固定 H 与固定 g0 仍是结构写回，自适应 gate 不作为已证实最优的默认组件。单一 g0 也不等于整套模型单一配置，因为其他任务参数仍沿用历史选择。",'',
        '本轮只改变写回步长，不增加图像编码或额外网络；30-arm 扫描的墙钟耗时不代表单次推理速度，不能与之前的独立成本测试混用。','',
        '![十协议步长曲线](g0_dataset_curves.png)','',
        '![平均mIoU曲线](g0_macro_curves.png)','',
        '完整审计、配对区间、逐域留出选择与历史复现差值见 [详细报告](REPORT.md)；机器可读汇总见 SUMMARY.json。','']
    (root/'REPORT_ZH.md').write_text('\n'.join(zh),encoding='utf-8')
    make_figures(root,protocol,results,analyses)
    print(json.dumps(dict(best_shared_g0=best_g,macro_miou=retained['best_global_macro'],
        per_dataset_reference=retained['per_dataset_oracle_macro'],grouped_lodo=retained['grouped_lodo_macro']),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='research/geov2_g0_sweep_20261014')
    parser.add_argument('--bootstrap',type=int,default=2000)
    main(parser.parse_args())
