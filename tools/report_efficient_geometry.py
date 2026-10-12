"""One version-consistent table of best frozen accuracy and paired execution cost."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'research/efficient_geometry_20261012'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def protocol_cost(cost, name, protocol):
    rows=[r for r in cost['costs'][name]['images'] if r['protocol']==protocol]
    return dict(median_ms=statistics.median(t for r in rows for t in r['milliseconds']),
        peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows))


def baseline_cost_section(dataset_order, references):
    """Use verified current costs in the shared baseline report."""
    summary_path=FOLDER/'comparison_summary.json'
    if not summary_path.exists():return [],None
    summary=read(summary_path)
    if summary['status']!='complete':return [],None
    rows=summary['comparisons']
    if set(rows)!=set(dataset_order) or summary['completed_costs']!=15:
        raise RuntimeError('Latest Geometry protocol/cost coverage differs from baselines.')
    total=sum(row['images'] for row in rows.values())
    if total!=45200:raise RuntimeError('Latest Geometry coverage is incomplete.')
    lines=['','## Latest Geometry whole-image cost (2026-10-12)','',
        'All 15 protocols / 45,200 protocol images completed paired compact-versus-efficient '
        'full evaluation. Zero prediction pixels changed; both backends exactly reproduce '
        'the retained historical aggregate confusion matrices and all mIoU/per-class scores. '
        'Unique coverage, per-image confusion sums and scored target counts are verified. '
        'The following current costs accompany the Geometry accuracy column above.', '',
        'A800; seven fixed RGB images per protocol, two warmups and seven synchronized '
        'repeats per image. Latency is the median over measured repetitions; memory is '
        'peak CUDA allocated MiB over those images. Whole-image inference includes resize, '
        'all visual forwards/readouts, reconstruction, original-resolution restoration and '
        'CPU class IDs. Decode, initialization, offline vocabulary selection and labeled '
        'development are excluded. Geometry uses Torch 2.6/CUDA 12.4; the five native '
        'baselines retain their separately recorded Torch 2.4 environment. VIP costs are '
        'historical same-text measurements on the same image keys, rather than a simultaneous '
        'remeasurement; its timing bank may differ from the strongest VIP accuracy reference.', '',
        '|Protocol|Images|Geometry mIoU|Compact ms|Efficient ms|Compact MiB|Efficient MiB|Prior VIP ms|Prior VIP MiB|Efficient/VIP time|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    published={}
    for d in dataset_order:
        row=rows[d];full=read(FOLDER/'merged'/f'{d}.json');cost=read(FOLDER/'cost'/f'{d}.json')
        if not all(full[f] for f in ('coverage_verified','scored_target_counts_match','per_image_sums_verified')):
            raise RuntimeError('Unverified latest Geometry result: '+d)
        if any(full['changed_pixels'].values()) or not all(full['exact_historical_confusion'].values()):
            raise RuntimeError('Latest Geometry accuracy differs: '+d)
        for p in ('P','D') if d=='loveda' else (d,):
            key=d+'__'+p;reference=references[key]
            miou=row['metrics']['efficient'][p]['mean_iou_percent']
            if miou!=reference['ours']:raise RuntimeError('Geometry accuracy reference differs: '+key)
            old,new=(protocol_cost(cost,name,p) for name in ('compact','efficient'))
            vip=reference['vip_cost'];ratio=new['median_ms']/vip['median_ms']
            published[key]=dict(dataset=d,protocol=p,images=row['images'],mean_iou_percent=miou,
                compact=old,efficient=new,prior_same_text_vip=vip,efficient_vip_time_ratio=ratio,
                changed_prediction_pixels=0,exact_historical_confusion=True)
            label=d+(' '+p if d=='loveda' else '')
            lines.append(f'|{label}|{row["images"]}|{miou:.4f}|{old["median_ms"]:.2f}|'
                f'{new["median_ms"]:.2f}|{old["peak_allocated_mib"]:.2f}|{new["peak_allocated_mib"]:.2f}|'
                f'{vip["median_ms"]:.2f}|{vip["peak_allocated_mib"]:.2f}|{ratio:.3f}|')
    lines+=['',
        'LoveDA P/D costs are separated by readout; their images are counted once in the '
        '45,200 total. VOC20/21 and PC59/60 share photographs. UDD5 is the established '
        '40-image five-class protocol. Median latency reduction across the 15 protocols '
        'is 6.08% versus the already compact backend. Several small-class protocols have '
        'peak-memory increases of about 4–24 MiB; both backends remain available. '
        'The original vocabulary, coefficients, visual view budget and numerical dtypes '
        'are retained. These are execution savings with unchanged accuracy, not a new '
        'model-selection result.', '',
        'Detailed execution changes and per-class replay: '
        '[EFFICIENT_GEOMETRY_20261012.md](EFFICIENT_GEOMETRY_20261012.md). '
        'Machine-readable comparisons: '
        '[comparison_summary.json](efficient_geometry_20261012/comparison_summary.json). '
        'Per-image timing/memory records: [cost/](efficient_geometry_20261012/cost/).']
    data=dict(status='complete',protocols=15,protocol_images=total,standalone_costs=15,
        changed_prediction_pixels=0,source='efficient_geometry_20261012/comparison_summary.json',rows=published)
    return lines,data


def update_baseline_report():
    target=ROOT/'research/TRAINING_FREE_BASELINES_FULL_20261011.md'
    summary_path=ROOT/'research/training_free_baselines_full_20261011/comparison_summary.json'
    if not target.exists() or not summary_path.exists():return
    protocol=read(ROOT/'research/training_free_baselines_full_20261011/protocol.json')
    summary=read(summary_path)
    lines,data=baseline_cost_section(protocol['dataset_order'],summary['previous_references'])
    if not lines:return
    body=target.read_text(encoding='utf-8');end='## Standalone whole-image cost'
    start='## Latest Geometry whole-image cost (2026-10-12)'
    before=body.split(start,1)[0] if start in body else body.split(end,1)[0]
    after=body[body.index(end):]
    target.write_text(before.rstrip()+'\n'+'\n'.join(lines)+'\n\n'+after,encoding='utf-8')
    summary['latest_geometry_execution']=data
    summary_path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')


def report():
    protocol=read(FOLDER/'protocol.json');rows={};cost_rows={}
    lines=['# Geometry: all-protocol efficient execution, latest retained configurations','',
        'The vocabulary, readout coefficients, background rules, view sizes/counts, pretrained weights and '
        'FP32/BF16/FP16/FP64 arithmetic roles are frozen. VOC20, VOC21 and PC60 use the latest retained '
        'variable-word configurations, not their older Kev trial settings. This is an execution optimization, '
        'not a new accuracy search or proof of a global optimum.','',
        '## Execution changes','',
        '- Cache the static spatial Geometry term. Keep the original FP64 reconstruction solve and omit only its unused diagnostic residual matrix multiplication.',
        '- Reuse the same raw-patch VIP proxy relation across the two wide head blocks. Preserve the existing self-Value fallback only on empty proxy rows.',
        '- Batch equal-size alias groups in the local log-mean-exp and wide salience/log-sum-exp. Preserve each class\'s alias count, inherited salience and aggregation mass; no words are deleted.',
        '- Bound FP64 dense-softmax workspace with spatial slabs. Restore class planes in chunks when background calibration does not need a classwise sum. Keep exact original calibration in bounded spatial slabs otherwise.',
        '- Retain the earlier compact deployment optimizations: text towers off GPU, requested head only, cached text metadata, uint8 CPU IDs. Original and first compact paths remain available.','',
        '## Full accuracy and whole-image cost','',
        'Costs use the same seven fixed whole RGB inputs per protocol on A800, two warmups and seven synchronized repeats. '
        'They include resizing, every visual encoding/readout, reconstruction, original-resolution restoration and CPU IDs. '
        'Decode, initialization, offline language selection and labeled development are excluded. Costs are scheduled '
        'one at a time within this suite; unrelated user jobs are preserved. Prior VIP timings are historical '
        'measurements on these same sample keys, not simultaneous measurements or paper numbers.','',
        '| Protocol | Images | Historical best mIoU | Compact replay | Efficient mIoU | Compact ms | Efficient ms | Compact MiB | Efficient MiB | Prior VIP ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d,e in protocol['datasets'].items():
        merged_path=FOLDER/'merged'/f'{d}.json';cost_path=FOLDER/'cost'/f'{d}.json'
        if cost_path.exists():
            cost=read(cost_path);primary='D' if d=='loveda' else d
            old,new=(protocol_cost(cost,name,primary) for name in ('compact','efficient'))
            vip=read(ROOT/'research/vip_kev_same_vocabulary_20261010'/d/'cost.json')
            cost_rows[d]=dict(compact_ms=old['median_ms'],efficient_ms=new['median_ms'],
                compact_mib=old['peak_allocated_mib'],efficient_mib=new['peak_allocated_mib'],
                prior_vip_ms=vip['median_ms'],sample_changed_pixels=cost['changed_pixels'],
                latency_reduction_percent=100*(1-new['median_ms']/old['median_ms']),
                memory_reduction_percent=100*(1-new['peak_allocated_mib']/old['peak_allocated_mib']))
        if not merged_path.exists() or not cost_path.exists():
            for p,previous in e['incumbent_metrics'].items():
                label=d+(' '+p if d=='loveda' else '')
                if cost_path.exists():old,new=(protocol_cost(cost,name,p) for name in ('compact','efficient'))
                values=(f'{old["median_ms"]:.2f}',f'{new["median_ms"]:.2f}',f'{old["peak_allocated_mib"]:.1f}',
                    f'{new["peak_allocated_mib"]:.1f}',f'{vip["median_ms"]:.2f}') if cost_path.exists() else ('pending',)*5
                lines.append(f'| {label} | {e["total_images"]} | {previous["mean_iou_percent"]:.4f} | pending | pending | '+' | '.join(values)+' |')
            continue
        full=read(merged_path);cost=read(cost_path)
        for field in ('coverage_verified','scored_target_counts_match','per_image_sums_verified'):
            if not full[field]:raise RuntimeError('Unverified full result: '+d)
        old,new=(protocol_cost(cost,name,'D' if d=='loveda' else d) for name in ('compact','efficient'))
        vip=read(ROOT/'research/vip_kev_same_vocabulary_20261010'/d/'cost.json')
        rows[d]=dict(images=e['total_images'],metrics=full['metrics'],historical=e['incumbent_metrics'],
            changed_pixels=full['changed_pixels'],exact_historical_confusion=full['exact_historical_confusion'],
            compact_ms=old['median_ms'],efficient_ms=new['median_ms'],compact_mib=old['peak_allocated_mib'],
            efficient_mib=new['peak_allocated_mib'],prior_vip_ms=vip['median_ms'],
            latency_reduction_percent=100*(1-new['median_ms']/old['median_ms']),
            memory_reduction_percent=100*(1-new['peak_allocated_mib']/old['peak_allocated_mib']),config=full['config'])
        for p,previous in e['incumbent_metrics'].items():
            label=d+(' '+p if d=='loveda' else '')
            old,new=(protocol_cost(cost,name,p) for name in ('compact','efficient'))
            lines.append(f'| {label} | {e["total_images"]} | {previous["mean_iou_percent"]:.4f} | '
                f'{full["metrics"]["compact"][p]["mean_iou_percent"]:.4f} | {full["metrics"]["efficient"][p]["mean_iou_percent"]:.4f} | '
                f'{old["median_ms"]:.2f} | {new["median_ms"]:.2f} | {old["peak_allocated_mib"]:.1f} | '
                f'{new["peak_allocated_mib"]:.1f} | {vip["median_ms"]:.2f} |')
    complete=len(rows)==15
    lines+=['','## Verification and limitations','',
        f'{len(rows)}/15 protocols have both completed paired full evaluation and standalone costs. '
        'Every complete image is inferred with the old compact and new efficient backends, and changed prediction pixels '
        'are counted directly. Per-image confusion matrices sum to the reported aggregates. Complete unique '
        'sample coverage and scored target counts are checked against the already verified local VIP inputs.',
        '', 'The frozen protocol totals45,200 protocol images; VOC20/21 and PC59/60 share photographs. '
        'UDD5 here is the established40-image five-class validation protocol, not the older540-image UDD run. '
        'LoveDA P and D are independent six-/seven-class readouts; eight-remote means count D once. '
        'PC60 retains its401-concept residual ontology. LandCover.ai substitutes for unlabeled iSAID.',
        '', 'The retained configurations used labeled development and previously viewed full validation results. '
        'These results remain exploratory. Training-free means no weight fitting, not zero inference cost. '
        'Latency and peak CUDA allocated memory are reported together with mIoU; no universal venue-specific '
        'latency or VRAM cutoff is assumed. No semantic accuracy gain is attributed to execution changes. '
        'Tiny floating-point differences, if any, must be reported and cannot be called bit-exact replay.']
    if cost_rows:
        regressions={d:v['efficient_mib']-v['compact_mib'] for d,v in cost_rows.items() if v['efficient_mib']>v['compact_mib']}
        lines+=['','Completed standalone cost protocols: '+str(len(cost_rows))+'.',
            'Small peak-memory increases (MiB): `'+json.dumps(regressions)+'`. '
            'The compact backend remains available where minimum memory matters more than these latency savings.',
            '', 'The preserved preliminary `profile/` results include a larger finite-check workspace '
            '(ADE150 efficient3766.5MiB versus compact3295.4MiB). The final `cost/` backend checks '
            'finite values in slabs and removes that regression. These are different execution revisions, '
            'not selected repetitions of an unchanged implementation. Preliminary measurements are not '
            'substituted for final standalone costs.']
    if rows:
        lines+=['',f'Median paired latency reduction: {statistics.median(v["latency_reduction_percent"] for v in rows.values()):.2f}%.',
            f'Total changed prediction pixels: {sum(sum(v["changed_pixels"].values()) for v in rows.values())}.']
    remote=('vdd','potsdam','udd5','oem','vaihingen','landcoverai','loveda','flair1')
    if all(d in rows for d in remote):
        means={name:statistics.mean(rows[d]['metrics'][name]['D' if d=='loveda' else d]['mean_iou_percent'] for d in remote)
            for name in ('compact','efficient')}
        lines+=['','Eight-remote mean, LoveDA D counted once: `'+json.dumps(means)+'`.']
    for d,row in rows.items():
        lines+=['','## '+d,'', 'Frozen profile: `'+json.dumps(row['config']['profile'])+'`.','',
            'Changed pixels: `'+json.dumps(row['changed_pixels'])+'`. Historical confusion agreement: `'+json.dumps(row['exact_historical_confusion'])+'`.']
        for p,old in row['historical'].items():
            new=row['metrics']['efficient'][p]
            lines+=['','Protocol '+p+':','', '| Class | Historical IoU | Efficient IoU | Δ pp |','| --- | ---: | ---: | ---: |']
            for a,b in zip(old['per_class'],new['per_class']):
                before,after=a['iou_percent'],b['iou_percent']
                values=('n/a','n/a','n/a') if before is None or after is None else (f'{before:.4f}',f'{after:.4f}',f'{after-before:+.4f}')
                lines.append('| '+a['name']+' | '+' | '.join(values)+' |')
    summary=dict(status='complete' if complete else 'partial',completed_protocols=len(rows),protocol_images=45200,
        completed_costs=len(cost_rows),costs=cost_rows,comparisons=rows)
    (FOLDER/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    target=ROOT/'research/EFFICIENT_GEOMETRY_20261012.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    update_baseline_report()
    print(json.dumps(dict(report=str(target),status=summary['status'],protocols=len(rows))))


if __name__=='__main__':report()
