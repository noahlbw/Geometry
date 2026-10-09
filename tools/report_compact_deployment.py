"""Summarize measured execution changes without replacing frozen accuracy."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'research/compact_deployment_20261010'


def main():
    protocol=json.loads((ROOT/'research/kev_alias_search_20261009/protocol.json').read_text())
    lines=['# Frozen Geometry deployment: execution-only optimization','',
        '2026-10-10. Architecture, frozen per-task configurations, selected words, visual budget and numerical dtypes are unchanged. '
        'This optimizes the retained Kev-selected model; it does not select a new model from the full results.','',
        '## Changes','',
        '- Remove both text towers from online GPU residence after encoding. The portable API drops them before CUDA loading when a matching text cache already exists, and releases their CPU storage after fresh encoding.',
        '- Compute only the requested Geometry head: omit unused native diagnostic heads and unused alternate Geometry heads.',
        '- Cache wide text conversions, normalized text means and integer class groups. Keep template-level BF16 multiplication and the original within-class reduction; no averaging-before-multiplication approximation.',
        '- Transfer final class IDs as uint8 for these <=256-class protocols. Defer the unused PC60 intermediate prediction/CPU copy. Class indices and calibrated decisions are unchanged.',
        '- Keep Geometry FP32 and wide FP16 visual weight copies. Sharing them would change rounding. Keep original local/wide crops, alias counts, thresholds, FP64 reconstruction and output interpolation.','',
        '## Paired whole-image costs','',
        'Same A800, same seven deterministic whole RGB inputs per protocol, two warmups and seven synchronized repeats. '
        'Includes resize, all encoding/readout, reconstruction, full-size restoration and CPU prediction; excludes decode/init/offline text and Kev selection. '
        'The baseline and compact are measured in the same process in successive phases. VDD/UDD5 were additionally repeated alone after observed system-level timing variance; '
        'their isolated measurements are primary below when present. Other domains were measured while independent GPU jobs could share the host CPU. '
        'VIP is the earlier same-text native measurement on the same samples, not a simultaneous remeasurement or a paper number. '
        'Peak is CUDA allocated MiB during warmed online inference; initialization and first vocabulary encoding are separate costs.','',
        '| Protocol | Reference ms | Compact ms | Latency reduction | Reference MiB | Compact MiB | VIP ms | Compact/VIP |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows={}
    for d in protocol['datasets']:
        path=FOLDER/('isolated_'+d+'.json') if d in ('vdd','udd5') and (FOLDER/('isolated_'+d+'.json')).exists() else FOLDER/('final_'+d+'.json')
        row=json.loads(path.read_text())
        if row['status']!='complete' or not row['exact_sample_predictions'] or row['images']!=7:
            raise RuntimeError('Final cost/accuracy verification incomplete: '+d)
        old,new=row['results']['reference'],row['results']['compact']
        vip=json.loads((ROOT/'research/vip_kev_same_vocabulary_20261010'/d/'cost.json').read_text())
        vipms=vip['median_ms']
        item=dict(reference_ms=old['median_ms'],compact_ms=new['median_ms'],reference_mib=old['peak_allocated_mib'],
            compact_mib=new['peak_allocated_mib'],vip_ms=vipms,ratio=new['median_ms']/vipms,source=path.name,
            latency_reduction_percent=100*(1-new['median_ms']/old['median_ms']),
            memory_reduction_percent=100*(1-new['peak_allocated_mib']/old['peak_allocated_mib']),sample_predictions_exact=True)
        rows[d]=item
        lines.append(f'| {d} | {item["reference_ms"]:.2f} | {item["compact_ms"]:.2f} | {item["latency_reduction_percent"]:+.1f}% | '
            f'{item["reference_mib"]:.1f} | {item["compact_mib"]:.1f} | {vipms:.2f} | {item["ratio"]:.3f} |')
    lines+=['','Initial simultaneous host-use measurements are retained, including negative speed outcomes:', '',
        '| Dataset | Initial reference ms | Initial compact ms | Isolated reference ms | Isolated compact ms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for d in ('vdd','udd5'):
        initial=json.loads((FOLDER/('final_'+d+'.json')).read_text())['results']
        isolated=rows[d]
        lines.append(f'| {d} | {initial["reference"]["median_ms"]:.2f} | {initial["compact"]["median_ms"]:.2f} | '
            f'{isolated["reference_ms"]:.2f} | {isolated["compact_ms"]:.2f} |')
    lines+=['','This change of scheduling regime shows sensitivity to host/system conditions; it does not establish a specific causal bottleneck. '
        'Do not choose arbitrary best repetitions or claim uniform acceleration under contention. '
        'The earlier implementation-only phase is retained as `sample_*.json`. Its UDD5 successive-phase timing was anomalous '
        '(336.10 ->428.27ms); an alternating-order check with both text towers on CPU measured470.03 ->456.67ms. '
        'Do not erase the negative phase or interpret its difference as a precision change. Final endpoint costs above additionally include the byte-label/deferred-output optimization.','',
        '## Accuracy verification','',
        'All15 protocols: seven deterministic full images each, zero changed prediction pixels between reference and compact. '
        'This is sampled agreement, not full15-protocol accuracy re-evaluation. Five complete protocols also replayed the optimized visual/readout path and matched every stored per-image confusion matrix:', '',
        '| Dataset | Full images | mIoU | Per-image confusion |',
        '| --- | ---: | ---: | --- |']
    full={}
    for d in ('vdd','potsdam','udd5','voc21','ade150'):
        row=json.loads((FOLDER/('full_'+d+'.json')).read_text())
        if row['status']!='complete' or not row['exact_per_image_confusion'] or row['images']!=protocol['datasets'][d]['total_images']:
            raise RuntimeError('Full verification incomplete: '+d)
        metric=row['metrics'][d]['mean_iou_percent']
        full[d]=dict(images=row['images'],miou=metric,exact_per_image_confusion=True)
        lines.append(f'| {d} | {row["images"]} | {metric:.4f} | exact |')
    lines+=['','The final uint8 transfer is a lossless class-ID conversion for the frozen <=256-class protocols; final15-protocol sampled checks cover that endpoint. '
        'Full checks used historical text caches; they do not assert that all re-encoded caches on other machines are bit-identical.','',
        'Three focused CPU tests cover exact requested Geometry head outputs, text release preserving visual weights, and deferred prediction preserving probabilities.','',
        '## Independent reproduction entry','',
        'YAML/configuration and actual selected words are exported for15 protocols. `infer_frozen_geometry.py` and `evaluate_frozen_geometry.py` '
        'build directly from checkpoint/words without historical experiment directories. Fresh-encoding smoke tests:', '',
        '| Protocol | Local vector max error | Wide vector max error | Changed prediction pixels |',
        '| --- | ---: | ---: | ---: |']
    portable={}
    for d in ('vdd','oem','ade150','context60','loveda'):
        path=FOLDER/('portable_'+d+'.json')
        if not path.exists():continue
        row=json.loads(path.read_text());portable[d]=row
        if row['status']!='complete' or not row['exact_prediction'] or row.get('P_exact_prediction',True) is not True:
            raise RuntimeError('Portable smoke differs: '+d)
        lines.append(f'| {d} | {row["local_vector_max_error"]:.3g} | {row["wide_vector_max_error"]:.3g} | {row["changed_pixels"]} |')
    cached={}
    for path in FOLDER.glob('portable_cached_*.json'):
        row=json.loads(path.read_text());cached[row['dataset']]=row
        if not row['exact_prediction']:raise RuntimeError('Cached visual-only constructor differs.')
    lines+=['','PC60 includes the exact401 residual names and its protected rule. LoveDA P is an independent six-class readout. '
        'Fresh encode smoke does not replace full accuracy verification of all regenerated caches. Cache identities and each smoke output are included. '
        'The cached visual-only constructor is separately checked where `portable_cached_*.json` exists.','',
        '## Interpretation for a training-free paper','',
        'Training-free means no weight training, not zero inference cost. Report full-image mIoU/latency/peak memory together with hardware, resolution, view count, '
        'warmup, timing boundaries and offline language/development costs. There is no universal CVPR maximum latency or VRAM allowance. '
        'Use same-device VIP for direct efficiency comparison; the paper screenshot is not a matched-hardware experiment. '
        'Keep accuracy gains, residual weaknesses and added costs explicit; reducing deployment waste is an implementation contribution, not a new semantic-module contribution.','',
        'The frozen source words and per-task profiles used labeled development and previously developed data. All inherited accuracy comparisons remain exploratory; '
        'these execution checks do not turn them into independent SOTA validation. The original model/reports and raw negative timings remain preserved.','']
    result=dict(status='complete',protocols=15,sampled_protocol_images=105,
        full_verified_images=sum(v['images'] for v in full.values()),costs=rows,full_accuracy=full,
        portable=portable,cached_constructor=cached,
        median_latency_reduction_percent=statistics.median(v['latency_reduction_percent'] for v in rows.values()),
        memory_reduction_range_percent=[min(v['memory_reduction_percent'] for v in rows.values()),max(v['memory_reduction_percent'] for v in rows.values())])
    (FOLDER/'comparison_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'research/COMPACT_DEPLOYMENT_20261010.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('costs','portable','cached_constructor','full_accuracy')}))


if __name__=='__main__':main()
