"""Bounded source-first cross-support alias diagnosis on the fixed96 RS panel."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dinotool.bounded_alias_path_audit import capture, assemble, donor_targets
from dinotool.bounded_patch_only import PRIMARY
from dinotool.cross_support_alias_diagnostic import IMPLEMENTATION, PROTOCOL, source_diagnostic, semantic_audit
from dinotool.model import checkpoint_manifest
from eval_bounded_alias_path_audit import confusion
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


REFERENCE = Path('/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/bounded_alias_path_audit_r2_20261006/full')


@torch.inference_mode()
def main(args):
    output, samples, load_image, load_mask, geometry, banks, vip, queries, checkpoints, old = panel_inputs(args)
    if args.num_shards != 1 or args.shard_index:
        raise ValueError('One bounded diagnostic worker per domain required.')
    expected = json.loads((REFERENCE/args.dataset/'s0/results.json').read_text())
    keys = [s.key for s in samples]
    if (keys != expected['signature']['sample_keys'] or old['signature']['vocabulary'] != expected['signature']['vocabulary']
            or checkpoint_manifest(checkpoints) != expected['signature']['checkpoints']):
        raise RuntimeError('Panel or frozen identities differ.')
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output/'pre_mask').mkdir()
    started = time.perf_counter()
    rows, per_image, diagnostics = [], {p: [] for p in banks}, []
    signature = dict(implementation=IMPLEMENTATION, protocol=PROTOCOL, sample_keys=keys,
        vocabulary=old['signature']['vocabulary'], classes=expected['signature']['classes'],
        checkpoints=checkpoint_manifest(checkpoints), geometry=asdict(geometry.config))
    torch.cuda.reset_peak_memory_stats()
    with np.load(REFERENCE/args.dataset/'s0/per_image_audit.npz', allow_pickle=False) as prior:
        if prior['sample_keys'].tolist() != keys:
            raise RuntimeError('Ordered source/audit keys differ.')
        for number, sample in enumerate(samples):
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            sources = capture(image, geometry, banks, vip, queries, retain_alias_evidence=True)
            pending_by_protocol, image_rows = {}, []
            predictions = {p: assemble(source) for p, source in sources.items()}
            for p, source in sources.items():
                records, pending = source_diagnostic(source, banks[p].features)
                for record in records:
                    record.update(sample_key=sample.key, sample_index=number, protocol=p)
                image_rows.extend(records)
                pending_by_protocol[p] = pending
            save(output/'pre_mask'/f'{number:03d}.json', dict(sample_key=sample.key, records=image_rows,
                target_mask_loaded=False, source_statistics_complete=True))
            for p, source in sources.items():
                target = load_mask(sample, p, source.output_size)
                cm = confusion(target, predictions[p], banks[p].class_count)
                if not np.array_equal(cm, prior[p+'__'+PRIMARY][number]):
                    raise RuntimeError('Original per-image confusion changed: '+sample.key+'/'+p)
                per_image[p].append(cm)
                labels = [donor_targets(target, tile, source).cpu().numpy() for tile in source.tiles]
                semantic_audit(pending_by_protocol[p], labels)
            rows.extend(image_rows)
            diagnostics.append({p: source.diagnostics for p, source in sources.items()})
            save(output/'results.json', dict(status='running', processed_images=number+1, total_images=len(samples),
                signature=signature, original_per_image_confusions_exact=True,
                source_statistics_persisted_before_masks=True, diagnostic_only=True,
                wall_seconds=time.perf_counter()-started))
            print(json.dumps(dict(dataset=args.dataset, sample=sample.key, processed=number+1,
                tested=sum(r['status'] == 'tested' for r in image_rows), rows=len(image_rows))), flush=True)
            del sources, predictions, pending_by_protocol, image_rows
    save(output/'records.json', rows)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{p: np.stack(v) for p, v in per_image.items()})
    save(output/'results.json', dict(status='complete', processed_images=len(samples), total_images=len(samples),
        signature=signature, original_per_image_confusions_exact=True, source_statistics_persisted_before_masks=True,
        diagnostic_only=True, records=len(rows), diagnostics=diagnostics,
        wall_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
        **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--mode', default='full', choices=('full',))
    main(parser.parse_args())
