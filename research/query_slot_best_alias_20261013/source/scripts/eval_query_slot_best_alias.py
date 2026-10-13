"""Evaluate both enabled modules with retained best aliases; no inference labels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.frozen_deployment import FrozenGeometry
from dinotool.response_geometry_deployment import EfficientResponseGeometry
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


@torch.inference_mode()
def main(args):
    root = Path(args.suite_root)
    suffix = '_s'+str(args.shard_index) if args.mode == 'full' else ''
    output = root/args.mode/(args.dataset+suffix+'.json')
    if output.exists():
        raise RuntimeError('Preserve completed result.')
    entry = read(root/'protocol.json')['datasets'][args.dataset]
    config = read(root/entry['config'])
    vocabulary = read((root/entry['config']).parent/config['vocabulary'])
    model = FrozenGeometry(root/entry['config'], checkpoint_dir=args.checkpoint_dir,
        dinov3_repo=args.dinov3_repo, upstream_root=args.upstream_root,
        text_cache=root/'text_cache'/f'{args.dataset}.pt', efficient=True,
        wide_branch='query_slot', query_slot_capacity=24)
    assert model.strengths == ('original',)
    assert isinstance(model.compact_geometry, EfficientResponseGeometry)
    assert model.wide_branch == 'query_slot' and model.local_branch == 'donor_before'
    assert all(branch.backbone.model.text_model is None for branch in (model.geometry, model.vip))
    entry = dict(entry, **config, banks={'original': next(iter(vocabulary['banks'].values()))})
    args.data_root = entry['data_root']
    args.sample_seed, args.vdd_ontology = 20260923, 'official'
    validation, samples, images, masks, names = load_samples(args, entry)
    if len(validation) != entry['total_images']:
        raise RuntimeError('Full validation size differs.')
    metadata = dict(dataset=args.dataset, config=model.config, cache_identity=model.cache_identity,
        query_slot_capacity=24, query_slot_beta=.5, query_slot_temperature=.1,
        both_modules_active=True, donor_before_strength='original',
        alias_source=entry['alias_source'], original_profile=entry['original_profile'])
    def predict(image, protocol):
        return model.predict(image, protocol=protocol if args.dataset == 'loveda' else None)
    if args.mode in ('cost', 'smoke'):
        lookup = {sample.key: sample for sample in validation}
        keys = read(root/'cost_keys.json')[args.dataset]
        if args.mode == 'smoke':
            keys = keys[:1]
        rows = []
        for key in keys:
            image = images(lookup[key])
            for protocol in names:
                for _ in range(2 if args.mode == 'cost' else 0):
                    predict(image, protocol)
                times, allocated, reserved = [], [], []
                for _ in range(7 if args.mode == 'cost' else 1):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = predict(image, protocol)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-started)*1000.)
                    allocated.append(torch.cuda.max_memory_allocated()/1048576.)
                    reserved.append(torch.cuda.max_memory_reserved()/1048576.)
                if result.shape != tuple(image.shape[-2:]) or result.dtype != np.uint8:
                    raise RuntimeError('Output shape/dtype differs.')
                rows.append(dict(key=key, protocol=protocol, shape=list(image.shape[-2:]),
                    milliseconds=times, peak_allocated_mib=max(allocated), peak_reserved_mib=max(reserved)))
        costs = {p: dict(median_ms=float(np.median([t for row in rows if row['protocol']==p for t in row['milliseconds']])),
            peak_allocated_mib=max(row['peak_allocated_mib'] for row in rows if row['protocol']==p),
            peak_reserved_mib=max(row['peak_reserved_mib'] for row in rows if row['protocol']==p)) for p in names}
        write(output, dict(status='complete', **metadata, costs=costs, images=rows,
            target_masks_loaded=False, boundary='CPU RGB to CPU uint8 original-size IDs; '
                'decode/init/offline text excluded. Two warmups, seven synchronized repeats, seven fixed inputs.'))
        print(json.dumps(dict(dataset=args.dataset, mode=args.mode, costs=costs)), flush=True)
        return
    matrices = {p: np.zeros((len(n),len(n)),np.int64) for p,n in names.items()}
    per_image = {p+'__'+kind: [] for p in names for kind in ('tp','target','predicted')}
    keys = []
    started = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = images(sample)
        for protocol, class_names in names.items():
            actual = predict(image, protocol)
            target = masks(sample, protocol, tuple(image.shape[-2:]))
            cm = confusion(actual, target, len(class_names))
            matrices[protocol] += cm
            per_image[protocol+'__tp'].append(cm.diagonal().copy())
            per_image[protocol+'__target'].append(cm.sum(1))
            per_image[protocol+'__predicted'].append(cm.sum(0))
        keys.append(sample.key)
        if number == 1 or number % 25 == 0 or number == len(samples):
            progress = dict(dataset=args.dataset, processed=number, total=len(samples),
                shard_index=args.shard_index, num_shards=args.num_shards)
            write(output.with_suffix('.progress.json'), progress)
            print(json.dumps(progress), flush=True)
    if not keys or len(keys) != len(set(keys)):
        raise RuntimeError('Empty or duplicated shard coverage.')
    packed = {key: np.stack(value) for key,value in per_image.items()}
    for p,cm in matrices.items():
        if not np.array_equal(packed[p+'__tp'].sum(0),cm.diagonal()):
            raise RuntimeError('Per-image true-positive sum differs.')
        if not np.array_equal(packed[p+'__target'].sum(0),cm.sum(1)):
            raise RuntimeError('Per-image target sum differs.')
    npz = output.with_suffix('.npz')
    np.savez_compressed(npz, keys=np.asarray(keys), **packed)
    write(output, dict(status='complete', **metadata, processed_images=len(keys), total_images=len(validation),
        num_shards=args.num_shards, shard_index=args.shard_index, sample_keys=keys,
        metrics={p: summary(cm,names[p],0) for p,cm in matrices.items()},
        per_image_statistics=str(npz), per_image_sums_verified=True, wall_seconds=time.perf_counter()-started))
    print(json.dumps(dict(dataset=args.dataset, mode='full', status='complete', metrics={
        p:summary(cm,names[p],0)['mean_iou_percent'] for p,cm in matrices.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('suite-root','dataset','checkpoint-dir','dinov3-repo','upstream-root'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('smoke','cost','full'), required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
