"""Frozen Geov2-24 shared-canvas ablations; masks only score predictions."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.geov2_wide_optimized import OptimizedWideReadout, WideReadoutConfig
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_fixed05_components import ARMS, Experiment
from eval_geov2_g0_sweep import metric, write


class HalfIdentity:
    """Apply .5I without allocating a matrix in ordinary-fusion cost runs."""
    def double(self):
        return self

    def __matmul__(self, value):
        return value * .5


class SharedExperiment(Experiment):
    def __init__(self, protocol, methods):
        super().__init__(dict(protocol, g0=.5), methods)
        self.model.profile = replace(self.model.profile, coupling=protocol['reference_g0'])
        self.shared = OptimizedWideReadout(self.model, WideReadoutConfig(direct=False))
        if self.model.vip.backbone.model.visual_model.backbone is not self.model.geometry.backbone.model.visual_model.backbone:
            raise RuntimeError('Backbone storage was not shared.')

    def read_cell(self, source, cell, cache, *, implicit_uniform=False):
        method, gain = cell['method'], cell['g0']
        previous = self.model.profile
        if method == 'UniformBlend':
            gain = 2 * cell['alpha']
        self.model.profile = replace(previous, coupling=previous.coupling if gain is None else gain)
        changed = None
        try:
            if method == 'UniformBlend' and cell['alpha'] in (0., 1.):
                # Match the endpoint renderer's FP32 arithmetic exactly.
                return self.read(source, 'LocalOnly' if cell['alpha'] == 0. else 'WideOnly', cache)
            if method in ('UniformBlend', 'DiagonalH'):
                changed_local = []
                for tile in source['local']:
                    h = tile['operator']
                    if method == 'UniformBlend':
                        operator = HalfIdentity() if implicit_uniform else torch.eye(len(h), device=h.device, dtype=h.dtype)*.5
                    else:
                        operator = torch.diag(h.diagonal())
                    changed_local.append(dict(tile, operator=operator))
                changed = dict(source, local=changed_local)
            return self.read(changed or source, 'Full' if changed is not None else method, cache)
        finally:
            self.model.profile = previous

    def predict_cell(self, image, cell):
        method = cell['method']
        kind, _, fusion = ARMS['Full' if method in ('UniformBlend', 'DiagonalH') else method]
        source = self.source(image, kind, solve=fusion == 'coupled' and method != 'UniformBlend')
        return self.read_cell(source, cell, {}, implicit_uniform=method == 'UniformBlend')


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    root = Path(args.experiment)
    protocol = json.loads((root/'protocol.json').read_text())
    record = protocol['datasets'][args.dataset]
    p = dict(protocol, **record, dataset=args.dataset)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard.')
    cells = record['cells']
    if args.mode == 'cost':
        cells = [c for c in cells if c['key'] == args.cell]
        if len(cells) != 1:
            raise ValueError('Unknown cost cell.')
    methods = list(dict.fromkeys(c['method'] for c in cells if c['method'] in ARMS))
    if any(c['method'] in ('UniformBlend', 'DiagonalH') for c in cells) and 'Full' not in methods:
        methods.append('Full')
    out = root/args.mode/args.dataset/(args.cell if args.mode == 'cost' else f's{args.shard_index}')
    if out.exists():
        raise RuntimeError('Preserve existing output: '+str(out))
    out.mkdir(parents=True)
    exp = SharedExperiment(p, methods)
    m = exp.model
    args.data_root, args.sample_seed, args.vdd_ontology = record['data_root'], 20260923, 'official'
    loader_args = argparse.Namespace(**vars(args))
    loader_args.mode, loader_args.num_shards, loader_args.shard_index = 'full', 1, 0
    entry = dict(m.config, banks={'original':next(iter(m.vocabulary['banks'].values()))})
    validation, _, images, masks, names_map = load_samples(loader_args, entry)
    if len(validation) != record['total_images']:
        raise RuntimeError('Coverage changed.')
    common = dict(dataset=args.dataset, reference_g0=record['reference_g0'], profile=m.profile.record(),
        config=m.config, cache_identity=m.cache_identity, alias_counts=exp.alias_counts,
        architecture='DonorBefore + QuerySlot24 + shared_canvas + fixed structural writer',
        adaptive_gain_enabled=False, cells=cells, cpu_threads=1)
    if args.mode == 'cost':
        gc.collect()
        torch.cuda.empty_cache()
        lookup = {s.key:s for s in validation}
        rows = []
        for key in record['cost_keys']:
            image = images(lookup[key])
            if cells[0]['method'] == 'UniformBlend':
                source = exp.source(image, 'full')
                dense = exp.read_cell(source, cells[0], {})
                implicit = exp.read_cell(source, cells[0], {}, implicit_uniform=True)
                if not np.array_equal(dense, implicit):
                    raise RuntimeError('Implicit ordinary fusion changed the prediction.')
                del source, dense, implicit
                gc.collect()
                torch.cuda.empty_cache()
            for _ in range(2):
                exp.predict_cell(image, cells[0])
            times, allocated, reserved = [], [], []
            for _ in range(7):
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = exp.predict_cell(image, cells[0])
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                allocated.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            if result.dtype != np.uint8 or result.shape != tuple(image.shape[-2:]):
                raise RuntimeError('Invalid cost prediction.')
            rows.append(dict(key=key, milliseconds=times, peak_allocated_mib=max(allocated), peak_reserved_mib=max(reserved)))
        write(out/'results.json', dict(status='complete', **common, images=rows,
            median_ms=float(np.median([t for r in rows for t in r['milliseconds']])),
            peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
            peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows), target_masks_loaded=False,
            implicit_uniform_verified=cells[0]['method'] == 'UniformBlend',
            gpu=torch.cuda.get_device_name(),
            boundary='CPU RGB tensor to original-size CPU uint8 labels; excludes decode/init/offline text; seven fixed images, two warmups and seven synchronized repeats. Beta-zero still computes unused routing.'))
        return
    samples = validation[:1] if args.mode == 'smoke' else validation[args.shard_index::args.num_shards]
    names = names_map[args.dataset]
    matrices = np.zeros((len(cells), len(names), len(names)), np.int64)
    tps, predicted, targets, keys = [], [], [], []
    start = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = images(sample)
        predictions = {}
        for kind in dict.fromkeys(ARMS[method][0] for method in methods):
            source = exp.source(image, kind)
            caches = {}
            for cell in cells:
                method = cell['method']
                arm = 'Full' if method in ('UniformBlend', 'DiagonalH') else method
                if ARMS[arm][0] != kind:
                    continue
                cache = caches.setdefault(ARMS[arm][1], {})
                predictions[cell['key']] = exp.read_cell(source, cell, cache)
            if args.mode == 'smoke' and kind == 'full':
                import dinotool.query_slot_memory as qs
                original = qs.query_slot_wide_observations
                qs.query_slot_wide_observations = lambda *_a, **_k:source
                try:
                    direct = m.predict(image)
                finally:
                    qs.query_slot_wide_observations = original
                full = 'Full__g'+format(record['reference_g0'], 'g')
                check = dict(same_observation_deployment_pixels=int((direct != predictions[full]).sum()),
                    weighted_half_vs_equal_pixels=int((predictions['UniformBlend__a0.5'] != predictions['EqualLogitFusion']).sum()),
                    weighted_zero_vs_local_pixels=int((predictions['UniformBlend__a0'] != predictions['LocalOnly']).sum()),
                    weighted_one_vs_wide_pixels=int((predictions['UniformBlend__a1'] != predictions['WideOnly']).sum()))
                if any(check.values()):
                    raise RuntimeError('Readout parity failed: '+str(check))
                write(out/'prediction_check.json', check)
                print(json.dumps(check), flush=True)
            del source, caches
        if set(predictions) != {c['key'] for c in cells}:
            raise RuntimeError('Missing grid cell.')
        # Labels cannot affect any prediction or arm choice above.
        target = masks(sample, args.dataset, tuple(image.shape[-2:]))
        cms = np.stack([confusion(predictions[c['key']], target, len(names)) for c in cells])
        if not np.all(cms.sum(-1) == cms[:1].sum(-1)):
            raise RuntimeError('Target support differs.')
        matrices += cms
        tps.append(np.diagonal(cms, axis1=-2, axis2=-1).copy())
        predicted.append(cms.sum(-2))
        targets.append(cms[0].sum(-1))
        keys.append(sample.key)
        if number == 1 or number % 25 == 0 or number == len(samples):
            progress = dict(dataset=args.dataset, processed=number, total=len(samples),
                shard=args.shard_index, wall_seconds=time.perf_counter()-start)
            write(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    np.savez_compressed(out/'statistics.npz', sample_keys=np.asarray(keys), tp=np.stack(tps),
        predicted=np.stack(predicted), target=np.stack(targets))
    write(out/'results.json', dict(status='complete', **common, processed_images=len(keys),
        global_sample_count=len(validation), sample_keys=keys, shard_index=args.shard_index,
        num_shards=args.num_shards, class_names=names, masks_used_only_for_scoring=True,
        metrics={c['key']:dict(mean_iou_percent=metric(cm), confusion_matrix=cm.tolist()) for c, cm in zip(cells, matrices)},
        wall_seconds=time.perf_counter()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', choices=('smoke', 'full', 'cost'), required=True)
    parser.add_argument('--cell', default='Full__g0.5')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    main(parser.parse_args())
