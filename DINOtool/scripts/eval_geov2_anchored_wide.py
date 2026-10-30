"""Matched anchored-slot experiment; targets are used only after prediction."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.anchored_slot_memory import anchored_slot_wide_observations
from dinotool.efficient_deployment import reconstruction
from dinotool.geov2_wide_optimized import factor_correction
from dinotool.anchored_infra import (
    AnchoredInfraConfig, cast_anchored_visual_weights, configure_fast_math,
    describe_config, factor_solver, prepare_streamed_source,
)
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import metric, write
from eval_geov2_multidataset_components import HalfIdentity, SharedExperiment


MODES = dict(Coordinate='coordinate', Legacy='legacy', Anchored='anchored',
    NoPrior='no_prior', NoReference='no_reference', AbsoluteMix='absolute_mix',
    Shuffled='shuffled', UniformBlend='anchored')


class AnchoredExperiment(SharedExperiment):
    def __init__(self, protocol, infra=None):
        super().__init__(protocol, ['Full'])
        self.infra = infra or AnchoredInfraConfig()
        self.infra.validate()
        self.infra_previous = configure_fast_math(self.infra.fast_math)
        # Cast both frozen visual heads after they are constructed.  The
        # backbone can be shared later by an execution wrapper; the helper
        # deduplicates shared parameters and leaves LayerNorm/RoPE/relations
        # in their original dtype.
        self.weight_stats = cast_anchored_visual_weights(
            self.model, self.infra.weight_dtype)
        self.initialization_peak_allocated_mib = torch.cuda.max_memory_allocated()/1048576

    def sources(self, image, methods, *, solve=True, audit=False):
        m = self.model
        modes = list(dict.fromkeys(MODES[method] for method in methods))
        solver = (factor_solver(self.infra.solver_dtype)
                  if solve and self.infra.solver_dtype != 'float64' else reconstruction) if solve else lambda _a, _v:(None, None)
        wide_policy = self.infra.wide_policy or m.profile.wide_policy
        sources = anchored_slot_wide_observations(
            image, m.compact_geometry, m.query_slot_observer, m.strengths,
            wide_policy=wide_policy, temperature=.1, beta=.5,
            reconstruction=solver, modes=modes, audit=audit)
        if self.infra.stream_logits and len(modes) == 1:
            mode = modes[0]
            source = sources[mode]
            bank = m.profile.bank
            source = prepare_streamed_source(
                source, reader=m.reader, profile=m.profile, bank_name=bank,
                wide_scorer=m.wide_scorer,
                local_scorer=m.efficient_local_scorers[bank])
            sources[mode] = source
        return sources

    def render(self, source, cell, cache, *, implicit_uniform=False):
        m = self.model
        previous = m.profile
        method = cell['method']
        gain = 2*cell['alpha'] if method == 'UniformBlend' else cell['g0']
        m.profile = replace(previous, coupling=gain)
        try:
            if method == 'UniformBlend':
                if cell['alpha'] in (0., 1.):
                    return self.read(source, 'LocalOnly' if cell['alpha'] == 0. else 'WideOnly', cache)
                local = []
                for tile in source['local']:
                    operator = HalfIdentity() if implicit_uniform else torch.eye(
                        len(tile['operator']), device=tile['operator'].device,
                        dtype=tile['operator'].dtype)*.5
                    local.append(dict(tile, operator=operator))
                source = dict(source, local=local)
            return self.read(source, 'Full', cache)
        finally:
            m.profile = previous

    def predict_cell(self, image, cell):
        sources = self.sources(image, [cell['method']], solve=cell['method'] != 'UniformBlend')
        return self.render(sources[MODES[cell['method']]], cell, {},
            implicit_uniform=cell['method'] == 'UniformBlend')


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
    methods = list(dict.fromkeys(c['method'] for c in cells))
    out = root/args.mode/args.dataset/(cells[0]['method'] if args.mode == 'cost' else f's{args.shard_index}')
    if out.exists():
        raise RuntimeError('Preserve existing output: '+str(out))
    out.mkdir(parents=True)
    infra = AnchoredInfraConfig(
        solver_dtype=args.solver_dtype,
        stream_logits=not args.no_stream_logits,
        fast_math=not args.no_fast_math,
        weight_dtype=args.weight_dtype,
        wide_policy=args.wide_policy,
    )
    exp = AnchoredExperiment(p, infra)
    m = exp.model
    loader_args = argparse.Namespace(dataset=args.dataset, data_root=record['data_root'],
        sample_seed=20260923, vdd_ontology='official', mode='full', num_shards=1, shard_index=0)
    entry = dict(m.config, banks={'original':next(iter(m.vocabulary['banks'].values()))})
    validation, _, images, masks, names_map = load_samples(loader_args, entry)
    if len(validation) != record['total_images']:
        raise RuntimeError('Validation coverage changed.')
    expected = json.loads((root/'expected_keys.json').read_text())[args.dataset]
    if [s.key for s in validation] != expected:
        raise RuntimeError('Validation ordering or coverage changed.')
    common = dict(dataset=args.dataset, reference_g0=record['reference_g0'], profile=m.profile.record(),
        config=m.config, cache_identity=m.cache_identity, alias_counts=exp.alias_counts,
        architecture='Frozen DonorBefore + matched 24-slot head + anchored residual read + original structural writer',
        cells=cells, cpu_threads=1, adaptive_gain_enabled=False,
        initialization_peak_allocated_mib=exp.initialization_peak_allocated_mib,
        anchored_infra=describe_config(exp.infra), weight_storage=exp.weight_stats)
    lookup = {s.key:s for s in validation}
    if args.mode == 'cost':
        gc.collect()
        torch.cuda.empty_cache()
        rows = []
        for key in record['cost_keys']:
            image = images(lookup[key])
            if cells[0]['method'] == 'UniformBlend':
                source = exp.sources(image, ['Anchored'])['anchored']
                dense = exp.render(source, cells[0], {})
                implicit = exp.render(source, cells[0], {}, implicit_uniform=True)
                if not np.array_equal(dense, implicit):
                    raise RuntimeError('Implicit fusion changed output.')
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
            rows.append(dict(key=key, milliseconds=times, peak_allocated_mib=max(allocated),
                peak_reserved_mib=max(reserved)))
            print(json.dumps(dict(dataset=args.dataset, method=cells[0]['method'], cost_images=len(rows))), flush=True)
        write(out/'results.json', dict(status='complete', **common, images=rows,
            median_ms=float(np.median([t for r in rows for t in r['milliseconds']])),
            peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
            peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows), target_masks_loaded=False,
            gpu=torch.cuda.get_device_name(),
            boundary='CPU RGB tensor to original-size CPU uint8 labels. Decode/init/offline text excluded. Single mode/fresh process; 7 fixed images, 2 warmups and 7 synchronized repeats. Coordinate skips unused routing.'))
        return
    keys = record['screen_keys'] if args.mode == 'screen' else expected[:1] if args.mode == 'smoke' else expected
    keys = keys[args.shard_index::args.num_shards]
    names = names_map[args.dataset]
    matrices = np.zeros((len(cells), len(names), len(names)), np.int64)
    tps, predicted, targets, processed = [], [], [], []
    start = time.perf_counter()
    for number, key in enumerate(keys, 1):
        image = images(lookup[key])
        sources = exp.sources(image, methods, audit=args.mode == 'smoke')
        caches = {mode:{} for mode in sources}
        predictions = {c['key']:exp.render(sources[MODES[c['method']]], c,
            caches[MODES[c['method']]]) for c in cells}
        if args.mode == 'smoke':
            import dinotool.query_slot_memory as qs
            original = qs.query_slot_wide_observations
            qs.query_slot_wide_observations = lambda *_a, **_k:sources['legacy']
            try:
                previous = m.profile
                m.profile = replace(previous, coupling=record['reference_g0'])
                direct = m.predict(image)
            finally:
                m.profile = previous
                qs.query_slot_wide_observations = original
            legacy_key = 'Legacy__g'+format(record['reference_g0'], 'g')
            check = dict(same_observation_deployment_pixels=int((direct != predictions[legacy_key]).sum()),
                local_objects_shared=all(s['local'] is sources['legacy']['local'] for s in sources.values()),
                reader_audit=sources['anchored'].get('anchored_audit'))
            if check['same_observation_deployment_pixels'] or not check['local_objects_shared']:
                raise RuntimeError('Matched execution failed: '+str(check))
            write(out/'prediction_check.json', check)
            print(json.dumps(check), flush=True)
        del sources, caches
        # No target participates in feature construction, routing or prediction.
        target = masks(lookup[key], args.dataset, tuple(image.shape[-2:]))
        cms = np.stack([confusion(predictions[c['key']], target, len(names)) for c in cells])
        if not np.all(cms.sum(-1) == cms[:1].sum(-1)):
            raise RuntimeError('Target supports differ.')
        matrices += cms
        tps.append(np.diagonal(cms, axis1=-2, axis2=-1).copy())
        predicted.append(cms.sum(-2))
        targets.append(cms[0].sum(-1))
        processed.append(key)
        if number == 1 or number % 25 == 0 or number == len(keys):
            progress = dict(dataset=args.dataset, processed=number, total=len(keys),
                shard=args.shard_index, wall_seconds=time.perf_counter()-start)
            write(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    np.savez_compressed(out/'statistics.npz', sample_keys=np.asarray(processed),
        tp=np.stack(tps), predicted=np.stack(predicted), target=np.stack(targets))
    write(out/'results.json', dict(status='complete', **common, processed_images=len(processed),
        global_sample_count=len(validation), sample_keys=processed, shard_index=args.shard_index,
        num_shards=args.num_shards, class_names=names, masks_used_only_for_scoring=True,
        metrics={c['key']:dict(mean_iou_percent=metric(cm), confusion_matrix=cm.tolist()) for c, cm in zip(cells, matrices)},
        wall_seconds=time.perf_counter()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', choices=('smoke', 'screen', 'full', 'cost'), required=True)
    parser.add_argument('--cell', default='Anchored__g0.5')
    parser.add_argument('--solver-dtype', choices=('float32', 'float64'), default='float32',
        help='Cholesky factor dtype for the H action; float64 is compatibility mode.')
    parser.add_argument('--no-stream-logits', action='store_true',
        help='Keep the historical high-dimensional feature lists.')
    parser.add_argument('--weight-dtype', choices=('none', 'bfloat16'), default='none',
        help='Optional BF16 storage for frozen visual Linear/Conv weights.')
    parser.add_argument('--wide-policy', default=None,
        help='Optional long-edge override such as long384; unset uses the frozen YAML.')
    parser.add_argument('--no-fast-math', action='store_true')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    main(parser.parse_args())
