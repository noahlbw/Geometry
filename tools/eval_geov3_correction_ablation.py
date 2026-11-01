"""Geov3 correction operator ablation: direct, Gram and spectral write-back."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from eval_geov3_path_ablation import build
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import metric, write
from dinotool.geov2_wide_optimized import factor_reconstruction, factor_reconstruction_batch, factor_correction
from dinotool.bounded_physical_coupling import resize_geometry
from dinotool.compact_deployment import _iter_local_observations
from dinotool.efficient_deployment import restore_and_finalize, write_dense
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator

ARMS = ('LocalOnly', 'DirectFusion', 'GramPropagation', 'SpectralFilter')

@torch.inference_mode()
def predict_all(m, opt, image, gain):
    resized = resize_geometry(image); h, w = resized.shape[-2:]
    memory = opt.wide_memory(image)
    tiles = list(_iter_local_observations(
        resized, (h, w), m.compact_geometry, ('original',),
        reconstruction=lambda r, v: factor_reconstruction(r, v, torch.float32),
        include_raw=True, batch_size=8,
        reconstruction_batch=lambda r, v: factor_reconstruction_batch(r, v, torch.float32)))
    wide_features = opt.tile_wide_features_batch(tiles, memory, (h, w))
    p = m.profile; bank = m.banks[p.bank]; reader = m.reader
    accum = {x: DeviceProbabilityAccumulator(bank.class_count, h, w, reader.blend.device) for x in ARMS}
    try:
        eye_cache = {}
        for tile, wf in zip(tiles, wide_features):
            wide = opt.scorer(wf[None] if wf.ndim == 2 else wf).float()
            alias = tile['features']['original'].float() @ reader.text[p.bank].T
            local = m.efficient_local_scorers[p.bank].uniform_reference(alias)
            if p.background_rule == 'union_max' and reader.background is not None:
                local[:, reader.background] = alias[:, reader.members[p.bank]].amax(-1)
            if p.background_rule == 'residual_max':
                local[:, reader.background] = (tile['features']['original'].float() @ reader.residual_text.T).amax(-1)
            local = local / p.temperature
            diff = wide.double() - local.double()
            factor, valid = tile['operator']
            n = factor.shape[-1]
            eye = eye_cache.setdefault(n, torch.eye(n, device=factor.device, dtype=factor.dtype))
            gram = factor @ factor.transpose(-1, -2) - eye
            gram = gram.masked_fill(~valid[:, None], 0).masked_fill(~valid[None, :], 0)
            gram_diff = torch.matmul(gram[None].double(), diff)
            spectral = factor_correction(tile['operator'], diff)
            logits = {
                'LocalOnly': local.double(),
                'DirectFusion': local.double() + float(gain) * diff,
                'GramPropagation': local.double() + float(gain) * gram_diff,
                'SpectralFilter': local.double() + float(gain) * spectral.double(),
            }
            ah, aw = min(512, h - tile['top']), min(512, w - tile['left'])
            for name, value in logits.items():
                write_dense(accum[name], value, bank.class_count, ah, aw, reader.blend, tile['left'], tile['top'])
        return {name: restore_and_finalize(acc.probabilities / acc.normalizer[None], tuple(image.shape[-2:]), m.config['background_bias'], m.config['background_threshold'], reader.background) for name, acc in accum.items()}
    finally:
        for acc in accum.values(): acc.__exit__(None, None, None)

@torch.inference_mode()
def main(a):
    root = json.loads(Path(a.protocol).read_text()); rec = dict(root['datasets'][a.dataset])
    out = Path(a.output) / a.mode / a.dataset
    if out.exists(): raise RuntimeError('preserve existing output ' + str(out))
    out.mkdir(parents=True)
    m, opt = build(root, rec)
    ns = argparse.Namespace(**vars(a)); ns.data_root = rec['loader']['data_root']; ns.sample_seed = 20260923; ns.vdd_ontology = 'official'; ns.mode = 'full'; ns.num_shards = 1; ns.shard_index = 0
    entry = dict(m.config, banks={'original': next(iter(m.vocabulary['banks'].values()))})
    validation, _, images, masks, names_map = load_samples(ns, entry)
    if len(validation) != rec['total_images']: raise RuntimeError('coverage mismatch')
    names = names_map[a.dataset]; samples = validation[:1] if a.mode == 'smoke' else validation
    mats = {x: np.zeros((len(names), len(names)), np.int64) for x in ARMS}; start = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = images(sample); preds = predict_all(m, opt, image, rec['gain']); target = masks(sample, a.dataset, tuple(image.shape[-2:]))
        for arm in ARMS: mats[arm] += confusion(preds[arm], target, len(names))
        if number == 1 or number % 25 == 0 or number == len(samples):
            write(out / 'progress.json', dict(dataset=a.dataset, processed=number, total=len(samples), wall_seconds=time.perf_counter() - start)); print(json.dumps(dict(dataset=a.dataset, processed=number, total=len(samples))), flush=True)
    write(out / 'results.json', dict(status='complete', implementation='geov3_correction_ablation_20261101', dataset=a.dataset, arms=ARMS, model='Geov3 QuerySlot24 + DonorBefore + Anchored', g0=rec['gain'], processed_images=len(samples), global_sample_count=len(validation), class_names=names, metrics={x: dict(mean_iou_percent=metric(mats[x]), confusion_matrix=mats[x].tolist()) for x in ARMS}, masks_used_only_for_scoring=True, wall_seconds=time.perf_counter() - start))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--protocol', required=True); p.add_argument('--dataset', required=True); p.add_argument('--mode', choices=('smoke', 'full'), required=True); p.add_argument('--output', required=True); main(p.parse_args())
