"""Paired fixed/domain-transfer/adaptive full evaluation from shared observations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

import eval_development_readout as base
from dinotool.development_readout import Profile
from dinotool.evidence_adaptive_readout import IMPLEMENTATION, competition_scores, adaptive_coupled, residual_prediction
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.inference import hann_blend_window
from eval_geometry_vip_reliability import sample_broad, summary
from eval_rival_fine_full import save, frozen_state, check_frozen

METHODS = ('Reference', 'DomainShared', 'LeaveDomainOut', 'AdaptiveCompetition', 'AdaptiveResidual')


@torch.inference_mode()
def adaptive_probability(source, profile, bank, broad):
    h, w = source['size']
    text = F.normalize(bank.features.float(), dim=-1)
    blend = torch.from_numpy(hann_blend_window(512)).to(text.device)
    gains = []
    with DeviceProbabilityAccumulator(bank.class_count, h, w, text.device) as accumulator:
        for tile in source['local']:
            alias = tile['features'][profile.strength].float() @ text.T
            local = competition_scores(alias, bank) / profile.temperature
            top, left = tile['top'], tile['left']
            wide = sample_broad(broad, top, left, h, w).reshape_as(local)
            logits, gain = adaptive_coupled(local, wide, tile['operator'], tile['relation'])
            gains.append(gain)
            dense = F.interpolate(logits.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
            ah, aw = min(512, h-top), min(512, w-left)
            accumulator.add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
        result = F.interpolate((accumulator.probabilities / accumulator.normalizer[None])[None],
                               source['output_size'], mode='bilinear', align_corners=False)[0]
    values = torch.cat(gains)
    return result, dict(gain_mean=float(values.mean()), gain_min=float(values.min()), gain_max=float(values.max()))


@torch.inference_mode()
def main(args):
    root, output = Path(args.suite_root), Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Existing output; preserve it.')
    manifest = json.loads((root/'protocol.json').read_text())
    entry = manifest['datasets'][args.dataset]
    validation, samples, load_image, load_mask = base.load_samples(args, entry)
    candidates = (Profile(), Profile(**entry['shared_profile']), Profile(**entry['transfer_profile']))
    needed = tuple(dict.fromkeys(p.bank for p in candidates))
    geometry, vip, banks, queries, checkpoints = base.load_models(args, entry, needed)
    state = frozen_state(geometry, vip)
    names, classes = banks['original'].class_names, banks['original'].class_count
    matrices = {m: np.zeros((classes, classes), np.int64) for m in METHODS}
    per_image = {m: [] for m in METHODS}
    costs, diagnostics = [], []
    signature = dict(implementation=IMPLEMENTATION, dataset=args.dataset, methods=METHODS,
        classes={args.dataset:names}, gear=dict(core=manifest['retained_core'], cap=[4,4,0]), competitive=None,
        vocabulary=entry['banks'], checkpoints=base.checkpoint_manifest(checkpoints),
        global_sample_count=len(validation), global_sample_keys_sha256=base.digest([s.key for s in validation]),
        sample_keys=[s.key for s in samples], sample_keys_sha256=base.digest([s.key for s in samples]),
        num_shards=args.num_shards, shard_index=args.shard_index, config=vars(args),
        profiles=[p.record() for p in candidates], source_label_development=manifest['source_label_development'])
    output.mkdir(parents=True)
    started = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        source = base.observations(image, geometry, vip, tuple(dict.fromkeys(p.strength for p in candidates)))
        # Derive the same relation from the existing reconstruction operator.
        # Store the actual Geometry relation by observing the prepare output below.
        cache, wide_cache, probabilities = {}, {}, {}
        for method, p in zip(METHODS[:3], candidates):
            key = p.bank, p.tau, p.tem
            if key not in wide_cache:
                wide_cache[key] = base.wide_scores(source, queries[p.bank], p.tau, p.tem)
            probabilities[method] = base.probabilities(source,p,banks[p.bank],wide_cache[key],cache)
        p = candidates[1]
        q, diagnostic = adaptive_probability(source,p,banks[p.bank],wide_cache[p.bank,p.tau,p.tem])
        probabilities['AdaptiveCompetition'] = q
        adaptive_bg, bg_diagnostic = residual_prediction(q,entry['background_index'])
        diagnostic.update(bg_diagnostic)
        diagnostics.append(diagnostic)
        if not all(bool(torch.isfinite(v).all()) for v in probabilities.values()):
            raise RuntimeError('Nonfinite prediction.')
        if args.mode == 'smoke':
            reference, _ = base.retained_predict(image,geometry,{args.dataset:banks['original']},vip,
                {args.dataset:queries['original']}, methods=(base.REFERENCE,))
            if not np.array_equal(probabilities['Reference'].argmax(0).cpu().numpy(),reference[args.dataset][base.REFERENCE]):
                raise RuntimeError('Reference replay failed.')
            save(output/'results.json',dict(status='complete',processed_images=1,total_images=1,
                target_masks_loaded=False,exact_retained_prediction=True,diagnostics=diagnostic,
                **check_frozen(state,geometry,vip)))
            return
        # Ground truth is read only after all predictions and adaptation are frozen.
        target = load_mask(sample,source['output_size'])
        for method in METHODS:
            prediction = adaptive_bg if method == 'AdaptiveResidual' else probabilities[method].argmax(0)
            cm = base.confusion(prediction.cpu().numpy(),target,classes)
            matrices[method] += cm
            per_image[method].append(cm)
        costs.append((len(source['local']),len(source['wide'])))
        if number == 1 or number%10 == 0 or number == len(samples):
            result = dict(status='running',processed_images=number,total_images=len(samples),signature=signature,
                metrics={args.dataset:{m:summary(cm,names,0) for m,cm in matrices.items()}},
                diagnostics={args.dataset:dict(tiles=sum(v[0] for v in costs),geometry_encodings=float(np.mean([v[0] for v in costs])),
                    wide_encodings=float(np.mean([v[1] for v in costs])),fine_forwards=0,
                    adaptive_gain_mean=float(np.mean([v['gain_mean'] for v in diagnostics])),
                    residual_rejection_images=sum(v['rejection'] for v in diagnostics))},
                wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
                target_labels_used_for_online_adaptation=False)
            save(output/'results.json',result)
            print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
        del source,probabilities,cache,wide_cache,q,adaptive_bg
    np.savez_compressed(output/'per_image_confusions.npz',sample_keys=np.asarray(signature['sample_keys']),
        **{args.dataset+'__'+m:np.stack(v) for m,v in per_image.items()})
    save(output/'adaptive_diagnostics.json',dict(samples=signature['sample_keys'],diagnostics=diagnostics))
    result.update(status='complete',**check_frozen(state,geometry,vip))
    save(output/'results.json',result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        parser.add_argument('--'+option,required=True)
    parser.add_argument('--mode',choices=('smoke','full'),required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--sample-seed',type=int,default=20260923)
    parser.add_argument('--vdd-ontology',default='official')
    # Observe existing prepare results without a second backbone call.
    original_prepare=base.GeometryExecution.prepare_image
    captured=[]
    def capture(self,*a,**kw):
        prepared=original_prepare(self,*a,**kw)
        captured.append(prepared.geometry_patch_conditional[0].float())
        return prepared
    base.GeometryExecution.prepare_image=capture
    original_observations=base.observations
    def observe(*a,**kw):
        captured.clear()
        source=original_observations(*a,**kw)
        for tile,relation in zip(source['local'],captured):
            tile['relation']=relation
        return source
    base.observations=observe
    main(parser.parse_args())
