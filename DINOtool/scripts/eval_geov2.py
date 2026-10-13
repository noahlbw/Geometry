"""Fixed Geov2 v0.1 validation and isolated whole-image A800 cost measurement."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.geov2 import Geov2, AliasScorer, Geov2Config
from dinotool.model import DINOTextSegmenter
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geometry_vip_reliability import summary
from eval_stride_ov_loveda_e1 import make_checkpoints

TOOL = Path(__file__).resolve().parents[1]
BASE = TOOL.parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Preserve completed output: '+str(path))
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.rename(path)


def prepare(args):
    root = Path(args.output_root)
    cache_path = root/'text_cache'/(args.dataset+'.pt')
    source = TOOL/'results/kev_alias_search_20261009'
    published = read(source/'protocol.json')['datasets'][args.dataset]
    bank = published['banks'][args.text_bank]
    if args.text_bank != 'official_imagenet' and args.dataset != 'voc21':
        raise ValueError('The segmentation-template control is declared for VOC21 only.')
    if args.mode != 'prepare' and args.num_shards > 1 and not cache_path.exists():
        raise RuntimeError('Run --mode prepare once before dispatching parallel shards.')
    if not cache_path.exists():
        packed = torch.load(source/'text_cache'/(args.dataset+'.pt'), map_location='cpu', weights_only=True)
        if packed['identity']['banks'][args.text_bank] != bank:
            raise ValueError('Official words differ from cached identity.')
        row = packed['encoded'][args.text_bank]
        templates = packed['stores'][row['template']][row['indices']]
        features = F.normalize(F.normalize(templates.float(), dim=-1).mean(1), dim=-1)
        union_classes = [0] if args.dataset == 'voc21' else []
        identity = dict(bank=bank, source=str(source/'text_cache'/(args.dataset+'.pt')),
                        source_template_storage_dtype=str(templates.dtype),
                        encoding='cached normalized prompt vectors -> FP32 template normalization/mean/normalization',
                        checkpoints=packed['identity']['checkpoints'], union_classes=union_classes,
                        extra_residual_concepts=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(identity=identity, features=features, parents=row['parents'].clone()), cache_path)
        save(root/'text_cache'/(args.dataset+'.json'), identity)
    else:
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        if cached['identity']['bank'] != bank:
            raise ValueError('Output root already contains a different frozen text bank.')
    return cache_path, published, bank


@torch.inference_mode()
def main(args):
    cache_path, entry, bank = prepare(args)
    if args.mode == 'prepare':
        print(json.dumps(dict(status='prepared', dataset=args.dataset, cache=str(cache_path))), flush=True)
        return
    root = Path(args.output_root)
    filename = args.dataset+(f'_s{args.shard_index}' if args.mode == 'full' else '')+'.json'
    output = root/args.mode/filename
    if output.exists():
        raise RuntimeError('Preserve completed output.')
    args.data_root = entry['data_root']
    args.sample_seed = 20260923
    args.vdd_ontology = 'official'
    data_entry = dict(entry, banks={'original':bank})
    validation, samples, images, masks, names_map = load_samples(args, data_entry)
    if len(validation) != entry['total_images']:
        raise RuntimeError('Validation image count differs.')
    names = names_map[args.dataset]
    if args.mode == 'dev':
        lookup = {s.key:s for s in validation}
        samples = [lookup[k] for k in entry['development_keys'][:16]]
    cached = torch.load(cache_path, map_location='cpu', weights_only=True)
    # The upstream local hub entry point keeps a zero-argument tokenizer
    # factory.  Pin it to the checkpoint vocabulary so the standalone Geov2
    # runner is reproducible outside the original VIP checkout.
    from dinov3.eval.text import tokenizer as _dino_tokenizer
    _dino_tokenizer.get_tokenizer = lambda *_a, **_k: _dino_tokenizer.Tokenizer(
        vocab_path=str(Path(args.checkpoint_dir) / 'bpe_simple_vocab_16e6.txt.gz'))
    backbone = DINOTextSegmenter(make_checkpoints(args), device=args.device, visual_only=True)
    scorer = AliasScorer(cached['features'].to(backbone.device), cached['parents'].to(backbone.device),
                         names, cached['identity']['union_classes'])
    config = Geov2Config(gate_mode=args.gate_mode, writeback_gain=args.writeback_gain,
                         gate_min=args.gate_min, gate_max=args.gate_max,
                         gate_temperature=args.gate_temperature,
                         gate_adaptation=args.gate_adaptation)
    model = Geov2(backbone, scorer, config)
    common = dict(dataset=args.dataset, model='Geov2-v0.2-adaptive-writeback', config=model.specification(),
                  text_identity=cached['identity'], gpu=torch.cuda.get_device_name(),
                  visual_parameter_dtype=str(next(backbone.model.visual_model.parameters()).dtype),
                  text_tower_on_gpu=False, sample_protocol='official',
                  total_images=len(validation), num_shards=args.num_shards, shard_index=args.shard_index)
    if args.mode == 'cost':
        keys = read(TOOL/'results/efficient_geometry_20261012/cost_keys.json')[args.dataset]
        lookup = {s.key:s for s in validation}
        rows = []
        torch.cuda.empty_cache()
        for key in keys:
            image = images(lookup[key])
            for _ in range(2):
                model.predict(image)
            times, peaks, reserved = [], [], []
            for _ in range(7):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                model.predict(image)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                peaks.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            rows.append(dict(key=key, shape=list(image.shape[-2:]), milliseconds=times,
                             peak_allocated_mib=max(peaks), peak_reserved_mib=max(reserved)))
        result = dict(common, status='complete', images=rows,
                      median_ms=float(np.median([t for r in rows for t in r['milliseconds']])),
                      peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
                      peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows),
                      target_masks_loaded=False,
                      timing='Whole CPU RGB tensor -> CPU IDs; excludes decode/init/text cache; same historical seven keys, two warmups, seven repeats.')
        save(output, result)
        print(json.dumps({k:result[k] for k in ['dataset','median_ms','peak_allocated_mib','peak_reserved_mib']}), flush=True)
        return
    diagnostics = args.mode in ('smoke','dev') or args.diagnostics
    arms = ('Geov2','Local','Wide','Mean') if diagnostics else ('Geov2',)
    matrices = {k:np.zeros((len(names),len(names)), np.int64) for k in arms}
    per_image = {k:[] for k in arms}
    keys = []
    start = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = images(sample)
        actual = model.predict(image, diagnostics=diagnostics)
        actual = actual if diagnostics else {'Geov2':actual}
        target = masks(sample, args.dataset, tuple(image.shape[-2:]))
        for k in arms:
            cm = confusion(actual[k], target, len(names))
            matrices[k] += cm
            per_image[k].append(cm)
        keys.append(sample.key)
        if number % 25 == 0 or number == len(samples):
            progress = dict(dataset=args.dataset, shard=args.shard_index, processed=number, selected=len(samples),
                            seconds=round(time.perf_counter()-start, 2))
            print(json.dumps(progress), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), keys=np.asarray(keys),
                        **{k:np.stack(v) for k,v in per_image.items()})
    metrics = {k:summary(v,names,0) for k,v in matrices.items()}
    save(output, dict(common, status='complete', processed_images=len(samples), sample_keys=keys,
                      metrics=metrics, confusion={k:v.tolist() for k,v in matrices.items()},
                      wall_seconds=time.perf_counter()-start))
    print(json.dumps(dict(dataset=args.dataset, metrics=metrics)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('voc21','vdd'), required=True)
    p.add_argument('--mode', choices=('prepare','smoke','dev','full','cost'), required=True)
    p.add_argument('--output-root', default=str(TOOL/'results/geov2_v01_20261013'))
    p.add_argument('--checkpoint-dir', default=str(BASE/'ckpt/DINO'))
    p.add_argument('--dinov3-repo', default=str(TOOL/'dinov3_hub'))
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--text-bank', choices=('official_imagenet','official_segmentation'), default='official_imagenet')
    p.add_argument('--gate-mode', choices=('adaptive', 'fixed', 'target'), default='adaptive')
    p.add_argument('--writeback-gain', type=float, default=1.)
    p.add_argument('--gate-min', type=float, default=.5)
    p.add_argument('--gate-max', type=float, default=2.)
    p.add_argument('--gate-temperature', type=float, default=.25)
    p.add_argument('--gate-adaptation', type=float, default=.75)
    main(p.parse_args())
