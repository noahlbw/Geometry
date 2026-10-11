"""Native official models with matched images/targets and explicit transfer configs.

Only this adapter handles data/metrics. All visual/text/model/postprocessing
operators execute from pinned upstream source. Labels are opened after inference.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(keys):
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def load_rgb(row, dataset):
    sample = SimpleNamespace(**{**row, 'image_path': Path(row['image_path']), 'mask_path': Path(row['mask_path'])})
    if dataset in ('flair1', 'landcoverai'):
        from dinotool.gear_datasets import load_rgb as loader
        return loader(sample, dataset)
    if dataset in ('vdd', 'potsdam', 'vaihingen'):
        from dinotool.rs_external import load_rgb as loader
        return loader(sample, dataset)
    if dataset == 'oem':
        from dinotool.oem import _read_rgb
        array = np.asarray(_read_rgb(sample.image_path), dtype=np.uint8).copy()
    else:
        with Image.open(sample.image_path) as source:
            array = np.array(source.convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def load_target(row, dataset, protocol, shape):
    sample = SimpleNamespace(**{**row, 'image_path': Path(row['image_path']), 'mask_path': Path(row['mask_path'])})
    if dataset in ('flair1', 'landcoverai'):
        from dinotool.gear_datasets import load_target as loader
        return loader(sample, dataset, shape)
    if dataset in ('vdd', 'potsdam', 'vaihingen'):
        from dinotool.rs_external import load_target as loader
        return loader(sample, dataset, shape)
    if dataset == 'oem':
        from dinotool.oem import _read_mask, _validate_raw_labels, OEM_RAW_TO_CLASS_INDEX
        raw = _read_mask(sample.mask_path)
        _validate_raw_labels(raw, sample.key)
        lut = np.full(256, 255, np.int64)
        for raw_id, index in OEM_RAW_TO_CLASS_INDEX.items():
            lut[raw_id] = index
        result = lut[raw]
    elif dataset in ('loveda', 'udd5'):
        with Image.open(sample.mask_path) as source:
            raw = np.asarray(source).copy()
        if raw.ndim == 3:
            raw = raw[..., 0]
        if dataset == 'udd5':
            if set(np.unique(raw)) - set(range(5)):
                raise RuntimeError('Unexpected UDD5 labels')
            result = raw.astype(np.int64)
        else:
            lower = 2 if protocol == 'P' else 1
            valid = (raw >= lower) & (raw <= 7)
            result = np.full(raw.shape, 255, np.int64)
            result[valid] = raw[valid].astype(np.int64)-lower
    else:
        from dinotool.natural_evaluation import load_target as loader
        return loader(sample, dataset, shape)
    if result.shape != shape:
        raise RuntimeError('Matched target shape changed: '+row['key'])
    return result


def summary(matrix, names, background=None):
    matrix = np.asarray(matrix, np.int64)
    tp = np.diag(matrix)
    target, predicted = matrix.sum(1), matrix.sum(0)[:len(names)]
    union = target+predicted-tp
    iou = np.divide(tp, union, out=np.full(len(names), np.nan), where=union>0)*100
    result = dict(mean_iou_percent=round(float(np.nanmean(iou)), 4) if np.isfinite(iou).any() else None,
        pixel_accuracy_percent=round(float(tp.sum()/max(target.sum(), 1)*100), 4),
        per_class=[dict(name=n, iou_percent=None if np.isnan(iou[i]) else round(float(iou[i]), 4),
            target_pixels=int(target[i]), predicted_pixels=int(predicted[i]),
            precision_percent=None if predicted[i]==0 else float(tp[i]/predicted[i]*100),
            recall_percent=None if target[i]==0 else float(tp[i]/target[i]*100)) for i,n in enumerate(names)],
        confusion_matrix=matrix.tolist(), valid_pixels=int(matrix.sum()))
    if background is not None:
        fg=np.delete(iou,background)
        result['non_residual_mean_iou_percent'] = round(float(np.nanmean(fg)), 4) if np.isfinite(fg).any() else None
    return result


def load_model(config, root):
    source = (root/'sources'/config['source_path'].split('/sources/',1)[-1]).resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    # Supply the pinned, preloaded DINO source to the author's torch.hub call.
    hub_load = torch.hub.load
    def pinned_hub(repo, model, *args, **kwargs):
        if repo == 'facebookresearch/dino:main':
            return hub_load(str(root/'dependencies/dino'), model, *args, source='local', **kwargs)
        return hub_load(repo, model, *args, **kwargs)
    torch.hub.load = pinned_hub
    module = importlib.import_module(config['module'])
    kwargs = dict(config['model'])
    kwargs['name_path'] = str(root/config['word_file'])
    kwargs.pop('type', None)
    if config['method'] == 'PEARL':
        kwargs['clip_path'] = str(root/'checkpoints/ViT-B-16.pt')
    if config['method'] == 'GLA-CLIP':
        from cfg import ModelConfig
        model_cfg = ModelConfig(**kwargs)
        for key,value in config['full_flags'].items():
            setattr(model_cfg, key, value)
        model_cfg.work_dir = str(root/'gla_work')
        kwargs['model_cfg'] = model_cfg
    if config['method'] == 'SegEarth-OV':
        kwargs['feature_up_cfg'] = dict(model_name='jbu_one',
            model_path=str(root/'checkpoints/xclip_jbu_one_million_aid.ckpt'))
    if config['method'] == 'CorrCLIP':
        kwargs['mask_generator'] = 'sam2'
        # Native SAM2 loader uses this relative checkpoint filename.
        if not (source/'sam2_hiera_large.pt').exists():
            raise FileNotFoundError('Native SAM2 checkpoint symlink missing')
    model = getattr(module, config['class'])(**kwargs).cuda().eval()
    return model


@torch.inference_mode()
def predict(model, image, row, config):
    import mmcv
    from mmseg.structures import SegDataSample
    shape = tuple(image.shape[-2:])
    rgb = image.mul(255).round().byte().permute(1,2,0).numpy()
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    resize = config['resize']
    if resize['keep_ratio']:
        bgr = mmcv.imrescale(bgr, tuple(resize['scale']), interpolation='bilinear')
    else:
        bgr = mmcv.imresize(bgr, tuple(resize['scale']), interpolation='bilinear')
    sample = SegDataSample()
    sample.set_metainfo(dict(ori_shape=shape, img_shape=bgr.shape[:2], pad_shape=bgr.shape[:2],
        img_path=row['image_path'], scale_factor=(bgr.shape[1]/shape[1], bgr.shape[0]/shape[0]),
        padding_size=[0,0,0,0], flip=False))
    data = model.data_preprocessor(dict(inputs=[torch.from_numpy(bgr).permute(2,0,1)],
        data_samples=[sample]), training=False)
    result = model.predict(data['inputs'], data['data_samples'])[0]
    for key in ('seg_logits', 'seg_logit'):
        if key in result and not torch.isfinite(getattr(result,key).data).all():
            raise FloatingPointError('Non-finite native output: '+row['key'])
    pred = result.pred_sem_seg.data.squeeze(0)
    if tuple(pred.shape) != shape:
        raise RuntimeError('Native prediction must return original resolution')
    if config.get('prediction_map') is not None:
        pred=torch.tensor(config['prediction_map'],device=pred.device)[pred.long()]
    return pred


def main(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise RuntimeError('Existing output preserved: '+str(output))
    output.mkdir(parents=True)
    suite = json.loads((root/'protocol.json').read_text())
    entry = suite['datasets'][args.dataset]
    manifest = json.loads((root/'samples.json').read_text())[args.dataset]
    config = suite['methods'][args.method]['configs'][args.dataset]
    rows = manifest['samples'][args.shard_index::args.shards]
    if args.phase in ('smoke', 'cost'):
        if args.phase=='smoke':rows=rows[:1]
        else:
            lookup={r['key']:r for r in manifest['samples']}
            rows=[lookup[k] for k in json.loads((root/'cost_keys.json').read_text())[args.dataset]]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(20261011)
    model = load_model(config, root)
    initial_weights = {k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    protocols = manifest['classes']
    # LoveDA P is a separate six-query inference, not D argmax shifted by one.
    models = {p:model for p in protocols}
    if args.dataset == 'loveda':
        pconfig = dict(config, word_file=config['foreground_word_file'],model={**config['model'],'prob_thd':0.})
        models['P'] = load_model(pconfig, root)
    matrices = {p:np.zeros((len(names),len(names)+1),np.int64) for p,names in protocols.items()}
    per_image = {p:[] for p in protocols}
    timings = []
    keys = []
    started = time.time()
    signature = dict(implementation=suite['implementation'], method=args.method, dataset=args.dataset,
        config=config, source_commit=suite['methods'][args.method]['commit'],
        checkpoint_manifest=json.loads((root/'checkpoint_manifest.json').read_text()),
        runtime=json.loads((root/'runtime.json').read_text()), sample_manifest_sha256=manifest['global_sample_keys_sha256'])
    def record(status):
        value = dict(status=status, phase=args.phase, processed_images=len(keys), total_images=len(rows),
            signature=signature, sample_keys=keys, sample_keys_sha256=digest(keys),
            results={p:summary(matrix,protocols[p], None if p=='P' else entry['background_index'])
                for p,matrix in matrices.items()}, wall_seconds=time.time()-started,
            target_masks_loaded=args.phase=='evaluate' and bool(keys),weights_frozen=status=='complete',
            prediction_seconds=timings, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(output/'results.json',value)
        print(json.dumps(dict(method=args.method,dataset=args.dataset,phase=args.phase,
            status=status,processed=len(keys),total=len(rows))),flush=True)
    torch.cuda.reset_peak_memory_stats()
    record('running')
    if args.phase == 'cost':
        # No labels. Single resident model; LoveDA cost reports D, not double work P+D.
        if args.dataset=='loveda':
            del models['P']
            torch.cuda.empty_cache()
        costs=[]
        for row in rows:
            image=load_rgb(row,args.dataset)
            for _ in range(2):
                predict(model,image,row,config)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples=[]
            for _ in range(7):
                torch.cuda.synchronize(); begin=time.perf_counter()
                pred=predict(model,image,row,config)
                torch.cuda.synchronize(); samples.append((time.perf_counter()-begin)*1000)
                del pred
            costs.append(dict(key=row['key'],original_shape=list(image.shape[-2:]),milliseconds=samples,
                median_ms=float(np.median(samples)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        save(output/'cost.json',dict(status='complete',signature=signature,images=costs,
            median_ms=float(np.median([x for row in costs for x in row['milliseconds']])),
            includes='BGR conversion/official resize/preprocess/native inference/postprocess; CorrCLIP includes online SAM2',
            excludes='disk decode, model/text initialization, metric/target loading'))
        keys.extend(r['key'] for r in rows);record('complete');return
    for row in rows:
        image=load_rgb(row,args.dataset)
        torch.cuda.synchronize(); begin=time.perf_counter()
        predictions={p:predict(m,image,row,dict(config,
            word_file=config.get('foreground_word_file') if p=='P' else config['word_file']))
            for p,m in models.items()}
        torch.cuda.synchronize();timings.append(time.perf_counter()-begin)
        if args.phase!='smoke':
            for p,pred in predictions.items():
                target=load_target(row,args.dataset,p,tuple(image.shape[-2:]))
                prediction=pred.cpu().numpy().astype(np.int64)
                c=len(protocols[p]);valid=(target>=0)&(target<c)
                if np.any((prediction<0)|(prediction>c)):
                    raise RuntimeError('Native prediction outside locked taxonomy')
                matrix=np.bincount(target[valid]*(c+1)+prediction[valid],minlength=c*(c+1)).reshape(c,c+1)
                matrices[p]+=matrix;per_image[p].append(matrix)
        del predictions
        keys.append(row['key'])
        if len(keys)%10==0 or len(keys)==len(rows):
            record('running')
    for k,v in model.named_parameters():
        if not torch.equal(v.detach().cpu(),initial_weights[k]):
            raise RuntimeError('Frozen baseline weight changed: '+k)
    if args.phase=='evaluate':
        np.savez_compressed(output/'per_image_confusions.npz',**{p:np.stack(v) for p,v in per_image.items()})
    record('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','output','dataset','method'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--phase',choices=('smoke','evaluate','cost'),required=True)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    main(parser.parse_args())
