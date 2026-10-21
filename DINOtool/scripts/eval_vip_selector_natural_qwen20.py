"""VIP VG/SC screening on the shared Qwen20 natural-image candidate pool."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from dinotool.config import CheckpointConfig
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.model import DINOTextSegmenter
from dinotool.vip_official_adapter import VIPSettings
from dinotool.vip_paper_distillation import BackboneAttentionCapture, PaperAliasAccumulator, PaperDistillationConfig, canonical_indices, image_crops, source_logits, valid_patches, select_from_scores

CODE_ROOT=Path('/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926')
UPSTREAM=CODE_ROOT.parent/'third_party/VIP_official_5bd25ee'; CHECKPOINTS=CODE_ROOT.parent/'ckpt/DINO'
ROOTS={'voc21':Path('/data/test/datasets/VIP_natural/VOCdevkit/VOC2012'),'coco_object81':Path('/data/test/datasets/VIP_natural/COCOObject'),'ade150':Path('/data/test/datasets/VIP_natural/ADEChallengeData2016')}

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(x): return ' '.join(str(x).split())
def load_rgb(p):
    with Image.open(p) as im: a=np.asarray(im.convert('RGB'),dtype=np.float32)/255
    return torch.from_numpy(a.transpose(2,0,1))
def cfg(): return CheckpointConfig(dinov3_repo=CODE_ROOT/'dinov3_hub',checkpoint_dir=CHECKPOINTS,dinotxt_weights=CHECKPOINTS/'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth',lvd_weights=CHECKPOINTS/'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',sat_weights=CHECKPOINTS/'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',bpe_path=CHECKPOINTS/'bpe_simple_vocab_16e6.txt.gz')
def sample_paths(dataset, keys):
    root=ROOTS[dataset]; out=[]
    for k in keys:
        if dataset=='voc21': p=root/'JPEGImages'/(k+'.jpg')
        elif dataset=='coco_object81': p=root/'images/val2017'/(k+'.jpg')
        else: p=root/'images/training'/(k+'.jpg')
        if not p.is_file(): raise FileNotFoundError(p)
        out.append((k,p))
    return out
def bind_tokenizer():
    from dinov3.eval.text import tokenizer
    tokenizer.get_tokenizer=lambda *_a,**_k: tokenizer.Tokenizer(vocab_path=str(CHECKPOINTS/'bpe_simple_vocab_16e6.txt.gz'))

def canonical_indices_normalized(names, aliases, parents):
    """The frozen ADE pool contains a few harmless trailing spaces."""
    norm=lambda x: " ".join(str(x).split()).casefold()
    out=[]
    for i,name in enumerate(names):
        hit=[j for j,a in enumerate(aliases) if int(parents[j])==i and norm(a)==norm(name)]
        if len(hit)!=1: raise ValueError(f"canonical alias missing or duplicated: {name} ({len(hit)})")
        out.append(hit[0])
    return torch.tensor(out,device=parents.device,dtype=torch.long)

@torch.inference_mode()
def main(args):
    bind_tokenizer(); pool=load(args.pool); classes=[{'name':clean(c['name']),'synonyms':[clean(a) for a in c['synonyms']]} for c in pool['classes']]; names=tuple(c['name'] for c in classes)
    groups=[]
    for c in classes:
        # Keep residual/background canonical during selector comparison.
        groups.append((c['name'],) if c['name'].casefold() in {'background','other'} else tuple(c['synonyms']))
    samples=sample_paths(args.dataset, load(args.keys)['development_keys'])
    model=FiniteVIPObserver(DINOTextSegmenter(cfg(),device='cuda',amp=True),UPSTREAM)
    queries=model.encode_queries(names,tuple(groups)); settings=VIPSettings(); distill=PaperDistillationConfig()
    canonical=canonical_indices_normalized(queries.class_names,queries.aliases,queries.parents.cpu()); text=torch.nn.functional.normalize(queries.features.float().mean(1),dim=-1); cosine=(text*text[canonical[queries.parents.cpu()]]).sum(-1)
    collector=PaperAliasAccumulator(len(queries.aliases),model.device); started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with BackboneAttentionCapture(model.backbone.model.visual_model.backbone) as capture:
        for num,(key,path) in enumerate(samples,1):
            image=load_rgb(path); _,_,crops=image_crops(image,settings)
            for crop,bounds in crops:
                capture.reset(); patch=model.crop_patch_features(crop); attention=capture.mean(); valid=valid_patches(bounds,model.device); logits,_=source_logits(patch,queries,settings); collector.update_crop(logits.flatten(1).T,queries.parents,canonical.to(model.device),attention,valid,distill)
            collector.finalize_image()
            if num==1 or num%10==0 or num==len(samples): print(json.dumps({'dataset':args.dataset,'processed':num,'total':len(samples)}),flush=True)
    state=collector.state(); selection=select_from_scores(state,queries.class_names,queries.aliases,queries.parents.cpu(),cosine.cpu(),distill)
    selected=[]
    for i,c in enumerate(classes):
        # VIP treats VOC/COCO background as a fixed residual group.  It is
        # therefore not sent through the foreground VG/SC admission (the
        # official background group has no literal ``background`` anchor),
        # but it must be carried into the emitted bank unchanged.  This keeps
        # the experiment faithful to the official VIP background handling and
        # makes the foreground selector the only variable.
        if c['name'].casefold() in {'background', 'other'}:
            aliases = list(c['synonyms'])
        else:
            aliases = [a for a in selection['selected_alias_groups'][i]]
        selected.append({'name':c['name'],'synonyms':aliases})
    bank={'dataset':args.dataset,'classes':selected,'template':pool.get('template','seg_template'),'background_rule':'canonical_only'}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    save(out/'selection.json',{'status':'complete','implementation':'vip-vg-sc-qwen20-natural-v1-20261021','dataset':args.dataset,'pool_sha256':hashlib.sha256(Path(args.pool).read_bytes()).hexdigest(),'candidate_source':str(Path(args.pool).resolve()),'selection_samples':[k for k,_ in samples],'target_masks_loaded':False,'selection_uses_miou':False,'statistics':state,'query_metadata':{'class_names':list(queries.class_names),'aliases':list(queries.aliases),'parents':queries.parents.cpu().tolist(),'canonical':canonical.cpu().tolist(),'cosine':cosine.cpu().tolist()},'selection':selection,'wall_seconds':time.perf_counter()-started,'peak_cuda_memory_mib':torch.cuda.max_memory_allocated()/1048576,'bank':bank})
    save(out/'vip_bank.json',bank)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=tuple(ROOTS)); p.add_argument('--pool',required=True); p.add_argument('--keys',required=True); p.add_argument('--out',required=True); main(p.parse_args())
