"""Offline frozen language judgments. Selected vocabularies never use image labels."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from kev.checkpoint import Checkpoint, LoadOptions
from dinotool.kev_alias_selection import (IMPLEMENTATION,CHECKPOINT,THRESHOLDS,
    candidate_groups,context,question,restored,admit)
from eval_rival_fine_full import save


@torch.inference_mode()
def main(args):
    root=Path(args.root);output=root/'language'/args.dataset
    if (output/'selection.json').exists():raise RuntimeError('Completed language judgments are preserved.')
    output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text());entry=protocol['datasets'][args.dataset]
    code_commit=subprocess.check_output(['git','-C',args.kev_root,'rev-parse','HEAD'],text=True).strip()
    if code_commit!=protocol['kev_code_commit']:raise RuntimeError('Kev code revision differs.')
    ck=Checkpoint(CHECKPOINT)
    tokenizer,model=ck.load('cuda',LoadOptions(dtype=torch.bfloat16,merge=True,fused=False,cuda_graphs=False))
    model.eval().requires_grad_(False)
    names=[c['name'] for c in entry['banks']['original']['classes']]
    groups=candidate_groups(entry);scores=[];started=time.perf_counter();questions=0
    for i,(name,words) in enumerate(zip(names,groups)):
        saved=output/f'class_{i:03}.json'
        if saved.exists():
            row=json.loads(saved.read_text())
            if row['class']!=name or row['aliases']!=words:raise RuntimeError('Saved taxonomy judgments differ.')
            scores.append(row['scores']);continue
        results={}
        if i!=entry['background_index']:
            pending=[a for a in words if a.casefold()!=name.casefold()]
            rotations=[]
            for rotation in (0,2):
                mapped={}
                for begin in range(0,len(pending),12):
                    batch=pending[begin:begin+12]
                    specs=[question(a,rotation) for a in batch]
                    record=dict(state=context(entry,i),questions=[s[0] for s in specs])
                    # Kev's branch limit includes the shared taxonomy state.
                    encoded=model.encode(tokenizer,record,max_state=8192,max_branch=8192,strict=True)
                    predictions=model.probs(encoded)
                    if len(predictions)!=len(batch):raise RuntimeError('Question/probability count differs.')
                    for alias,p,(_,order) in zip(batch,predictions,specs):
                        if torch.is_tensor(p):p=p.detach().float().cpu().tolist()
                        mapped[alias]=restored(p,order)
                    questions+=len(batch)
                rotations.append(mapped)
            for alias in pending:
                probability=[(rotations[0][alias][j]+rotations[1][alias][j])/2 for j in range(4)]
                results[alias]=dict(probabilities=probability,
                    admission_score=probability[0]+.5*probability[1],rotations=[r[alias] for r in rotations])
        scores.append(results)
        save(saved,dict(**{'class':name},aliases=words,scores=results))
        save(output/'results.json',dict(status='running',processed_classes=i+1,total_classes=len(names),
            questions=questions,wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(dataset=args.dataset,classes=i+1,total=len(names),questions=questions)),flush=True)
    background=entry['background_index']
    # Residual/background concepts retain their existing curated ontology.
    if background is not None:groups[background]=entry['banks']['semantic_segmentation' if entry['family']=='natural' else 'original']['classes'][background]['synonyms']
    banks={}
    for template in ('ImageNet80','seg_template'):
        for threshold in THRESHOLDS:
            key=f'kev{int(threshold*100)}_'+('imagenet' if template=='ImageNet80' else 'segmentation')
            banks[key]=dict(classes=admit(names,groups,scores,threshold,background),template=template)
        banks['pool_'+('imagenet' if template=='ImageNet80' else 'segmentation')]=dict(
            classes=[dict(name=n,synonyms=g) for n,g in zip(names,groups)],template=template)
    row=dict(status='complete',implementation=IMPLEMENTATION,checkpoint=CHECKPOINT,
        kev_code_commit=code_commit,base=ck.meta.base,base_revision=ck.meta.base_revision,
        temperature=ck.meta.temperature,questions=questions,banks=banks,
        counts={k:[len(c['synonyms']) for c in b['classes']] for k,b in banks.items()},
        taxonomy_only=True,target_masks_loaded=False,images_loaded=False,
        calibrated_visual_correctness=False,rotations=[0,2],wall_seconds=time.perf_counter()-started,
        peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
    save(output/'selection.json',row);save(output/'results.json',row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--dataset',required=True);p.add_argument('--kev-root',required=True)
    main(p.parse_args())
