"""Post-freeze scale strata from RGB headers and saved labelled confusions."""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from dinotool.natural_evaluation import discover_samples
from dinotool.natural_wide_resolution import target_size
from merge_competitive_evidence_shards import metric_summary
from eval_rival_fine_full import save

METHODS=('Frozen','NaturalShortEdge')


def main(root):
    output=root/'resolution_strata.json'
    if output.exists():raise RuntimeError('Existing diagnosis; preserve it.')
    suite=json.loads((root/'suite_results.json').read_text())
    if suite['status']!='complete':raise RuntimeError('Complete verified suite required.')
    entries=json.loads((root/'protocol.json').read_text())['datasets']
    result=dict(provenance='Post-freeze labelled audit only; scale strata use RGB dimensions, not target masks. No model/selector modification.',datasets={})
    for d,entry in entries.items():
        if entry['family']!='natural':continue
        merged=json.loads((root/'full'/d/'merged.json').read_text())
        if not merged['exact_frozen_confusion_replay']:raise RuntimeError('Frozen replay missing.')
        names=merged['signature']['classes'][d];n=len(names)
        strata={}
        for sample in discover_samples(d,entry['data_root']):
            with Image.open(sample.image_path) as image:w,h=image.size
            old=(int(h*448/max(h,w)+.5),int(w*448/max(h,w)+.5));new=target_size(h,w)
            strata[sample.key]='same' if old==new else 'increased' if new[0]*new[1]>old[0]*old[1] else 'reduced'
        cm={group:{m:np.zeros((n,n),np.int64) for m in METHODS} for group in ('same','increased','reduced')}
        seen=[]
        for shard in range(entry['shards']):
            with np.load(root/'full'/d/('s'+str(shard))/'per_image_confusions.npz',allow_pickle=False) as arrays:
                keys=arrays['sample_keys'].tolist();seen.extend(keys)
                groups=np.asarray([strata[k] for k in keys])
                for m in METHODS:
                    matrices=arrays[d+'__'+m]
                    for group in cm:cm[group][m]+=matrices[groups==group].sum(0)
        if len(seen)!=len(set(seen)) or set(seen)!=set(strata):raise RuntimeError('Strata coverage mismatch.')
        for m in METHODS:
            total=sum((v[m] for v in cm.values()),np.zeros((n,n),np.int64))
            if not np.array_equal(total,merged['metrics'][d][m]['confusion_matrix']):raise RuntimeError('Strata confusion sums differ.')
        rows={}
        for group,matrices in cm.items():
            support=np.zeros(n,bool)
            for a in matrices.values():support|=(a.sum(0)+a.sum(1)-a.diagonal())>0
            metrics={}
            targets=matrices['Frozen'].sum(1)
            for m,a in matrices.items():
                if not np.array_equal(targets,a.sum(1)):raise RuntimeError('Stratum scored targets differ.')
                union=a.sum(0)+a.sum(1)-a.diagonal()
                iou=np.divide(a.diagonal(),union,out=np.zeros(n,float),where=union>0)
                metrics[m]=metric_summary(a,names,0)
                metrics[m]['matched_support_miou_percent']=100*float(iou[support].mean()) if support.any() else None
            rows[group]=dict(images=sum(v==group for v in strata.values()),matched_support_classes=int(support.sum()),metrics=metrics)
        result['datasets'][d]=dict(global_sample_keys_sha256=merged['signature']['global_sample_keys_sha256'],
            global_confusion_sums_verified=True,strata=rows,
            same_resolution_confusion_identical=np.array_equal(cm['same']['Frozen'],cm['same']['NaturalShortEdge']))
    save(output,result)
    compact={d:{g:dict(images=v['images'],matched_miou={m:r['matched_support_miou_percent'] for m,r in v['metrics'].items()})
        for g,v in row['strata'].items()} for d,row in result['datasets'].items()}
    print(json.dumps(compact))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
