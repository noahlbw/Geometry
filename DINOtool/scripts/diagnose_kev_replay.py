"""Read-only identity diagnosis for the paired old-model replay."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np


def main(root,d):
    oldroot=root.parent/'alias_finalization_20261009'
    old=torch.load(oldroot/'text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
    new=torch.load(root/'text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
    for name,row in old['encoded'].items():
        if name not in new['encoded']:continue
        item=new['encoded'][name]
        wide=new['stores'][item['template']][item['indices']] if 'indices' in item else item['wide']
        local=F.normalize(wide.float().mean(1),dim=-1) if 'indices' in item else item['local']
        print(json.dumps(dict(bank=name,old_local_dtype=str(row['local'].dtype),new_local_dtype=str(local.dtype),
            wide_max_error=float((row['wide']-wide).abs().max()),local_max_error=float((row['local']-local).abs().max()),
            local_cosine_min=float(F.cosine_similarity(row['local'].float(),local.float()).min()))))
    baseline=json.loads((oldroot/'full'/d/'merged.json').read_text())
    folders=list((root/'full'/d).glob('s*/results.json'))
    for p in baseline['metrics']:
        matrices=[np.asarray(json.loads(f.read_text())['metrics'][p]['PreviousFinal']['confusion_matrix']) for f in folders]
        newcm=sum(matrices);oldcm=np.asarray(baseline['metrics'][p]['Uniform']['confusion_matrix'])
        print(json.dumps(dict(protocol=p,old=baseline['metrics'][p]['Uniform']['mean_iou_percent'],
            new=[json.loads(f.read_text())['metrics'][p]['PreviousFinal']['mean_iou_percent'] for f in folders],
            confusion_l1=int(abs(newcm-oldcm).sum()),targets_equal=bool(np.array_equal(newcm.sum(1),oldcm.sum(1))))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',required=True)
    a=p.parse_args()
    if a.dataset=='all':
        entries=json.loads((a.root/'protocol.json').read_text())['datasets']
        for d,e in entries.items():
            paths=[a.root/'full'/d/('s'+str(i))/'results.json' for i in range(e['shards'])]
            if all(f.exists() and json.loads(f.read_text())['status']=='complete' for f in paths):
                print('DATASET '+d,flush=True);main(a.root,d)
    else:main(a.root,a.dataset)
