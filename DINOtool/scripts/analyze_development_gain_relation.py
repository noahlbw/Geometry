"""Isolate gain/read-strength on existing labelled development histograms.

No model inference, new labels, fitting, or full-test winner selection.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from dinotool.development_readout import profiles,score
from dinotool.taxonomy_readout import rank_fraction,natural_profile


def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[]
    for d,entry in protocol['datasets'].items():
        path=root/'search'/d/'development_histograms.npz'
        if not path.exists():raise RuntimeError('Missing completed development: '+d)
        with np.load(path) as archive:
            hist=archive['histograms']
            candidates=profiles(entry['banks'])
            if len(hist)!=len(candidates):raise RuntimeError('Profile order mismatch.')
            # Zero bias, all confidence bins: no threshold rejection.
            matrices=hist[:,0].sum(1)
        support=matrices[0].sum(1)>0
        cache=torch.load(root/'text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
        banks=('semantic_segmentation',) if entry['family']=='natural' else ('original_imagenet','focused20')
        for name in banks:
            data=cache['encoded'][name]
            bank=SimpleNamespace(features=data['local'],parent_indices=data['parents'],
                canonical_mask=data['canonical'],class_count=len(entry['banks'][name]['classes']))
            r=rank_fraction(bank)
            selected,decision=natural_profile(bank)
            grid=[]
            for strength in (1.,2.,3.,'original'):
                values={}
                for gain in (.5,1.,2.):
                    matches=[i for i,p in enumerate(candidates) if p.bank==name and p.strength==strength
                             and p.coupling==gain and p.temperature==.07 and p.tau==1. and p.tem==1.]
                    if len(matches)!=1:raise RuntimeError('Controlled profile missing.')
                    values[str(gain)]=score(matrices[matches[0]],support)
                best=max(values,key=values.get)
                grid.append(dict(strength=strength,development_miou=values,best_gain=float(best)))
            fixed=next(x for x in grid if x['strength']==2.)
            rank_score=fixed['development_miou'][str(selected.coupling)]
            rows.append(dict(dataset=d,family=entry['family'],bank=name,
                class_count=bank.class_count,development_count=len(entry['development_keys']),
                development_source=entry['development_source'],canonical_rank_fraction=r,
                rank_selected_gain=selected.coupling,grid=grid,
                fixed_strength2_best_gain=fixed['best_gain'],fixed_strength2_rank_regret=max(fixed['development_miou'].values())-rank_score))
    return dict(scope='Existing labelled development only; frozen bank, T.07/tau1/tem1, zero bias and no rejection. No independent generalization claim or new parameter selection.',rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    print(json.dumps(analyze(p.parse_args().root)))
