"""Read RGB headers only, never masks; quantify the proposed fixed view prior."""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from dinotool.natural_evaluation import discover_samples
from dinotool.inference import tile_starts
from dinotool.natural_wide_resolution import target_size
from eval_rival_fine_full import save


def main(root):
    output=root/'wide_size_audit.json'
    if output.exists():raise RuntimeError('Existing metadata audit.')
    entries=json.loads((root/'protocol.json').read_text())['datasets']
    result=dict(target_masks_loaded=False,datasets={})
    for d,entry in entries.items():
        if entry['family']!='natural':continue
        rows=[]
        for sample in discover_samples(d,entry['data_root']):
            with Image.open(sample.image_path) as image:
                w,h=image.size
            old=(int(h*448/max(h,w)+.5),int(w*448/max(h,w)+.5))
            new=target_size(h,w)
            counts=[len(tile_starts(a,336,224))*len(tile_starts(b,336,224)) for a,b in (old,new)]
            rows.append((new[0]/old[0],min(old),min(new),*counts))
        data=np.asarray(rows)
        result['datasets'][d]=dict(images=len(rows),linear_scale_ratio_median=float(np.median(data[:,0])),
            proportion_higher_resolution=float((data[:,0]>1).mean()),proportion_lower_resolution=float((data[:,0]<1).mean()),
            old_short_edge_median=float(np.median(data[:,1])),new_short_edge_median=float(np.median(data[:,2])),
            old_wide_calls_mean=float(data[:,3].mean()),new_wide_calls_mean=float(data[:,4].mean()),new_wide_calls_max=int(data[:,4].max()))
    save(output,result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root)
