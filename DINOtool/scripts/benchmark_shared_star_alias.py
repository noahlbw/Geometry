"""Exclusive-GPU singleton timing, independent of full accuracy outputs."""
import argparse
from pathlib import Path
import statistics

import numpy as np
import torch

from benchmark_grouped_class_readout import CheckedGroupedCache
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.shared_star_alias import IMPLEMENTATION, BASELINE, MEAN, PRIMARY, BUDGET4, predict_image
from dinotool.vip_official_adapter import upstream_settings
from eval_alias_full_comparison_r3 import setup
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def main(args):
    output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing benchmark output; preserve it.')
    require_available_gpu(torch.device(args.device))
    lease=torch.empty(1,device=args.device)
    _,_,samples,loader,_,g,b,v,q,words,_,reference,_=setup(args)
    if reference is not None:raise RuntimeError('Unexpected historical model.')
    frozen=frozen_state(g,v);cache=CheckedGroupedCache();cache.verify=False
    names=(BASELINE,MEAN,PRIMARY,BUDGET4,'VIP_Matched20')
    def call(name,image):
        if name=='VIP_Matched20':
            return {p:{name:v.predict(image,query,upstream_settings(args.dataset))[0]} for p,query in q.items()},{}
        return predict_image(image,g,b,v,q,methods=(name,),cache=cache)
    output.mkdir(parents=True);rows=[]
    for index in sorted({0,len(samples)//2,len(samples)-1}):
        sample=samples[index];image=loader(sample)
        require_available_gpu(torch.device(args.device))
        expected={name:call(name,image)[0] for name in names}
        times={name:dict(seconds=[],peaks=[]) for name in names}
        for repeat in range(args.repetitions):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                require_available_gpu(torch.device(args.device))
                actual,timing=measure(call,(name,image),torch.device(args.device),1)
                require_available_gpu(torch.device(args.device))
                if any(not np.array_equal(actual[0][p][name],expected[name][p][name]) for p in b):
                    raise RuntimeError('Warmed prediction changed.')
                times[name]['seconds']+=timing['seconds'];times[name]['peaks'].append(timing['peak_allocated_mib'])
        for row in times.values():
            row.update(median_seconds=statistics.median(row['seconds']),peak_allocated_mib=max(row.pop('peaks')))
        rows.append(dict(sample_key=sample.key,image_size=list(image.shape[-2:]),timings=times))
        save(output/'results.json',dict(status='running',images=rows,implementation=IMPLEMENTATION,target_masks_loaded=False))
    require_available_gpu(torch.device(args.device))
    save(output/'results.json',dict(status='complete',images=rows,implementation=IMPLEMENTATION,
        exclusive_gpu_verified=True,benchmark_predictions_equal=True,target_masks_loaded=False,
        source_vocabulary=words,memory_scope='shared Geometry/VIP and frozen head snapshots resident; not standalone VIP memory',
        **check_frozen(frozen,g,v)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--mode',default='full')
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--repetitions',type=int,default=5)
    main(p.parse_args())
