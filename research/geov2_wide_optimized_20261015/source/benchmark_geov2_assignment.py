"""Measure slot construction only; no model quality or end-to-end claim."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dinotool.context_memory_observer import ContextMemoryConfig, context_assignments
from dinotool.geov2_wide_optimized import wide_assignments


@torch.inference_mode()
def main(args):
    torch.manual_seed(20261015)
    rows=[]
    for n in (784,1008):
        raw=torch.randn(1,n,1024,device='cuda',dtype=torch.bfloat16)
        for k in (24,48):
            config=ContextMemoryConfig(capacity=k,center=False)
            old=context_assignments(raw,config)
            new=wide_assignments(raw,config)
            error=float((old-new).abs().max())
            if error>1e-6:
                raise RuntimeError('Slot assignment changed: '+str(error))
            row=dict(tokens=n,capacity=k,max_abs_error=error,
                     same_slot_argmax=torch.equal(old.argmax(-1),new.argmax(-1)))
            for name,fn in (('incumbent',context_assignments),('gather_scatter',wide_assignments)):
                for _ in range(3):fn(raw,config)
                times=[]
                for _ in range(12):
                    torch.cuda.synchronize();start=time.perf_counter()
                    fn(raw,config);torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    fn(raw,config);torch.cuda.synchronize()
                scalar=sum(e.count for e in prof.key_averages() if e.key=='aten::_local_scalar_dense')
                row[name]=dict(median_ms=float(np.median(times)),milliseconds=times,host_scalar_reads=scalar)
            rows.append(row)
    result=dict(status='complete',gpu=torch.cuda.get_device_name(),rows=rows,
                scope='Synthetic slot-construction microbenchmark; excludes image encoding/readout.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    main(p.parse_args())
