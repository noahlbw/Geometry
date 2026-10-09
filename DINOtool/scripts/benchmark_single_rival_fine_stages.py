"""One fixed image, synchronized attribution of the cost-rejected sparse reader."""
import argparse
from pathlib import Path
import time

import torch

import dinotool.single_rival_fine_alias as reader
import dinotool.fine_observer_burst as burst
import eval_development_readout as development
import eval_rival_fine_full as fine
import eval_matched_contribution_alias as projection
from dinotool.bounded_fine_execution import FineCoverageExecution
from eval_single_rival_fine_alias import (original_setup,curated_setup,task_text,save,
    require_available_gpu)


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing profiling output; preserve it.')
    device = torch.device(args.device);require_available_gpu(device)
    lease = torch.empty(1,device=device);output.mkdir(parents=True)
    setup = curated_setup if args.dataset=='ade150' else original_setup
    protocol,entry,samples,loader,masks,g,b,v,q,words,identity,reference,_ = setup(args)
    b,q,_ = task_text(Path(args.suite_root),args.dataset,g,v,b,q,identity)
    plans = {p:reader.prepare_plan(bank) for p,bank in b.items()}
    execution = FineCoverageExecution(cached=True,burst=True)
    image = loader(samples[0])
    expected,_ = reader.predict_image(image,g,b,v,q,plans,args.dataset,
        methods=(reader.PRIMARY,),execution=execution)
    # All replacements are timing wrappers; tensor arguments/results pass unchanged.
    totals,stack,originals = {},[],[]
    def instrument(owner,name,label):
        original = getattr(owner,name);originals.append((owner,name,original))
        def timed(*positional,**keywords):
            torch.cuda.synchronize(device);start = time.perf_counter()
            node = [label,0.];stack.append(node)
            result = original(*positional,**keywords)
            torch.cuda.synchronize(device);elapsed = time.perf_counter()-start
            stack.pop()
            row = totals.setdefault(label,dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
            row['calls'] += 1;row['inclusive_seconds'] += elapsed;row['exclusive_seconds'] += elapsed-node[1]
            if stack:stack[-1][1] += elapsed
            return result
        setattr(owner,name,timed)
    for owner,name,label in ((development,'observations','base_visual_observations'),
        (development,'wide_scores','base_wide_projection'),(fine,'observe_fine','fine_read_and_projection'),
        (burst.FineObserverBurstGraph,'__call__','fine_encoder_burst'),
        (projection,'crop_from_features','wide_template_projection'),
        (fine,'crop_from_features','fine_template_projection'),
        (fine,'crop_stencil','fine_coverage_stencil'),(reader,'crop_stencil','cache_stencil'),
        (reader,'cache_crop','profile_and_cache'),(reader,'sample_cache','interpolate_alias_and_class'),
        (reader,'attenuated_delta','slot_writer'),(reader,'tile_fields','tile_fields')):
        instrument(owner,name,label)
    try:
        torch.cuda.synchronize(device);start = time.perf_counter()
        actual,diag = reader.predict_image(image,g,b,v,q,plans,args.dataset,
            methods=(reader.PRIMARY,),execution=execution)
        torch.cuda.synchronize(device);elapsed = time.perf_counter()-start
        import numpy as np
        if any(not np.array_equal(actual[p][reader.PRIMARY],expected[p][reader.PRIMARY]) for p in b):
            raise RuntimeError('Profiling wrappers changed prediction.')
        save(output/'results.json',dict(status='complete',sample_key=samples[0].key,
            target_masks_loaded=False,exact_original_prediction=True,instrumented_wall_seconds=elapsed,
            stages=totals,diagnostics=diag,
            caveat='Synchronized nested intervals perturb dispatch; exclusive spans subtract nested spans. Not singleton performance.'))
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--mode',default='full')
    parser.add_argument('--sample-seed',type=int,default=20260923);parser.add_argument('--vdd-ontology',default='official')
    main(parser.parse_args())
