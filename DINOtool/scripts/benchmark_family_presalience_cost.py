"""Correct the overhead comparator without changing any frozen prediction rule."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

import eval_family_lme_alias as harness
from benchmark_natural_sense_fast import measure,require_available_gpu
from dinotool import family_presalience_readout as trial
from dinotool import family_mass_alias as old
from dinotool.vip_official_adapter import upstream_settings
from eval_canonical_rival_trial import official_speed_queries,official_natural_single,OFFICIAL_SPEED
from eval_rival_fine_full import frozen_state,check_frozen,save


BASE='Original_BaseUniform'


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing timing supplement; preserve it.')
    device=torch.device(args.device);require_available_gpu(device);lease=torch.empty(1,device=device)
    output.mkdir(parents=True);save(output/'worker_status.json',dict(status='running',phase='loading'))
    setup=harness.curated_setup if args.dataset=='ade150' else harness.original_setup
    protocol,entry,samples,loader,masks,g,b,v,q,words,checkpoints,reference,_=setup(args)
    for name,wanted in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()!=wanted:
            raise RuntimeError('Frozen method source differs: '+name)
    harness.prepare_plan=trial.prepare_plan
    banks,queries,plans,identity,text_cost=harness.input_banks(root,args.dataset,g,v,b,q,checkpoints,protocol)
    old_plans={name:plan.mass for name,plan in plans.items()};frozen=frozen_state(g,v)
    official_q,settings,official_identity=official_speed_queries(args.dataset,v,{args.dataset:banks['Complete20']})
    names=(BASE,trial.PRIMARY,trial.VIP,OFFICIAL_SPEED);rows=[]
    def call(name,image):
        if name==BASE:
            out,diag=old.predict(image,g,v,banks,queries,old_plans,args.dataset,(old.UNIFORM,))
            return {name:out[old.UNIFORM]},diag
        if name==trial.VIP:return {name:v.predict(image,queries['Complete20'],upstream_settings(args.dataset))[0]},{}
        if name==OFFICIAL_SPEED:
            pred=official_natural_single(image,v,official_q[args.dataset],settings) if args.dataset=='ade150' else v.predict(image,official_q[args.dataset],settings)[0]
            return {name:pred},{}
        return trial.predict(image,g,v,banks,queries,plans,args.dataset,(name,))
    for index in sorted({0,len(samples)//2,len(samples)-1}):
        image=loader(samples[index]);expected={name:call(name,image)[0][name] for name in names}
        inefficient,_=trial.predict(image,g,v,banks,queries,plans,args.dataset,(trial.COMPLETE_BASE,))
        if not np.array_equal(expected[BASE],inefficient[trial.COMPLETE_BASE]):raise RuntimeError('Base predictions differ.')
        values={name:dict(seconds=[],peaks=[]) for name in names}
        for repeat in range(5):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                require_available_gpu(device);actual,t=measure(call,(name,image),device,1);require_available_gpu(device)
                if not np.array_equal(actual[0][name],expected[name]):raise RuntimeError('Warm prediction changed.')
                values[name]['seconds']+=t['seconds'];values[name]['peaks'].append(t['peak_allocated_mib'])
        for value in values.values():value.update(median_seconds=statistics.median(value['seconds']),peak_allocated_mib=max(value.pop('peaks')))
        ratios={name:values[trial.PRIMARY]['median_seconds']/values[name]['median_seconds'] for name in (trial.VIP,OFFICIAL_SPEED)}
        rows.append(dict(sample_key=samples[index].key,timings=values,ratios=ratios,below6x=max(ratios.values())<=6,base_prediction_equal=True))
    save(output/'timing.json',dict(status='complete',images=rows,cost_gate_passed=all(row['below6x'] for row in rows),
        target_masks_loaded=False,official_speed_reference=official_identity,source_text_identity=identity,
        baseline='Original frozen FamilyMass Uniform route; no unnecessary family amplification computation',
        includes='resize, encoders, actual alias rule, Geometry writeback, restoration, argmax',
        excludes='decode/frozen setup',scope='three fixed complete images, five rotated warmed singleton repetitions',
        memory_scope='both resident models/banks',**check_frozen(frozen,g,v)))
    save(output/'worker_status.json',dict(status='complete',phase='timing',processed=3,total=3))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--mode',default='benchmark')
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
