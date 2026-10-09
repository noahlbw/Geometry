"""Pinned one-pass text-only generation; no images, labels or score inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from dinotool.semantic_coverage_input import prompt,parse_reply
from dinotool.semantic_role_admission import FrozenSemanticRoles,SOURCE
from eval_rival_fine_full import save
from run_region_semantic_suite_a800 import idle


def main(args):
    if args.output.exists() or not idle(int(os.environ['CUDA_VISIBLE_DEVICES'])):
        raise RuntimeError('Existing output or occupied physical GPU.')
    protocol=json.loads(args.protocol.read_text())
    if args.num_shards!=protocol['generation']['num_shards'] or not 0<=args.shard_index<args.num_shards:
        raise RuntimeError('Generation shard configuration differs from the frozen protocol.')
    for relative,expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen generation source differs: '+relative)
    records=protocol['records'][args.shard_index::args.num_shards]
    if not records:raise RuntimeError('Empty generation shard.')
    model=FrozenSemanticRoles(SOURCE);torch.manual_seed(20261009);started=time.perf_counter()
    row=dict(status='running',implementation=protocol['implementation'],source_manifest=model.manifest,
        records=records,generation=protocol['generation'],
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        target_images_loaded=False,target_masks_loaded=False,
        target_label_tuning=False,semantic_quality_verified=False,num_shards=args.num_shards,shard_index=args.shard_index)
    pending=[i for i,r in enumerate(records) if r['eligible_slots']]
    for record in records:
        record.update(accepted=[],rejected=[],raw_response=None,format_complete=not record['eligible_slots'],request_count=0)
    save(args.output,row)
    for start in range(0,len(pending),protocol['generation']['batch_size']):
        indices=pending[start:start+protocol['generation']['batch_size']]
        questions=[prompt(records[i]) for i in indices]
        rendered=[model.processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=q)])],
            tokenize=False,add_generation_prompt=True) for q in questions]
        inputs=model.processor.tokenizer(rendered,padding=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated=model.model.generate(**inputs,do_sample=False,num_beams=1,
                max_new_tokens=protocol['generation']['max_new_tokens'],use_cache=True,
                pad_token_id=model.processor.tokenizer.pad_token_id)
        replies=model.processor.tokenizer.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for i,q,reply in zip(indices,questions,replies):
            record=records[i];record.update(prompt=q,raw_response=reply,request_count=1)
            try:
                record['accepted'],record['rejected']=parse_reply(reply,record);record['format_complete']=True
            except (ValueError,TypeError,json.JSONDecodeError) as error:
                record.update(format_error=str(error),format_complete=False,accepted=[],rejected=[])
        row.update(processed=sum(r['request_count'] or not r['eligible_slots'] for r in records),total=len(records),
            wall_seconds=time.perf_counter()-started,peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576)
        save(args.output,row)
        print(json.dumps({k:row[k] for k in ('processed','total','wall_seconds')}),flush=True)
    row.update(status='complete',processed=len(records),total=len(records),wall_seconds=time.perf_counter()-started,
        weights_frozen=all(not p.requires_grad for p in model.model.parameters()),
        schema_failed_classes=[r['class_index'] for r in records if not r['format_complete']],
        fallback_rule='A schema failure retains the entire existing20 group; no retry or selective salvage')
    save(args.output,row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--num-shards',type=int,default=4)
    p.add_argument('--shard-index',type=int,required=True)
    main(p.parse_args())
