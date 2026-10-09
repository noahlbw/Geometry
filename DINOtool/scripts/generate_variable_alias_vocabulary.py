"""One pinned Qwen metadata-only generation, no image/label/performance inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import torch
from dinotool.semantic_role_admission import FrozenSemanticRoles,SOURCE
from dinotool.variable_alias_vocabulary import prompt,parse_reply,remove_shared_additions
from eval_rival_fine_full import save
from run_region_semantic_suite_a800 import idle


def main(protocol_path,output):
    if output.exists() or not idle(int(os.environ['CUDA_VISIBLE_DEVICES'])):
        raise RuntimeError('Existing output or occupied mapped GPU.')
    protocol=json.loads(protocol_path.read_text());entries=protocol['datasets'];records=[]
    for d,e in entries.items():
        bank='semantic_segmentation' if e['family']=='natural' else 'original_imagenet'
        classes=e['banks'][bank]['classes'];anchors=[c['synonyms'][0] for c in classes]
        for i,c in enumerate(classes):
            if i==e['background_index']:continue
            competitors=[a for j,a in enumerate(anchors) if i!=j]
            records.append(dict(dataset=d,index=i,name=c['name'],anchor=anchors[i],competitors=competitors,
                question=prompt(c['name'],anchors[i],competitors,e['family']),accepted=[],rejected=[],responses=[]))
    model=FrozenSemanticRoles(SOURCE);started=time.perf_counter();torch.manual_seed(20261007)
    result=dict(status='running',source_manifest=model.manifest,target_images_loaded=False,target_masks_loaded=False,
        target_label_tuning=False,records=records,generation=dict(do_sample=False,num_beams=1,max_new_tokens=512,
        batch_size=4,seed=20261007,maximum_format_attempts=2),semantic_quality_verified=False,
        rule='Preserve semantic anchor;0..12 optional typed additions. Exact competitor/duplicate/shared-addition checks only. Background/residual concepts unchanged; LLM denotation claims are not verified visual usefulness.')
    pending=list(range(len(records)))
    for attempt in range(2):
        failed=[]
        for start in range(0,len(pending),4):
            indices=pending[start:start+4]
            questions=[records[i]['question']+(' Your previous output was invalid. Follow the exact JSON schema.' if attempt else '') for i in indices]
            texts=[model.processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=q)])],tokenize=False,add_generation_prompt=True) for q in questions]
            inputs=model.processor.tokenizer(texts,padding=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated=model.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=512,use_cache=True,
                    pad_token_id=model.processor.tokenizer.pad_token_id)
            replies=model.processor.tokenizer.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
            for i,q,reply in zip(indices,questions,replies):
                record=records[i];answer=dict(prompt=q,raw_response=reply,attempt=attempt)
                try:
                    record['accepted'],record['rejected']=parse_reply(reply,record['anchor'],record['competitors'])
                    record['format_complete']=True
                except (ValueError,TypeError) as ex:
                    answer['format_error']=str(ex);failed.append(i)
                record['responses'].append(answer)
            result.update(wall_seconds=time.perf_counter()-started,complete_classes=sum(r.get('format_complete',False) for r in records),total_classes=len(records))
            save(output,result);print(json.dumps({k:result[k] for k in ('complete_classes','total_classes','wall_seconds')}),flush=True)
        pending=failed
        if not pending:break
    if pending:
        result.update(status='failed',format_failed_indices=pending);save(output,result)
        raise RuntimeError('Generation format failed; preserve raw responses, do not replace with hand-picked aliases.')
    datasets={}
    for d,e in entries.items():
        cohort=[r for r in records if r['dataset']==d];remove_shared_additions(cohort)
        banks=copy.deepcopy(e['banks'])
        for bank in banks.values():
            for record in cohort:
                i=record['index'];anchor=bank['classes'][i]['synonyms'][0]
                bank['classes'][i]['synonyms']=[anchor]+[x['phrase'] for x in record['accepted']]
        datasets[d]=banks
    result.update(status='complete',datasets=datasets,weights_frozen=all(not p.requires_grad for p in model.model.parameters()),wall_seconds=time.perf_counter()-started)
    save(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.protocol,a.output)
