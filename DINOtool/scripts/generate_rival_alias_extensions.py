"""Generate a provenance-recorded nested40 bank without images or labels."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from dinotool.rival_alias_count import COUNTS, nested_classes, parse_additions
from dinotool.semantic_role_admission import FrozenSemanticRoles, SOURCE
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import TOOL, idle


PREVIOUS = TOOL/'results/rival_fine_coupling_20261003'
DATASETS = ('vdd', 'potsdam', 'udd5', 'oem', 'loveda', 'vaihingen', 'landcoverai', 'flair1')


def prompt(name, existing, needed):
    return ('Generate '+str(needed)+' additional short English candidate phrases for the semantic class '
        +json.dumps(name)+' in remote sensing semantic segmentation. Use ordinary language-model '
        'expansion: synonyms, subtypes and descriptive phrases. No image is provided. '
        'Do not repeat any phrase in this exclusion list: '+json.dumps(existing)+'. '
        'Return only a JSON array of '+str(needed)+' strings. No explanation or markdown.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    physical_gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    if args.output.exists() or not idle(physical_gpu):
        raise RuntimeError('Preserve existing vocabulary source and require the mapped GPU idle.')
    bases, requests, lookup = {}, [], {}
    for dataset in DATASETS:
        old = json.loads((PREVIOUS/dataset/'merged.json').read_text())
        if not old['coverage_verified'] or old['processed_images'] != 8:
            raise RuntimeError('Verified64-window source required.')
        bases[dataset] = old['vocabularies']['clean']['vocabularies']
        for p, classes in bases[dataset].items():
            lookup[dataset, p] = []
            for cls in classes:
                identity = (cls['name'], tuple(cls['synonyms']))
                if identity not in requests:
                    requests.append(identity)
                lookup[dataset, p].append(requests.index(identity))
    model = FrozenSemanticRoles(SOURCE)
    torch.manual_seed(20261003)
    result = {'status': 'running', 'source_manifest': model.manifest,
        'generation': {'do_sample': True, 'temperature': .7, 'top_p': .9, 'repetition_penalty': 1.1,
            'seed': 20261003, 'num_beams': 1, 'max_new_tokens': 768, 'batch_size': 4,
            'maximum_structural_attempts': 3}, 'target_images_loaded': False, 'target_masks_loaded': False,
        'semantic_or_visual_filtering': False, 'local_geometry_alias_count': 20,
        'context_alias_counts': list(COUNTS), 'records': [], 'datasets': {}}
    started = time.perf_counter()
    records = [{'name': name, 'base20': list(base), 'additional20': [], 'responses': []} for name, base in requests]
    for attempt in range(3):
        pending = [i for i, r in enumerate(records) if len(r['additional20']) < 20]
        if not pending:
            break
        for start in range(0, len(pending), 4):
            indices = pending[start:start+4]
            prompts = [prompt(records[i]['name'], records[i]['base20']+records[i]['additional20'],
                20-len(records[i]['additional20'])) for i in indices]
            texts = [model.processor.apply_chat_template([{'role': 'user', 'content':
                [{'type': 'text', 'text': value}]}], tokenize=False, add_generation_prompt=True) for value in prompts]
            inputs = model.processor.tokenizer(texts, padding=True, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.model.generate(**inputs, do_sample=True, temperature=.7, top_p=.9,
                    repetition_penalty=1.1, num_beams=1,
                    max_new_tokens=768, use_cache=True, pad_token_id=model.processor.tokenizer.pad_token_id)
            replies = model.processor.tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for i, question, reply in zip(indices, prompts, replies):
                record = records[i]
                entry = {'prompt': question, 'raw_response': reply, 'attempt': attempt}
                try:
                    additions = parse_additions(reply, record['base20']+record['additional20'], 20-len(record['additional20']))
                    record['additional20'] += additions
                    entry['accepted_structural_count'] = len(additions)
                except ValueError as error:
                    entry['format_error'] = str(error)
                record['responses'].append(entry)
            result['records'] = records
            result['wall_seconds'] = time.perf_counter()-started
            save(args.output, result)
            print(json.dumps({'complete_classes': sum(len(r['additional20']) == 20 for r in records),
                'total_classes': len(records), 'attempt': attempt}), flush=True)
    if any(len(r['additional20']) != 20 for r in records):
        result['status'] = 'failed'
        save(args.output, result)
        raise RuntimeError('Language output failed structural20-addition contract; preserve raw source.')
    for dataset, protocols in bases.items():
        result['datasets'][dataset] = {p: {str(k): nested_classes(classes,
            [records[i]['additional20'] for i in lookup[dataset, p]], k) for k in COUNTS}
            for p, classes in protocols.items()}
    result.update(status='complete', weights_frozen=all(not v.requires_grad for v in model.model.parameters()),
        wall_seconds=time.perf_counter()-started)
    save(args.output, result)
    print(json.dumps({'status': 'complete', 'classes': len(records), 'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
