"""Freeze image-free alias/class NLI posteriors before any evaluation masks."""
import argparse
import json
from pathlib import Path
import time

import torch

from dinotool.prompts import load_class_specs
from dinotool.semantic_membership_alias import MODEL_ID, REVISION, TEMPLATE
from run_region_semantic_suite_a800 import SETTINGS, TOOL, idle


@torch.inference_mode()
def build(output, cache_dir, gpu):
    if output.exists() or not idle(gpu):
        raise RuntimeError('Preserve cache and use an idle GPU only.')
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache_dir,
                                             trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION,
        cache_dir=cache_dir, use_safetensors=True, trust_remote_code=False).eval().requires_grad_(False).to('cuda')
    mapping = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if mapping != {0: 'contradiction', 1: 'entailment', 2: 'neutral'}:
        raise RuntimeError('Verified upstream NLI label order changed.')
    loaded = time.perf_counter()
    pairs, seen, vocabularies = [], set(), {}
    for dataset, _, vocabulary, _, _ in SETTINGS:
        specs = load_class_specs(TOOL/'configs'/vocabulary)
        if any(len(s.synonyms) != 20 for s in specs):
            raise RuntimeError('Unchanged20 class pools required.')
        vocabularies[dataset] = [dict(name=s.name, aliases=list(s.synonyms)) for s in specs]
        for own in specs:
            for alias in own.synonyms:
                for rival in specs:
                    pair = alias, rival.name
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
    entries = []
    for start in range(0, len(pairs), 64):
        batch = pairs[start:start+64]
        inputs = tokenizer([TEMPLATE.format(phrase=a) for a, _ in batch],
                           [TEMPLATE.format(phrase=c) for _, c in batch],
                           padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        values = model(**inputs).logits.float().softmax(-1).cpu()
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError('Nonfinite text-only semantic source.')
        entries.extend(dict(pair=pair, probabilities=row.tolist()) for pair, row in zip(batch, values))
        if start % 1024 == 0:
            print(json.dumps(dict(text_pairs=min(start+64, len(pairs)), total=len(pairs))), flush=True)
    torch.cuda.synchronize()
    setup = dict(load_seconds=loaded-started, inference_seconds=time.perf_counter()-loaded,
        total_seconds=time.perf_counter()-started, text_pairs=len(pairs), gpu=gpu,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576)
    result = dict(model=MODEL_ID, revision=REVISION, template=TEMPLATE, label_order=mapping,
        target_images_loaded=False, target_masks_loaded=False, model_weights_frozen=True,
        vocabularies=vocabularies, entries=entries, setup=setup)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(setup), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    build(args.output, args.cache_dir, args.gpu)
