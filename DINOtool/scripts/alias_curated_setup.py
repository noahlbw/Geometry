"""Load the previously frozen curated20 bank without altering class IDs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dinotool.geometry_execution import GeometryExecution
from dinotool.vip_official_adapter import VIPQueries
import eval_curated20_patchonly2 as natural


@torch.inference_mode()
def setup(args):
    protocol=json.loads((Path(args.suite_root)/'protocol.json').read_text())
    entry=protocol['datasets'][args.dataset]
    if args.dataset!='ade150' or entry['family']!='natural':
        raise RuntimeError('This input factorial is ADE150 only.')
    source=Path(protocol['word_source'])
    original=source/'vocabularies'/'ade150_original20.json'
    curated=source/'vocabularies'/'ade150_curated20.json'
    if str(curated)!=entry['vocabulary_config']:
        raise RuntimeError('Curated input path differs from the frozen protocol.')
    for arm,path in (('original20',original),('curated20',curated)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['input_sha256'][arm]:
            raise RuntimeError('Previously frozen input bank changed: '+arm)
    adapted=SimpleNamespace(**vars(args),original_vocabulary=str(original),
        curated_vocabulary=str(curated),text_cache=str(source/'text_cache'/'ade150.pt'),family='natural')
    samples,specs,loader,masks=natural.inputs(adapted)
    segmenter,all_banks,vip,all_queries,identity=natural.models(adapted,specs)
    key=args.dataset+'__curated20'
    bank,query=all_banks[key],all_queries[key]
    if len(samples)!=entry['total_images']:
        raise RuntimeError('Full ADE inventory count changed.')
    # Semantic anchor names may differ from the published display label.
    # Every anchor refers to an existing slot; no 21st alias is inserted.
    canonical=tuple(bank.alias_names[int(((bank.parent_indices==c)&bank.canonical_mask).nonzero().flatten()[0])]
                    for c in range(bank.class_count))
    query=VIPQueries(query.features,query.parents,canonical,query.aliases)
    if any(int((query.parents==c).sum())!=20 for c in range(bank.class_count)):
        raise RuntimeError('Curated bank no longer has twenty slots per class.')
    banks,queries={args.dataset:bank},{args.dataset:query}
    words=dict(sha256=identity['vocabulary_sha256']['curated20'],
        aliases={args.dataset:bank.alias_names},counts={args.dataset:[20]*bank.class_count})
    mask_loader=lambda sample,p,shape:natural.load_target(sample,args.dataset,shape)
    return (protocol,entry,samples,loader,mask_loader,GeometryExecution(segmenter),banks,vip,queries,
            words,identity['checkpoints'],None,None)
