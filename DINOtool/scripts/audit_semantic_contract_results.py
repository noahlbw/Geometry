"""Preserve failed strict replay; audit CPU/GPU text normalization without rerunning images."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_rival_fine_full import save
from run_sat_geometry_transport_suite import read_json
from run_semantic_contract_alias import verify
from run_region_semantic_suite_a800 import idle


def main(root, gpu):
    if not idle(gpu):raise RuntimeError('Use an idle GPU for text-only numerical audit.')
    if (root/'suite_results.json').exists():raise RuntimeError('Existing final suite; preserve it.')
    protocol=read_json(root/'protocol.json');d='ade150';entry=protocol['datasets'][d]
    dest=root/'full'/d/'verified_merged.json'
    if dest.exists():raise RuntimeError('Existing audit result; preserve it.')
    current=read_json(root/'full'/d/'merged.json')
    previous=read_json(Path(entry['reference']))
    if (not current['coverage_verified'] or current['processed_images']!=2000
            or current['signature']['sample_keys']!=previous['signature']['sample_keys']
            or current['signature']['checkpoints']!=previous['signature']['checkpoints']):
        raise RuntimeError('Full ADE inventory/checkpoint mismatch.')
    selected=torch.load(root/'text_cache'/ 'ade150.pt',map_location='cpu',weights_only=True)
    source=Path(entry['source_root'])
    old=torch.load(source/'text_cache/ade150.pt',map_location='cpu',weights_only=True)
    extra=torch.load(source/'family_text_cache/ade150.pt',map_location='cpu',weights_only=True)
    meta=read_json(source/'vocabularies/ade150.json')
    original=tuple(a for row in meta['Current20']['classes'] for a in row['synonyms'])
    expected=tuple(a for row in meta['Complete20']['classes'] for a in row['synonyms'])
    if tuple(selected['aliases'])!=expected:raise RuntimeError('Text strings differ.')
    device=torch.device('cuda');lookup={word:i for i,word in enumerate(original)}
    old_local=old['encoded'][d]['local'].to(device);old_wide=old['encoded'][d]['wide'].to(device)
    local,wide=old_local.clone(),old_wide.clone()
    new_ids=[]
    for i,word in enumerate(expected):
        if word in lookup:
            local[i]=old_local[lookup[word]];wide[i]=old_wide[lookup[word]]
        else:
            wide[i]=extra['encoded'][word].to(device)
            local[i]=F.normalize(wide[i].float().mean(0),dim=-1)
            new_ids.append(i)
    difference=(local.cpu()-selected['local']).abs()
    changed_rows=(difference>0).any(-1).nonzero().flatten().tolist()
    outside_new=[i for i in changed_rows if i not in new_ids]
    if (not torch.equal(wide.cpu(),selected['wide']) or outside_new
            or float(difference.max())>1e-6):
        raise RuntimeError('Difference is not bounded CPU/GPU normalization of new text rows.')
    keys=current['signature']['sample_keys'];lookup={k:i for i,k in enumerate(keys)};nc=150
    base=np.empty((len(keys),nc,nc),np.int64);primary=np.empty_like(base);seen=[]
    for path in current['shards']:
        worker=read_json(Path(path))
        if not worker['weights_frozen'] or not worker['head_weights_unchanged']:
            raise RuntimeError('Model weights changed.')
        with np.load(Path(path).parent/'per_image_confusions.npz',allow_pickle=False) as a:
            sk=a['sample_keys'].tolist();ix=[lookup[k] for k in sk];seen+=sk
            base[ix]=a[d+'__Same20_NoContract'];primary[ix]=a[d+'__Same20_ContractExcess']
    historical=np.empty_like(base)
    for path in previous['shards']:
        with np.load(Path(path).parent/'per_image_confusions.npz',allow_pickle=False) as a:
            sk=a['sample_keys'].tolist();historical[[lookup[k] for k in sk]]=a[d+'__'+entry['reference_method']]
    if (len(seen)!=len(set(seen)) or set(seen)!=set(keys)
            or not np.array_equal(base.sum(0),current['metrics'][d]['Same20_NoContract']['confusion_matrix'])
            or not np.array_equal(primary.sum(0),current['metrics'][d]['Same20_ContractExcess']['confusion_matrix'])
            or not np.array_equal(base.sum(-1),primary.sum(-1))
            or not np.array_equal(base.sum(-1),historical.sum(-1))):
        raise RuntimeError('Paired targets/confusion/coverage mismatch.')
    changed_cm=(base!=historical).any(axis=(1,2))
    audit=dict(status='complete',target_images_loaded=False,target_masks_loaded=False,image_encodings=0,
        source_bank_strings_exact=True,source_wide_features_exact=True,
        original_local_rows_exact=True,new_text_rows=len(new_ids),
        cpu_gpu_changed_local_rows=len(changed_rows),maximum_feature_difference=float(difference.max()),
        cause='New text rows were normalized on CPU when copying the existing bank; historical reader normalized those same rows on GPU. Only those FP32 rows differ; wide features and original rows are bit-identical.',
        historical_baseline_exact=False,paired_off_on_same_text_cache=True,
        historical_confusion_changed_images=int(changed_cm.sum()),historical_confusion_l1=int(np.abs(base-historical).sum()),
        changed_sample_keys=[keys[i] for i in np.flatnonzero(changed_cm)],
        baseline_miou=current['metrics'][d]['Same20_NoContract']['mean_iou_percent'],
        historical_miou=previous['metrics'][d][entry['reference_method']]['mean_iou_percent'],
        limitation='This establishes a bounded text representation difference, not equality of historical predictions or an exact count of changed pixels. Current paired off/on results remain valid; historical replay requirement failed.')
    save(root/'text_normalization_audit.json',audit)
    current.update(exact_predecessor_per_image_replayed=False,paired_scored_targets_equal=True,
        per_image_confusion_sums_verified=True,text_normalization_audit=str(root/'text_normalization_audit.json'))
    save(dest,current)
    datasets={d:dict(merged=str(dest),processed_images=2000)}
    for name in ('vdd','potsdam','voc20'):
        path=root/'full'/name/'merged.json'
        row=read_json(path)
        if row is None:row=verify(root,protocol,name)
        if not row.get('exact_predecessor_per_image_replayed'):raise RuntimeError('Unexpected other-domain replay failure.')
        datasets[name]=dict(merged=str(path),processed_images=row['processed_images'])
    result=dict(status='complete',outcome='paired_full_verified',processed_images=4033,datasets=datasets,
        historical_replay_all_exact=False,text_normalization_audit=audit,
        recovery='All image evaluations completed once. No inference relaunch or overwritten outputs. Preserve original failed controller and ADE merged.json; publish separately audited ADE verified_merged.json.')
    save(root/'suite_results.json',result)
    print(json.dumps(dict(audit=audit,status='complete',images=4033)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args();main(args.root,args.gpu)
