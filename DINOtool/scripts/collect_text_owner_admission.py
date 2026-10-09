"""Text-only admission on the previously frozen Qwen candidate pool."""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import eval_development_readout as base
import run_evidence_adaptive_readout as paths
from dinotool.text_owner_admission import admit
from eval_rival_fine_full import save,frozen_state,check_frozen


@torch.inference_mode()
def main(root):
    output=root/'admission.json'
    if output.exists() or not paths.idle(int(os.environ['CUDA_VISIBLE_DEVICES'])):raise RuntimeError('Existing output or busy GPU.')
    protocol=json.loads((root/'protocol.json').read_text());pool=json.loads((root/'pool.json').read_text())
    results={};checks={};diagnostics={}
    for d,e in protocol['datasets'].items():
        entry=copy.deepcopy(e);entry['banks']=pool['datasets'][d]
        args=SimpleNamespace(dataset=d,suite_root=str(root),device='cuda',mode='smoke',
            original_cache=str(paths.OLD/'text_cache'/(d+'.pt')),dinov3_repo=str(paths.TOOL/'dinov3_hub'),
            checkpoint_dir=str(paths.BASE/'ckpt/DINO'),upstream_root=str(paths.BASE/'third_party/VIP_official_5bd25ee'))
        needed=('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
        geometry,vip,banks,queries,_=base.load_models(args,entry,needed)
        frozen=frozen_state(geometry,vip);output_banks=copy.deepcopy(entry['banks']);diagnostics[d]={}
        for k,bank in banks.items():
            wide=queries[k].features.float().mean(1)
            keep,details=admit(bank.features,wide,bank.parent_indices,bank.canonical_mask,bank.class_count,entry['background_index'])
            records=[];offset=0
            for c,cls in enumerate(output_banks[k]['classes']):
                aliases=cls['synonyms'];selected=[]
                for j,alias in enumerate(aliases,offset):
                    if bool(keep[j]):selected.append(alias)
                    records.append(dict(class_name=cls['name'],alias=alias,keep=bool(keep[j]),
                        canonical=bool(bank.canonical_mask[j]),local_margin=float(details['local_margin'][j]),
                        wide_margin=float(details['wide_margin'][j]),local_rival=bank.class_names[int(details['local_rival'][j])],
                        wide_rival=bank.class_names[int(details['wide_rival'][j])]))
                cls['synonyms']=selected;offset+=len(aliases)
            diagnostics[d][k]=records
        checks[d]=check_frozen(frozen,geometry,vip);results[d]=output_banks
        save(output,dict(status='running',datasets=results,diagnostics=diagnostics,checks=checks,
            target_images_loaded=False,target_masks_loaded=False,target_label_tuning=False))
        print(json.dumps(dict(dataset=d,counts={k:[len(c['synonyms']) for c in b['classes']] for k,b in output_banks.items() if k in needed})),flush=True)
        del geometry,vip,banks,queries,frozen;torch.cuda.empty_cache()
    save(output,dict(status='complete',datasets=results,diagnostics=diagnostics,checks=checks,
        target_images_loaded=False,target_masks_loaded=False,target_label_tuning=False,
        rule='Alias must have strictly greater own-anchor cosine than every foreground rival in both retained local and wide text representations. Canonical/background queries protected. Zero is the comparison boundary, not a fitted confidence threshold. Natural local/wide representations need not be independent.',
        semantic_quality_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
