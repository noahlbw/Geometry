"""Copy/reconstruct existing text features on CPU, with no new text encoding."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main(root):
    protocol=json.loads((root/'protocol.json').read_text())
    for d,entry in protocol['datasets'].items():
        path=root/'text_cache'/(d+'.pt')
        if path.exists():raise RuntimeError('Existing selected text cache; preserve it.')
        classes=json.loads((root/'vocabularies'/(d+'.json')).read_text())['classes']
        aliases=tuple(a for row in classes for a in row['synonyms'])
        source=Path(entry['source_root'])
        if entry['source_kind']=='transfer_selected_cache':
            old=torch.load(source/'task_text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
            local,wide=old['local'],old['wide']
            identity=old['identity']['source']
            expected=tuple(a for row in json.loads((source/'vocabularies'/(d+'_roles.json')).read_text())['classes'] for a in row['synonyms'])
        else:
            old=torch.load(source/'text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
            extra=torch.load(source/'family_text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
            meta=json.loads((source/'vocabularies'/(d+'.json')).read_text())
            original=tuple(a for row in meta['Current20']['classes'] for a in row['synonyms'])
            expected=tuple(a for row in meta['Complete20']['classes'] for a in row['synonyms'])
            lookup={word:i for i,word in enumerate(original)}
            old_local,old_wide=old['encoded'][d]['local'],old['encoded'][d]['wide']
            local,wide=old_local.clone(),old_wide.clone()
            for i,word in enumerate(aliases):
                if word in lookup:
                    local[i]=old_local[lookup[word]];wide[i]=old_wide[lookup[word]]
                else:
                    wide[i]=extra['encoded'][word]
                    local[i]=F.normalize(wide[i].float().mean(0),dim=-1)
            identity=old['identity']
        if aliases!=expected or len(aliases)!=len(local) or not torch.isfinite(local).all() or not torch.isfinite(wide).all():
            raise RuntimeError('Original text bank reconstruction differs.')
        value=dict(local=local,wide=wide,aliases=aliases,classes=tuple(row['name'] for row in classes),
                   checkpoints=identity['checkpoints'],source_identity=identity,
                   vocabulary_sha256=entry['vocabulary_sha256'],template=entry['policy']['template'])
        torch.save(value,path)
        print(json.dumps(dict(dataset=d,aliases=len(aliases),new_text_forwards=0,cache=str(path),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
