"""Audit-only raw residual composition after the candidate ontology is frozen."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import numpy as np
from scipy.io import loadmat


def main(args):
    root=Path(args.experiment_root)
    output=Path(args.output)
    if output.exists():
        raise RuntimeError('Preserve existing audit.')
    protocol=json.loads((root/'protocol.json').read_text())
    identity=protocol.get('residual_identity')
    if identity is None or not (root/'residual_cache.pt').exists():
        raise RuntimeError('Vocabulary must be frozen before opening raw masks.')
    baseline=json.loads(Path(args.reference).read_text())
    wanted=set(baseline['signature']['sample_keys'])
    if len(wanted)!=5105:
        raise RuntimeError('Full PC60 reference required.')
    counts=np.zeros(460,dtype=np.int64)
    seen=set()
    with tarfile.open(protocol['metadata_archive'],'r:gz') as archive:
        metadata=archive.extractfile('labels.txt').read().decode()
        labels={int(line.split(':',1)[0]):line.split(':',1)[1].strip() for line in metadata.splitlines() if line.strip()}
        for member in archive:
            key=Path(member.name).stem
            if not member.name.endswith('.mat') or key not in wanted:
                continue
            if key in seen:
                raise RuntimeError('Duplicate original mask key.')
            raw=loadmat(io.BytesIO(archive.extractfile(member).read()))['LabelMap']
            if raw.min()<0 or raw.max()>459:
                raise RuntimeError('Unexpected original category ID.')
            counts+=np.bincount(raw.astype(np.int64).ravel(),minlength=460)
            seen.add(key)
    if seen!=wanted:
        raise RuntimeError('Incomplete original-mask coverage.')
    scored=set(protocol['scored_raw_ids'])-{0}
    residual_ids=[i for i in range(460) if i not in scored]
    residual_total=int(counts[residual_ids].sum())
    confusion=np.asarray(baseline['metrics']['context60']['TaxonomyAdaptive']['confusion_matrix'])
    if residual_total!=int(confusion[0].sum()) or int(counts.sum())!=int(confusion.sum()):
        raise RuntimeError('Raw/reduced evaluation pixel totals differ.')
    rows=sorted([dict(raw_id=i,name=labels.get(i,'unspecified/unannotated'),pixels=int(counts[i]),
        residual_fraction=float(counts[i]/residual_total)) for i in residual_ids],key=lambda v:-v['pixels'])
    result=dict(status='complete',images=len(seen),raw_reduced_pixel_totals_exact=True,
        residual_pixels=residual_total,raw_zero_fraction=float(counts[0]/residual_total),
        total_pixels=int(counts.sum()),unscored_named_fraction=float(counts[1:][[i not in scored for i in range(1,460)]].sum()/residual_total),
        residual_categories=rows,audit_only=True,vocabulary_frozen_before_masks=True,
        masks_used_for_candidate_selection=False,source=protocol['metadata_archive'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='residual_categories'}))
    print(json.dumps(rows[:12]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True)
    main(p.parse_args())
