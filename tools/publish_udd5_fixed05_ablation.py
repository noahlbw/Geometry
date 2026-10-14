"""Copy audited UDD5 evidence and source closure into the GitHub checkout."""
import json
from pathlib import Path
import shutil

from publish_query_slot_best_alias import source_dependencies,WORKSPACE

NAME='geov2_udd5_fixed05_ablation_20261014'


def main():
    root=WORKSPACE/'research'/NAME
    repo=WORKSPACE/'artifacts/geometry_github'
    summary=json.loads((root/'SUMMARY.json').read_text(encoding='utf-8'))
    if not summary['all_shards_audited'] or not summary['all_per_image_sums_verified'] or summary['images']!=40:
        raise ValueError('Publish only completed audited evidence.')
    roots=[WORKSPACE/'DINOtool/scripts'/name for name in
           ('eval_geov2_udd5_fixed05.py','eval_geov2_udd5_writer_controls.py')]
    files=source_dependencies(roots)
    files.add(WORKSPACE/'tools/report_udd5_fixed05_ablation.py')
    files.add(Path(__file__).resolve())
    for path in root.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if any(part.startswith('smoke_failed') for part in path.parts):
            continue
        if path.suffix in ('.json','.npz','.csv','.md','.png','.pdf','.yaml','.py','.sh'):
            files.add(path)
    changed=[]
    for src in sorted(files):
        dst=repo/src.relative_to(WORKSPACE)
        if dst.exists() and dst.read_bytes()==src.read_bytes():
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        changed.append(str(src.relative_to(WORKSPACE)))
    print(json.dumps(dict(copied=len(changed),files=changed),indent=2))


if __name__=='__main__':
    main()
