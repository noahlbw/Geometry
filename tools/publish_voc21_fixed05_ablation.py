"""Copy the audited VOC21 experiment and model source closure to the checkout."""
import json
from pathlib import Path
import shutil

from publish_query_slot_best_alias import source_dependencies,WORKSPACE

NAME='geov2_voc21_fixed05_ablation_20261014'


def main():
    root=WORKSPACE/'research'/NAME
    repo=WORKSPACE/'artifacts/geometry_github'
    summary=json.loads((root/'SUMMARY.json').read_text(encoding='utf-8'))
    if summary['images']!=1449 or not summary['all_shards_audited'] or not summary['all_per_image_sums_verified']:
        raise ValueError('Publish only audited complete evidence.')
    for subdir in ('configs','vocabularies'):
        suffix='.yaml' if subdir=='configs' else '.json'
        source=WORKSPACE/'research/query_slot_best_alias_20261013'/subdir/('voc21'+suffix)
        dest=root/subdir/source.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    references={
        'g0_previous_full.json':WORKSPACE/'research/geov2_g0_sweep_20261014/voc21_audited.json',
        'udd5_summary.json':WORKSPACE/'research/geov2_udd5_fixed05_ablation_20261014/SUMMARY.json',
    }
    for name,source in references.items():
        dest=root/'reference'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    roots=[WORKSPACE/'DINOtool/scripts/eval_geov2_fixed05_components.py']
    files=source_dependencies(roots)
    files.update([WORKSPACE/'tools/report_voc21_fixed05_ablation.py',Path(__file__).resolve()])
    for path in root.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.json','.npz','.csv','.md','.png','.pdf','.yaml','.py','.sh'):
            files.add(path)
    changed=[]
    for source in sorted(files):
        dest=repo/source.relative_to(WORKSPACE)
        if dest.exists() and dest.read_bytes()==source.read_bytes():
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        changed.append(source.relative_to(WORKSPACE).as_posix())
    print(json.dumps(dict(copied=len(changed),files=changed),indent=2))


if __name__=='__main__':
    main()
