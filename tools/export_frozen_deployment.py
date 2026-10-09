"""Export actual Kev-selected language/configuration and public replay inputs."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]


def export(root):
    source = WORKSPACE / 'research/kev_alias_search_20261009'
    protocol = json.loads((source / 'protocol.json').read_text())
    configdir, wordsdir = root / 'configs/frozen_20261010', root / 'vocabularies/frozen_20261010'
    configdir.mkdir(parents=True, exist_ok=True); wordsdir.mkdir(parents=True, exist_ok=True)
    for dataset, entry in protocol['datasets'].items():
        choice = json.loads((source / dataset / 'selection.json').read_text())
        bank = choice['profile']['bank']
        needed = ['semantic_segmentation'] if bank == '__previous_final__' else [bank]
        vocabulary = {k: choice['vocabulary'][k] for k in needed}
        wordpath = wordsdir / (dataset + '.json')
        wordpath.write_text(json.dumps(dict(dataset=dataset,banks=vocabulary,residual=entry.get('residual_identity')),
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (wordsdir / (dataset + '.txt')).write_text('\n'.join(c['name']+'\t'+' | '.join(c['synonyms'])
            for c in vocabulary[needed[0]]['classes'])+'\n',encoding='utf-8')
        config = dict(schema=1,dataset=dataset,family=entry['family'],total_images=entry['total_images'],
            background_index=entry['background_index'],vocabulary='../../vocabularies/frozen_20261010/'+dataset+'.json',
            profile=choice['profile'],background_bias=choice['background_bias'],background_threshold=choice['background_threshold'],
            geometry=dict(geometry_depth=2,geometry_temperature=.1,spatial_sigma=.25,maximum_aliases_per_class=20),
            visual_budget=dict(geometry_max=4,wide_max=4,fine=0),
            upstream=dict(vip_commit='5bd25ee03ec25c1538622cf7da661e8c0461e769',dinov3_commit='f6704c1b76137543328567c8f8b49a2dc318824c'),
            provenance=dict(labeled_development=True,prior_developed_data=True,selection_frozen=True,
                source='research/kev_alias_search_20261009/'+dataset+'/selection.json',
                caveat='This is a frozen task-specific exploratory configuration, not label-free adaptation.'))
        (configdir / (dataset+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    # Include the exact prompt source independently of the upstream import.
    templates=(WORKSPACE/'third_party/VIP_official/prompts/imagenet_template.py').read_text(encoding='utf-8')
    (wordsdir/'vip_templates.py').write_text('# VIP 5bd25ee prompt source; trailing whitespace normalized only.\n'+
        '\n'.join(line.rstrip() for line in templates.splitlines())+'\n',encoding='utf-8')
    from dinotool.prompts import REMOTE_SENSING_TEMPLATES
    (wordsdir/'local_rs6_templates.json').write_text(json.dumps(list(REMOTE_SENSING_TEMPLATES),indent=2)+'\n')
    (configdir/'assets.example.yaml').write_text(yaml.safe_dump(dict(
        dinov3_repo='DINOtool/dinov3_hub',checkpoint_dir='/path/to/checkpoints',
        upstream_root='/path/to/VIP',text_cache_dir='/path/to/generated-text-cache',
        datasets={d:'/path/to/datasets/'+d for d in protocol['datasets']}),sort_keys=False))
    print(json.dumps(dict(configs=len(protocol['datasets']),root=str(root))))


def stage(root):
    from prepare_geometry_release import SECRET
    copied=[]
    def copy(relative):
        source=WORKSPACE/relative
        if SECRET.search(source.read_text(encoding='utf-8',errors='replace')):
            raise RuntimeError('Credential-like material in '+str(relative))
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);copied.append(str(relative).replace('\\','/'))
    for folder in ('dinotool','scripts','tests','configs','dinov3_hub'):
        for source in (WORKSPACE/'DINOtool'/folder).rglob('*'):
            if source.is_file() and source.suffix in ('.py','.sh','.json','.toml','.yaml','.yml') and '__pycache__' not in source.parts:
                copy(source.relative_to(WORKSPACE))
    for folder in ('kev_alias_search_20261009','vip_kev_same_vocabulary_20261010','compact_deployment_20261010'):
        base=WORKSPACE/'research'/folder
        for name in ('protocol.json','comparison_summary.json'):
            if (base/name).is_file():copy((base/name).relative_to(WORKSPACE))
        for source in base.glob('*/merged.json'):copy(source.relative_to(WORKSPACE))
        for source in base.glob('*/cost.json'):copy(source.relative_to(WORKSPACE))
        if folder.startswith('kev_'):
            for source in base.glob('*/selection.json'):copy(source.relative_to(WORKSPACE))
        if folder.startswith('compact'):
            for source in base.glob('*.json'):copy(source.relative_to(WORKSPACE))
    for name in ('KEV_ALIAS_SEARCH_20261009.md','VIP_KEV_SAME_VOCABULARY_20261010.md','COMPACT_DEPLOYMENT_20261010.md'):
        if (WORKSPACE/'research'/name).exists():copy(Path('research')/name)
    for name in ('export_frozen_deployment.py','report_compact_deployment.py'):copy(Path('tools')/name)
    copy(Path('DINOtool/pyproject.toml'))
    manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text())
    manifest['deployment_update']=dict(date='2026-10-10',configurations=15,
        entry='DINOtool/scripts/infer_frozen_geometry.py',evaluation='DINOtool/scripts/evaluate_frozen_geometry.py',
        config_directory='configs/frozen_20261010',vocabulary_directory='vocabularies/frozen_20261010',
        files=copied,history='Original 2026-10-09 release and evidence retained.')
    (root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(staged_files=len(copied))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=WORKSPACE/'artifacts/geometry_github')
    parser.add_argument('--stage',action='store_true')
    args=parser.parse_args();export(args.root.resolve())
    if args.stage:stage(args.root.resolve())
