"""Copy only reproducibility sources and verified benchmark artifacts to release."""
import argparse
from pathlib import Path
import shutil

WORKSPACE=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=WORKSPACE/'artifacts/geometry_github')
    args=parser.parse_args();repo=args.repo.resolve()
    if not (repo/'.git').exists():raise RuntimeError('Existing publication checkout required')
    files=[WORKSPACE/'tools/training_free_baseline_experiment.py',
        WORKSPACE/'tools/fetch_training_free_baselines.py',Path(__file__).resolve(),
        WORKSPACE/'research/TRAINING_FREE_BASELINES_FULL_20261011.md']
    files+=list((WORKSPACE/'DINOtool/scripts').glob('*training_free*.py'))
    files+=list((WORKSPACE/'DINOtool/scripts').glob('*training_free*.sh'))
    root=WORKSPACE/'research/training_free_baselines_full_20261011'
    # No datasets, pretrained weights, features, credentials or third-party code.
    files+=list(root.rglob('*.json'))+list((root/'words').rglob('*.txt'))
    efficient=WORKSPACE/'research/efficient_geometry_20261012'
    if (efficient/'comparison_summary.json').exists():
        files += [WORKSPACE/'tools/report_efficient_geometry.py',
            WORKSPACE/'research/EFFICIENT_GEOMETRY_20261012.md']
        files += list(efficient.glob('*.json'))
        for folder in ('cost','merged','profile','smoke'):
            files += list((efficient/folder).glob('*.json'))
    context=WORKSPACE/'research/context_corrclip_tuning_20261012'
    if (context/'comparison_summary.json').exists():
        files += [WORKSPACE/'research/CONTEXT_CORRCLIP_TUNING_20261012.md',context/'README.md',
            WORKSPACE/'tools/context_corrclip_tuning_experiment.py']
        files += list(context.rglob('*.json'))+list((context/'configs').glob('*.yaml'))
        for relative in ('dinotool/efficient_deployment.py','dinotool/compact_deployment.py',
            'dinotool/natural_static_search.py','dinotool/frozen_deployment.py',
            'scripts/infer_frozen_geometry.py','scripts/evaluate_frozen_geometry.py',
            'scripts/eval_efficient_geometry.py','scripts/run_efficient_geometry.py',
            'scripts/eval_curated_variable_natural.py','scripts/eval_context_corrclip_tuning.py',
            'scripts/run_context_corrclip_tuning.py','tests/test_efficient_deployment.py'):
            files.append(WORKSPACE/'DINOtool'/relative)
    for source in files:
        relative=source.relative_to(WORKSPACE);target=repo/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    print(f'Synchronized {len(files)} scoped source/config/result files to {repo}')


if __name__=='__main__':main()
