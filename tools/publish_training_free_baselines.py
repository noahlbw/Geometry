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
    for source in files:
        relative=source.relative_to(WORKSPACE);target=repo/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    print(f'Synchronized {len(files)} scoped source/config/result files to {repo}')


if __name__=='__main__':main()
