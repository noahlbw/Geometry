"""Publish the model dependency closure and completed comparison evidence."""
import argparse
import ast
import json
from pathlib import Path
import shutil

WORKSPACE = Path(__file__).resolve().parents[1]


def source_dependencies(roots):
    pending, found = list(roots), set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        package = list(path.relative_to(WORKSPACE/'DINOtool').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    module = '.'.join(package[:len(package)-node.level+1] + module.split('.')).strip('.')
                modules = [module] + [module+'.'+alias.name for alias in node.names]
            for module in modules:
                candidates = [WORKSPACE/'DINOtool'/Path(*module.split('.')).with_suffix('.py'),
                    WORKSPACE/'DINOtool/scripts'/Path(*module.split('.')).with_suffix('.py')]
                for candidate in candidates:
                    if candidate.is_file() and candidate not in found:
                        pending.append(candidate)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=WORKSPACE/'artifacts/geometry_github')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo/'.git').exists():
        raise RuntimeError('An existing publication checkout is required.')
    suite = WORKSPACE/'research/query_slot_best_alias_20261013'
    result = json.loads((suite/'RESULTS.json').read_text(encoding='utf-8'))
    assert result['status']=='complete' and len(result['rows'])==16
    roots = [WORKSPACE/'DINOtool'/relative for relative in (
        'dinotool/frozen_deployment.py',
        'scripts/infer_frozen_geometry.py', 'scripts/evaluate_frozen_geometry.py',
        'scripts/eval_query_slot_best_alias.py', 'scripts/run_query_slot_best_alias.py',
        'tests/test_query_slot_memory.py', 'tests/test_context_memory_observer.py',
        'tests/test_geometry_semantic_response.py')]
    files = source_dependencies(roots)
    files.update([WORKSPACE/'tools/report_query_slot_best_alias.py', Path(__file__).resolve()])
    files.update(path for path in suite.rglob('*') if path.is_file()
        and path.suffix in ('.json', '.csv', '.md', '.txt', '.yaml', '.npz', '.py')
        and '__pycache__' not in path.parts and '.progress.' not in path.name
        and path.name != 'EVIDENCE_README.txt')
    changed = []
    for source in sorted(files):
        relative = source.relative_to(WORKSPACE)
        target = repo/relative
        if target.exists() and target.read_bytes()==source.read_bytes():
            continue
        changed.append(relative.as_posix())
        if not args.dry_run:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    print(json.dumps(dict(dependency_sources=len(source_dependencies(roots)),
        copied_files=len(changed), dry_run=args.dry_run,
        files=changed if args.dry_run else None), ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
