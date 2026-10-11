"""Explicit, recorded compatibility preparation for official baseline sources."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-mmcv22', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = importlib.util.find_spec('mmseg')
    path = Path(spec.origin)
    before = path.read_bytes()
    if args.allow_mmcv22:
        text = before.decode()
        replacements = {"mmcv_max_version = '2.2.0'":"mmcv_max_version = '2.2.1'",
            "MMCV_MAX = '2.2.0'":"MMCV_MAX = '2.2.1'"}
        if not any(old in text or new in text for old,new in replacements.items()):
            raise RuntimeError('Unexpected MMSeg version guard; inspect instead of silently patching.')
        for old,new in replacements.items():text=text.replace(old,new)
        path.write_text(text)
    packages = {}
    for name in ('torch', 'torchvision', 'mmcv', 'mmengine', 'mmsegmentation', 'numpy', 'transformers', 'timm'):
        packages[name] = metadata.version(name)
    output = Path(args.output)
    existing = json.loads(output.read_text()) if output.exists() else {}
    output.write_text(json.dumps(dict(packages=packages, compatibility=dict(
        operation='MMCV version-guard upper bound 2.2.0 -> 2.2.1 only',
        before_sha256=existing.get('compatibility', {}).get('before_sha256', hashlib.sha256(before).hexdigest()),
        after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())), indent=2))


if __name__ == '__main__':
    main()
