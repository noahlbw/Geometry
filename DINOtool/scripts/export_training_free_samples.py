"""Export the existing matched evaluation manifest without opening any masks."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace


def discover(dataset,entry):
    root=Path(entry['data_root'])
    names=entry.get('class_names') or [c['name'] for c in entry['banks']['original']['classes']]
    if entry['family']=='natural':
        from dinotool.natural_evaluation import discover_samples
        samples=discover_samples(dataset,root)
    elif dataset in ('vdd','potsdam','vaihingen'):
        from dinotool.rs_external import discover_samples
        samples=discover_samples(dataset,root)
    elif dataset in ('flair1','landcoverai'):
        from dinotool.gear_datasets import discover_samples
        samples=discover_samples(dataset,root)
    elif dataset=='oem':
        from dinotool.oem import discover_oem_samples
        samples=discover_oem_samples(root,'val',allow_missing_images=True)
    elif dataset=='udd5':
        samples=[]
        for line in (root/'metadata/val.txt').read_text().splitlines():
            image,mask=(root/part for part in line.split())
            samples.append(SimpleNamespace(key=image.name,image_path=image.resolve(),mask_path=mask.resolve()))
    elif dataset=='loveda':
        def order(value):
            p=Path(value)
            return (str(p.parent),0,int(p.stem)) if p.stem.isdigit() else (str(p.parent),1,str(value))
        pairs=[]
        for image_dir in sorted(root.rglob('images_png')):
            mask_dir=image_dir.parent/'masks_png'
            if image_dir.is_dir() and mask_dir.is_dir():pairs.append((image_dir,mask_dir))
        if not pairs:
            for image_dir in sorted(root.rglob('images')):
                mask_dir=image_dir.parent/'masks'
                if image_dir.is_dir() and mask_dir.is_dir():pairs.append((image_dir,mask_dir))
        samples=[]
        for image_dir,mask_dir in pairs:
            for image in sorted(image_dir.glob('*.png'),key=lambda p:order(p.name)):
                key=(image_dir.parent.relative_to(root)/image.name).as_posix()
                samples.append(SimpleNamespace(key=key,image_path=image,mask_path=mask_dir/image.name))
        samples.sort(key=lambda s:order(s.key));random.Random(20260923).shuffle(samples)
    else:raise ValueError('Unknown matched dataset: '+dataset)
    if any(not s.image_path.is_file() or not s.mask_path.is_file() for s in samples):
        raise FileNotFoundError('Missing paired input/target path; no masks were opened.')
    return samples,({'P':names[1:],'D':names} if dataset=='loveda' else {dataset:names})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    out = {}
    for dataset, entry in source['datasets'].items():
        samples,names=discover(dataset,entry)
        rows = [dict(key=s.key, image_path=str(s.image_path), mask_path=str(s.mask_path)) for s in samples]
        keys = [r['key'] for r in rows]
        if len(samples) != entry['total_images'] or len(set(keys)) != len(rows):
            raise RuntimeError('Invalid full manifest: '+dataset)
        out[dataset] = dict(total_images=len(rows), classes=names, samples=rows,
            global_sample_keys_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest())
    target = Path(args.output)
    out = json.loads(json.dumps(out))
    if target.exists():
        if json.loads(target.read_text()) != out:
            raise RuntimeError('Existing matched sample manifest differs.')
    else:
        target.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v['total_images'] for k, v in out.items()}))


if __name__ == '__main__':
    main()
