"""Convert original full-category ground truth, never classifier-predicted Context masks."""
import argparse
import io
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat

from dinotool.natural_evaluation import discover_samples
from dinotool.pascal_context_ground_truth import SOURCE_IDS, CLASSES, map_full_category_mask, validate_label_dictionary
from eval_rival_fine_full import save


def main(root, archive_path, output):
    if output.exists():
        raise RuntimeError('Existing completed preparation manifest; no repeated conversion.')
    started = time.perf_counter()
    keys = (root/'ImageSets/Main/val.txt').read_text().split()
    train = (root/'ImageSets/Main/train.txt').read_text().split()
    all_keys = (root/'ImageSets/Main/trainval.txt').read_text().split()
    if (len(keys) != 5105 or len(train) != 4998 or len(all_keys) != 10103
            or len(set(keys)) != len(keys) or set(keys) & set(train)
            or set(keys) | set(train) != set(all_keys)):
        raise RuntimeError('Official VOC2010 Context train/validation split differs.')
    masks = root/'SegmentationClassContext'
    masks.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        validate_label_dictionary(archive.extractfile('labels.txt').read().decode())
        members = {Path(m.name).stem: m for m in archive.getmembers() if m.isfile() and m.name.endswith('.mat')}
        if len(members) != 10103 or set(members) != set(all_keys):
            raise RuntimeError('Original MAT annotation IDs differ from the official image split.')
        for number, key in enumerate(keys, 1):
            raw = loadmat(io.BytesIO(archive.extractfile(members[key]).read()))['LabelMap']
            mapped = map_full_category_mask(raw)
            with Image.open(root/'JPEGImages'/(key+'.jpg')) as image:
                if image.size != (mapped.shape[1], mapped.shape[0]):
                    raise RuntimeError('Full image/mask shape mismatch: '+key)
                image.verify()
            path = masks/(key+'.png')
            if path.exists():
                with Image.open(path) as previous:
                    if not np.array_equal(np.asarray(previous), mapped):
                        raise RuntimeError('Existing mask differs; preserve it: '+str(path))
            else:
                Image.fromarray(mapped).save(path)
            if number % 100 == 0 or number == len(keys):
                print('Verified original Context image/mask '+str(number)+'/'+str(len(keys)), flush=True)
    split = root/'ImageSets/SegmentationContext/val.txt'
    split.parent.mkdir(parents=True, exist_ok=True)
    if split.exists() and split.read_text().split() != keys:
        raise RuntimeError('Existing Context validation split differs; preserve it.')
    if not split.exists():
        split.write_text('\n'.join(keys)+'\n')
    for protocol in ('context59', 'context60'):
        discover_samples(protocol, root)
    save(output, dict(status='ready', root=str(root), protocols=['Context59', 'Context60'],
        evaluation_images=len(keys), pairs_verified=True, original_full_category_masks=True,
        prediction_masks_used=False, raw_mask_archive=str(archive_path), source_raw_ids=SOURCE_IDS,
        classes=CLASSES, split='Official VOC2010 ImageSets/Main/val.txt; all10103 IDs match original MAT archive.',
        mapping='MMSeg sorted60 raw IDs. Other original concepts become residual background0; Context59 ignores0.',
        mapping_reference='https://github.com/open-mmlab/mmsegmentation/blob/main/tools/dataset_converters/pascal_context.py',
        validation_only=True, wall_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.archive, args.output)
