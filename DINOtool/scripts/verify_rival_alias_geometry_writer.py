"""Explain legacy Geometry interpolation from stored scores, without model forwards."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from eval_rival_alias_speed_soft import PREVIOUS


def main(args):
    root = Path(args.output_dir)
    output = root/'geometry_writer_audit.json'
    if output.exists():
        raise RuntimeError('Refusing existing writer audit.')
    row = json.loads((root/'results.json').read_text())
    samples, specs, _, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    differences = {}
    with np.load(PREVIOUS/args.dataset/'per_image_confusions.npz', allow_pickle=False) as historical, np.load(
            root/'per_image_confusions.npz', allow_pickle=False) as current:
        for p, group in row['metrics'].items():
            classes = len(group['Geometry']['per_class'])
            current_cm, legacy_cm = [], []
            for i, key in enumerate(row['sample_keys']):
                sample = lookup[key]
                with Image.open(sample.image_path) as image:
                    width, height = image.size
                with np.load(root/'scores'/f'{i}.npz', allow_pickle=False) as cached:
                    scores = torch.from_numpy(cached[p+'__Geometry'])
                with np.load(PREVIOUS/args.dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as cached:
                    original = torch.from_numpy(cached['clean__'+p+'__Geometry__scores'])
                if not torch.equal(scores, original.double()):
                    raise RuntimeError('Original Geometry source scores differ.')
                scores = scores.to(device=args.device, dtype=original.dtype)
                pred = F.interpolate(scores.T.reshape(1, classes, 32, 32), (512, 512),
                                     mode='bilinear', align_corners=False)[0].argmax(0).cpu().numpy()
                target = load_mask(sample, p, (height, width))[:512, :512]
                pred = pred[:target.shape[0], :target.shape[1]]
                valid = (target >= 0) & (target < classes)
                encoded = target[valid].astype(np.int64)*classes + pred[valid]
                legacy_cm.append(np.bincount(encoded, minlength=classes**2).reshape(classes, classes))
                current_cm.append(current[p+'__Geometry'][i])
            legacy_cm, current_cm = np.stack(legacy_cm), np.stack(current_cm)
            if not np.array_equal(legacy_cm, historical['clean__'+p+'__Geometry']):
                raise RuntimeError('The readout precision/order explanation does not reproduce legacy Geometry.')
            differences[p] = {'legacy_writer_reproduces_every_historical_per_image_confusion': True,
                              'current_vs_legacy_per_image_confusion_l1': int(np.abs(legacy_cm-current_cm).sum())}
    output.write_text(json.dumps({'status': 'complete', 'new_visual_forwards': 0,
        'source': 'stored pre-mask Geometry scores only', 'differences': differences,
        'legacy_score_dtype': str(original.dtype),
        'explanation': 'Legacy: interpolate cosine/.07 in the archived score dtype. Current full evaluator: interpolate cosine in float32, then divide by.07.'}, indent=2)+'\n')
    print(json.dumps(differences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'data-root', 'vocabulary-config', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
