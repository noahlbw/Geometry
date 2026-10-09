"""Image-only audit: grounded foreign-class phrases can evade view reversal."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(root, original):
    domains = {}
    for dataset in ('vdd', 'potsdam', 'udd5', 'oem', 'loveda', 'vaihingen', 'landcoverai', 'flair1'):
        result = json.loads((root/dataset/'merged.json').read_text())
        vocab = result['vocabularies']['wrong_parent']
        if not result['coverage_verified'] or result['processed_images'] != 8:
            raise ValueError('Verified prior vocabulary screen required.')
        counts = {}
        for p, entry in vocab['replacements'].items():
            classes = vocab['vocabularies'][p]
            names = [c['name'] for c in classes]
            counts[p] = {}
            for change in entry['replacements']:
                c, d = names.index(change['class']), names.index(change['declared_rival'])
                if any(classes[c]['synonyms'][slot] != word for slot, word in zip(change['slots'], change['after'])):
                    raise ValueError('Changed constructed wrong-parent mapping.')
                counts[p][change['class']] = {'declared_rival': names[d], 'words': change['after'],
                    'alias_indices': [20*c+slot for slot in change['slots']], 'rival_index': d,
                    'query_alias_count': 0, 'broad_advantage_count': 0, 'fine_agreement_count': 0,
                    'fine_reversal_count': 0, 'positive_risk_count': 0, 'risk_sum_on_broad_advantage': 0.}
        for i in range(8):
            with np.load(original/dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as anchor:
                valid = anchor['valid'].astype(bool)
            with np.load(root/dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as cache:
                for p, classes in counts.items():
                    prefix = 'wrong_parent__'+p+'__'
                    broad, fine, risk = [cache[prefix+k] for k in ('broad_margin', 'native_margin', 'risk')]
                    for row in classes.values():
                        ids, d = row['alias_indices'], row['rival_index']
                        b, f, r = [value[valid][:, ids, d] for value in (broad, fine, risk)]
                        active = b > 0
                        agree = active & (f >= 0)
                        if np.any(r[agree] != 0):
                            raise ValueError('View-risk identity differs.')
                        row['query_alias_count'] += int(b.size)
                        row['broad_advantage_count'] += int(active.sum())
                        row['fine_agreement_count'] += int(agree.sum())
                        row['fine_reversal_count'] += int((active & (f < 0)).sum())
                        row['positive_risk_count'] += int((r > 0).sum())
                        row['risk_sum_on_broad_advantage'] += float(r[active].astype(np.float64).sum())
        for classes in counts.values():
            for row in classes.values():
                n = row['broad_advantage_count']
                row['fine_agreement_retained_fraction'] = row['fine_agreement_count']/n if n else None
                row['mean_risk_on_broad_advantage'] = row['risk_sum_on_broad_advantage']/n if n else None
        domains[dataset] = counts
    return {'source': str(root), 'target_masks_loaded': False, 'new_network_forwards': 0,
            'scope': 'constructed attachment labels only; not pixel harm or semantic correctness', 'domains': domains}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.original), indent=2))
