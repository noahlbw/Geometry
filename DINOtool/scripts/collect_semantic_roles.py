"""Mask-free vocabulary-time semantic role observations on one idle GPU."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from dinotool.semantic_role_admission import (CONFIG, IMPLEMENTATION, SOURCE, FrozenSemanticRoles,
    roles, semantic_conflict, role_prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocabularies', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve existing semantic source observations.')
    vocabularies = json.loads(args.vocabularies.read_text())
    query_keys, lookups = [], {}
    for scene, entry in vocabularies.items():
        for protocol, classes in entry['vocabularies'].items():
            names = tuple(c['name'] for c in classes)
            group = []
            for parent, cls in enumerate(classes):
                for alias in cls['synonyms']:
                    key = (names, parent, alias)
                    if key not in query_keys:
                        query_keys.append(key)
                    group.append(query_keys.index(key))
            lookups[scene+'__'+protocol] = group
    # Sentinels are text-only contract checks, not target-vocabulary tuning.
    sentinels = [(('wall', 'roof', 'road', 'tree'), 0, 'roof', 'competitor:roof'),
        (('building', 'road', 'tree', 'water'), 0, 'building', 'direct'),
        (('building', 'road', 'tree', 'water'), 0, 'concrete', 'shared_material'),
        (('building', 'road', 'tree', 'water'), 0, 'chimney', 'part'),
        (('building', 'road', 'tree', 'water'), 0, 'unidentified thing', 'unknown')]
    requests = [(*key, reverse) for key in query_keys for reverse in (False, True)]
    requests += [(*row[:3], reverse) for row in sentinels for reverse in (False, True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = FrozenSemanticRoles(SOURCE)
    torch.cuda.reset_peak_memory_stats()
    responses = []
    for start in range(0, len(requests), CONFIG.source_batch_size):
        responses += model.score(requests[start:start+CONFIG.source_batch_size])
        print(json.dumps({'processed_prompts': len(responses), 'total_prompts': len(requests)}), flush=True)
    sources = {}
    for key, indices in lookups.items():
        pairs = [[responses[2*i+j][0] for i in indices] for j in range(2)]
        names = query_keys[indices[0]][0]
        parents = torch.tensor([query_keys[i][1] for i in indices])
        conflict = semantic_conflict(torch.tensor(pairs), parents, len(names))
        choice = torch.tensor(pairs).argmax(-1)
        sources[key] = {'classes': list(names), 'aliases': [query_keys[i][2] for i in indices],
            'parents': parents.tolist(), 'role_logits': pairs, 'semantic_conflict': conflict.tolist(),
            'choices': [[roles(names, int(parents[a]))[int(choice[j, a])] for a in range(len(indices))] for j in range(2)],
            'option_mass': [[responses[2*i+j][1] for i in indices] for j in range(2)]}
    checks = []
    offset = 2*len(query_keys)
    for i, (names, parent, alias, expected) in enumerate(sentinels):
        pair = responses[offset+2*i:offset+2*i+2]
        observed = [roles(names, parent)[int(torch.tensor(row[0]).argmax())] for row in pair]
        checks.append({'classes': names, 'parent': parent, 'alias': alias, 'expected': expected,
            'observed': observed, 'passed': observed == [expected, expected]})
    result = {'status': 'complete', 'implementation': IMPLEMENTATION, 'config': asdict(CONFIG),
        'source_manifest': model.manifest, 'text_only': True, 'target_images_loaded': False,
        'target_masks_loaded': False, 'weights_frozen': all(not p.requires_grad for p in model.model.parameters()),
        'query_keys': [{'classes': n, 'parent': p, 'alias': a} for n, p, a in query_keys],
        'raw_responses': responses, 'sources': sources, 'sentinels': checks,
        'sentinel_contract_passed': all(row['passed'] for row in checks),
        'prompt_example': role_prompt(*query_keys[0], tuple(range(len(roles(query_keys[0][0], query_keys[0][1]))))),
        'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'sentinel_contract_passed': result['sentinel_contract_passed'],
        'queries': len(query_keys), 'wall_seconds': result['wall_seconds'], 'sentinels': checks}), flush=True)


if __name__ == '__main__':
    main()
