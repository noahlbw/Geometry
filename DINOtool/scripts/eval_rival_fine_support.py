"""Frozen fine-support soft weights on exact20/30/40 and attachment stress."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_fine_support import IMPLEMENTATION, METHODS, PRIMARY, REPLAY, support_scores
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import summary
from eval_native_alias_noise import ORIGINAL
from eval_rival_fine_full import frozen_state, check_frozen, save
from eval_rival_projected_vocabulary_stress import (SOURCE as WORD_SOURCE, PROJECTED, VOCAB,
    SCENARIOS, variants_for, predict)
from audit_alias_action_capacity import dense
from run_region_semantic_suite_a800 import TOOL


SOURCE = TOOL/'results/rival_projected_vocabulary_stress_20261005'


@torch.inference_mode()
def main(args, context_factory=None):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing fine-support output.')
    prior = json.loads((SOURCE/args.dataset/'results.json').read_text())
    words = json.loads((WORD_SOURCE/args.dataset/'merged.json').read_text())
    pool = json.loads(VOCAB.read_text())
    if prior['status'] != 'complete' or not prior['coverage_verified']:
        raise RuntimeError('Completed frozen stress source required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    keys = prior['sample_keys']
    if len(keys) != 8 or len(set(keys)) != 8:
        raise RuntimeError('Frozen eight-window panel required.')
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if checkpoint_manifest(checkpoints) != prior['base_signature']['checkpoints']:
        raise RuntimeError('Original checkpoints changed.')
    variants, vocabularies = variants_for(args.dataset, pool, words, banks, vip, queries)
    if vocabularies != prior['context_vocabularies']:
        raise RuntimeError('Original word identities changed.')
    contexts = context_factory(variants) if context_factory is not None else None
    states = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output/'scores').mkdir()
    matrices = {s: {p: {m: [] for m in METHODS} for p in banks} for s in SCENARIOS}
    diagnostics, ignored = [], dict.fromkeys(banks, 0)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with np.load(SOURCE/args.dataset/'per_image_confusions.npz', allow_pickle=False) as historical:
        for index, key in enumerate(keys):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            with np.load(ORIGINAL/args.dataset/'numerical_cache'/f'{index}.npz', allow_pickle=False) as original, np.load(
                    PROJECTED/args.dataset/'scores'/f'{index}.npz', allow_pickle=False) as projected:
                replay, _, metadata, diag = predict(image, geometry, banks, vip, variants,
                    original, projected, capture_observations=True)
            values, packed, new_diagnostics = {}, {'sample_key': np.asarray(key)}, {}
            with np.load(SOURCE/args.dataset/'source_cache'/f'{index}.npz', allow_pickle=False) as cached:
                for s in SCENARIOS:
                    values[s], new_diagnostics[s] = {}, {}
                    for p in banks:
                        inputs = metadata[s][p]['reader_arguments']
                        reader = support_scores if contexts is None else contexts[s][p]
                        result, details = reader(*inputs)
                        for m in REPLAY:
                            if (not torch.equal(result[m], replay[s][p][m])
                                    or not np.array_equal(result[m].cpu().numpy(), cached[s+'__'+p+'__'+m+'__scores'])):
                                raise RuntimeError('Historical source score replay failed: '+s+'/'+p+'/'+m)
                        if index == 0 and not torch.equal(reader(*inputs, methods=(PRIMARY,))[0][PRIMARY], result[PRIMARY]):
                            raise RuntimeError('Standalone/all-arm candidate score differs.')
                        details['primary_singleton_checked'] = index == 0
                        values[s][p], new_diagnostics[s][p] = result, details
                        packed.update({s+'__'+p+'__'+m: v.cpu().numpy() for m, v in result.items()})
            np.savez_compressed(output/'scores'/f'{index}.npz', **packed)
            # No target is used by the weight source or by any identity control.
            for p, bank in banks.items():
                full = load_mask(sample, p, tuple(image.shape[-2:]))
                target = np.full((512, 512), -1, np.int64)
                ah, aw = min(512, image.shape[-2]), min(512, image.shape[-1])
                target[:ah, :aw] = full[:ah, :aw]
                valid = (target >= 0) & (target < bank.class_count)
                ignored[p] += int((~valid).sum())
                for s in SCENARIOS:
                    for m, value in values[s][p].items():
                        prediction = dense(value).argmax(-1).cpu().numpy()
                        encoded = target[valid]*bank.class_count+prediction[valid]
                        cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, bank.class_count)
                        if m in REPLAY and not np.array_equal(cm, historical[s+'__'+p+'__'+m][index]):
                            raise RuntimeError('Historical per-image confusion differs.')
                        matrices[s][p][m].append(cm)
            diagnostics.append({'sample_key': key, 'source': diag, 'support': new_diagnostics})
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'processed_images': index+1,
                'total_images': 8, 'sample_keys': keys[:index+1], 'methods': METHODS, 'scenarios': SCENARIOS,
                'base_signature': prior['base_signature'], 'vocabulary_source_sha256': prior['vocabulary_source_sha256'],
                'context_vocabularies': vocabularies, 'diagnostics': diagnostics,
                'historical_scores_bitwise_exact': True, 'historical_per_image_confusions_exact': True,
                'all_scores_persisted_before_masks': True, 'target_labels_used_by_weights': False,
                'metrics': {s: {p: {m: summary(np.stack(v).sum(0), banks[p].class_names, ignored[p])
                    for m, v in methods.items()} for p, methods in ps.items()} for s, ps in matrices.items()},
                'wall_seconds': time.perf_counter()-started, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
            save(output/'results.json', result)
            print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'controls_exact': True}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
        **{s+'__'+p+'__'+m: np.stack(v) for s, ps in matrices.items() for p, methods in ps.items() for m, v in methods.items()})
    result.update(status='complete', coverage_verified=True, **check_frozen(states, geometry, vip))
    save(output/'results.json', result)


if __name__ == '__main__':
    main(parse_args())
