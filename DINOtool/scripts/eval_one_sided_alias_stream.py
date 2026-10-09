"""Frozen rule equivalence and serial complete-image execution timing."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.one_sided_alias_audit import predict_image as previous_predict
from dinotool.one_sided_alias_stream import (IMPLEMENTATION, METHODS, PRIMARY,
    REFERENCE, OBSERVATION_MEAN, PROTOCOL, predict_image)
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def reader_profile(args, inputs):
    """Separate replay-only reader cost; not subtracted from whole-image timings."""
    import dinotool.one_sided_alias_stream as model
    from dinotool.one_sided_alias_audit import scores as previous_scores
    from benchmark_natural_sense_fast import measure
    original = model.scores
    timings = {}
    for name, scorer in ((REFERENCE, previous_scores), (PRIMARY, original)):
        scorer(*inputs, methods=(name,))
        _, timing = measure(scorer, inputs, torch.device(args.device), 5, dict(methods=(name,)))
        timings[name] = timing
    return dict(tile_reader=timings,
        scope='First tile, source observations already computed; separate diagnostic, not whole-image inference')


def predictor_for(execution):
    def call(*args, **kwargs):
        methods = kwargs.get('methods', METHODS)
        # Singleton reference latency must not include candidate computation.
        if methods == (REFERENCE,):
            return previous_predict(*args, **kwargs, execution=execution)
        result, diagnostics = predict_image(*args, **kwargs, execution=execution)
        if PRIMARY in methods and REFERENCE in methods:
            for p, values in result.items():
                different = int(np.count_nonzero(values[PRIMARY] != values[REFERENCE]))
                if different:
                    raise RuntimeError(f'Changed complete-image pixels: {p}: {different}')
                diagnostics[p]['prediction_mismatches'] = different
        return result, diagnostics
    return call


@torch.inference_mode()
def smoke(args):
    output, samples, load_image, _, geometry, banks, vip, queries, _, old = evaluator.inputs(args)
    state = frozen_state(geometry, vip)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    execution = FineCoverageExecution(cached=True, burst=True)
    predictor = predictor_for(execution)
    combined, diagnostics = predictor(image, geometry, banks, vip, queries)
    previous, _ = previous_predict(image, geometry, banks, vip, queries,
        methods=(*METHODS[:3], REFERENCE, OBSERVATION_MEAN), execution=execution)
    single, _ = predictor(image, geometry, banks, vip, queries, methods=(PRIMARY,))
    for p in banks:
        for m in (*METHODS[:3], REFERENCE, OBSERVATION_MEAN):
            if not np.array_equal(combined[p][m], previous[p][m]):
                raise RuntimeError('Original endpoint changed: '+p+'/'+m)
        if not np.array_equal(combined[p][PRIMARY], single[p][PRIMARY]):
            raise RuntimeError('Singleton execution changed predictions.')
    for row in diagnostics.values():
        if not (0 < row['fine_forwards'] <= 16 and row['geometry_encodings'] <= 4
                and row['wide_encodings'] <= 4 and row['canonical_risk_max'] == 0
                and row['score_max_abs_error'] <= 1e-10):
            raise RuntimeError('Source or rule contract changed.')
    output.mkdir(parents=True)
    save(output/'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        target_masks_loaded=False, sample_key=sample.key, original_endpoints_exact=True,
        singleton_primary_exact=True, complete_image_pixels_equal=True,
        diagnostics=diagnostics, execution=execution.report(),
        vocabulary=old['signature']['vocabulary'], **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke(args)
    else:
        execution = FineCoverageExecution(cached=True, burst=True)
        predictor = predictor_for(execution)
        if args.mode == 'full':
            evaluator.inputs = panel_inputs
            evaluator.evaluate(args, predictor=predictor, implementation=IMPLEMENTATION,
                methods=METHODS, view_protocol=PROTOCOL,
                competitive=dict(alias_admission=PROTOCOL['alias_admission'],
                    fixed_aliases_per_class=20, semantic_rule_changed=False,
                    target_labels_used_for_selection=False, fitted_parameters=0))
        else:
            import dinotool.one_sided_alias_stream as model
            records = []
            predictions, input_refs = {}, {}
            original = model.scores
            def capture(*pos, **kwargs):
                if not records and PRIMARY in kwargs.get('methods', ()):
                    records.append(pos)
                return original(*pos, **kwargs)
            def timed_predictor(image, *pos, **kwargs):
                result = predictor(image, *pos, **kwargs)
                name = kwargs['methods'][0]
                if name in (REFERENCE, PRIMARY):
                    input_refs[id(image)] = image
                    predictions[id(image), name] = result[0]
                return result
            with patch.object(model, 'scores', capture):
                evaluator.benchmark(args, predictor=timed_predictor, implementation=IMPLEMENTATION,
                    view_protocol=PROTOCOL, names=(METHODS[2], OBSERVATION_MEAN,
                                                  REFERENCE, PRIMARY, 'VIP_All20'))
            for key in input_refs:
                old, new = predictions[key, REFERENCE], predictions[key, PRIMARY]
                if any(not np.array_equal(old[p][REFERENCE], new[p][PRIMARY]) for p in old):
                    raise RuntimeError('Benchmark complete-image reference/candidate mismatch.')
            profile = reader_profile(args, records[0])
        path = Path(args.output_dir)/'results.json'
        result = json.loads(path.read_text())
        result['execution'] = execution.report()
        if args.mode == 'benchmark':
            result['reader_profile'] = profile
            result['benchmark_predictions_equal'] = True
        save(path, result)
