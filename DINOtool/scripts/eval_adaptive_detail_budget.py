"""Fixed developed panels and full singleton cost with at most one detail crop."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch
import torch.nn.functional as F

from dinotool.adaptive_detail_budget import IMPLEMENTATION, METHODS, PRIMARY, detail_rgb, replace_witness, select_detail
from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.bounded_alias_reuse import bounded_scores
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.sparse_alias_reuse import cache_sparse, profile_aliases, sparse_scores
from dinotool.stratified_soft_alias import WideCrop
from dinotool.prompts import load_class_specs
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import sample_broad
from eval_matched_contribution_alias import crop_from_features
from eval_rival_fine_graph import bind, graph_features
import eval_sparse_alias_reuse as reference


@torch.inference_mode()
def tile(image, top, left, geometry, banks, vip, queries, wide, crops, wide_count, *,
         methods=METHODS, layouts=None, readers=None):
    height, width = image.shape[-2:]
    rgb = _crop_at(image, top, left, 512).to(geometry.device)
    prepared = geometry.prepare_image(rgb)
    coordinates = reference.reference.tile_coordinates(top, left, geometry.device)
    relative = coordinates - coordinates.new_tensor((top, left))
    valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
    operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
    layouts = reference.layouts_for(queries) if layouts is None else layouts
    fields = dict(operator=operator, coordinates=coordinates, valid=valid)
    contexts = {}
    for p, bank in banks.items():
        raw = alias_class_scores((prepared.geometry_projected.float()
            @ F.normalize(bank.features.float(), dim=-1).T)[0], bank.parent_indices, bank.class_count)
        local = raw / .07
        broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
        contexts[p] = raw, local, broad
        fields[p + '__raw'], fields[p + '__local'] = raw, local
    chosen, acquisition = None, {}
    features = None
    if PRIMARY in methods:
        chosen, acquisition = select_detail([c[1] for c in contexts.values()], [c[2] for c in contexts.values()],
            operator, relative, valid, min(512, height - top), min(512, width - left))
        if chosen is not None:
            features = graph_features(vip, detail_rgb(rgb, chosen))
    output, diagnostics = {}, {}
    for p, bank in banks.items():
        raw, local, broad = contexts[p]
        base = local.double() + operator.double() @ (broad.double() - local.double())
        values = {'Geometry': local.double(), 'NoAdmission_Exact': base}
        query, layout = queries[p], layouts[p]
        stats = dict(operator_residual=residual, fine_forwards=float(chosen is not None), **acquisition)
        if PRIMARY in methods or 'SparseNativeSoft' in methods:
            observed = cache_sparse(crops['clean', p], wide_count, coordinates, (height, width), layout)
            blank = WideCrop(local.new_empty(1024, 1), local.new_empty(1), 0, 0, 512, 512, 32, 512)
            native_crop = crop_from_features(prepared.native_projected, query, blank)
            native = profile_aliases(native_crop.alias_logits, native_crop.salience, layout)
            native_score, native_stats, native_fields = sparse_scores(local, operator, broad, observed, native, valid, layout)
            values['SparseNativeSoft'] = native_score
            stats.update({'SparseNativeSoft__' + k: v for k, v in native_stats.items()})
            if PRIMARY in methods:
                covered = torch.zeros_like(valid)
                if features is None:
                    values[PRIMARY] = native_score
                    detail_stats = dict(native_stats, max_contenders_per_query=2.)
                else:
                    dt, dl, dh, dw = chosen
                    blank = WideCrop(features.new_empty(1024, 1), features.new_empty(1), dt, dl, dh, dw, 32, 256)
                    detail_crop = replace(crop_from_features(features, query, blank), grid_side=32, crop_side=256)
                    detail = cache_sparse([detail_crop], torch.ones(512, 512, device=geometry.device), relative,
                                          (512, 512), layout)[0]
                    witness, covered = replace_witness(native, detail, layout, valid)
                    _, detail_stats, detail_fields = bounded_scores(local, operator, broad, observed, witness,
                        covered, layout, witness_contenders=True)
                    potential = torch.where(covered[:, None], detail_fields['potential'], native_fields['potential'])
                    values[PRIMARY] = base + operator.double() @ potential
                detail_stats.update(fine_forwards=float(chosen is not None),
                    additional_visual_forwards=float(chosen is not None),
                    detail_replaced_donor_fraction=float(covered[valid].double().mean()) if bool(valid.any()) else 0.)
                stats.update({PRIMARY + '__' + k: v for k, v in detail_stats.items()})
        output[p] = {m: values[m] for m in methods}
        diagnostics[p] = stats
    return output, diagnostics, fields


def window(*args, methods=METHODS):
    return bind(reference.window, tile=tile)(*args, methods=methods)


def predict_image(*args, methods=(PRIMARY,), layouts=None):
    return bind(reference.predict_image, tile=tile)(*args, methods=methods, layouts=layouts, readers=None)


def save(path, result):
    if 'candidate_fine_observer_forbidden_at_runtime' in result:
        result.pop('candidate_fine_observer_forbidden_at_runtime')
        result['unbounded_four_fine_observer_forbidden_at_runtime'] = True
        result['detail_forward_budget_per_window'] = 1
        result['target_labels_used_by_acquisition'] = False
    if 'benchmark' in result:
        result['benchmark'].pop('candidate_fine_observer_forbidden_at_runtime', None)
        result['benchmark']['unbounded_four_fine_observer_forbidden_at_runtime'] = True
    reference.reference.save(path, result)


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing mask-free detail smoke outputs.')
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    prior = json.loads((reference.SOURCE / args.dataset / 'results.json').read_text())
    sample = {s.key: s for s in samples}[prior['sample_keys'][0]]
    models = make_models(args, specs)
    geometry, banks, vip, queries = models[:4]
    state = reference.reference.frozen_state(geometry, vip)
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    layouts = reference.layouts_for(queries)
    _, scores, diagnostics = window(image, *models[:4], layouts)
    previous = reference.TOOL / 'results/sparse_alias_reuse_20261005' / args.dataset / 'scores/0.npz'
    with np.load(previous, allow_pickle=False) as old:
        for p in banks:
            for m in METHODS[:-1]:
                if not np.array_equal(scores[p][m].cpu().numpy(), old[p + '__' + m]):
                    raise RuntimeError('Mask-free smoke changed original score: ' + p + '/' + m)
    output.mkdir(parents=True)
    combined = predict_image(image, *models[:4], output, methods=('NoAdmission_Exact', PRIMARY), layouts=layouts)
    single = predict_image(image, *models[:4], output, methods=(PRIMARY,), layouts=layouts)
    if any(not np.array_equal(combined[0][p][PRIMARY], single[0][p][PRIMARY]) for p in banks):
        raise RuntimeError('Budgeted detail singleton versus multi-arm complete prediction differs.')
    for fields in diagnostics.values():
        if not 0 <= fields[PRIMARY + '__fine_forwards'] <= 1 or fields[PRIMARY + '__max_contenders_per_query'] > 4:
            raise RuntimeError('Mask-free smoke exceeded detail/competition budget.')
    save(output / 'results.json', dict(status='complete', implementation=IMPLEMENTATION,
        sample_key=sample.key, image_size=list(image.shape[-2:]), target_masks_loaded=False,
        original_three_scores_exact=True, singleton_complete_predictions_exact=True,
        diagnostics=diagnostics, complete_diagnostics=single[1],
        **reference.reference.check_frozen(state, geometry, vip)))
    print(json.dumps({'status': 'complete', 'mask_free_smoke': True, 'dataset': args.dataset}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    benchmark = bind(reference.complete_image_benchmark, PRIMARY=PRIMARY, predict_image=predict_image)
    with torch.inference_mode():
        args = parser.parse_args()
        if args.smoke:
            smoke(args)
        else:
            scope = SimpleNamespace(**dict(vars(reference.reference), save=save))
            bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
                 window=window, complete_image_benchmark=benchmark, reference=scope)(args)
