"""Mask-free exact replay of a vocabulary-count prior on the frozen reader."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.natural_count_calibration import IMPLEMENTATION, choose_count_strength
from dinotool.natural_evaluation import discover_samples, load_rgb
from dinotool.natural_text_adaptation import class_logits, trusted_witnesses
from dinotool.natural_variable_alias_reader import retained_variable_scores
from eval_excess_alias_rejection import prepare_wide
from eval_geometry_vip_reliability import sample_broad
from eval_natural_text_adaptation import models, bank_from_query, subset, sampled_aliases
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save, tile_reference_coordinates
from eval_stratified_soft_alias import tile_coordinates


@torch.inference_mode()
def calibrate(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Existing count-calibration output; refusing overwrite.')
    source_path = Path(args.source_selection)
    source = json.loads(source_path.read_text())
    if (source['status'] != 'complete' or source['dataset'] != args.dataset
            or source['target_masks_loaded'] or source['target_label_tuning']):
        raise ValueError('Complete mask-free source calibration required.')
    if args.max_images < 0:
        raise ValueError('Nonnegative smoke image limit required.')
    samples = {s.key: s for s in discover_samples(args.dataset, args.data_root)}
    keys = source['image_keys'][:args.max_images or None]
    if not keys or len(keys) != len(set(keys)) or any(k not in samples for k in keys):
        raise ValueError('Changed or empty calibration image coverage.')
    started = time.perf_counter()
    geometry, vip, encoded, _, _, identity, _ = models(args)
    if json.loads(json.dumps(identity)) != source['identity']:
        raise ValueError('Changed source vocabulary/checkpoints/text cache.')
    states = frozen_state(geometry, vip)
    choice = source['chosen']
    query = subset(encoded[choice['template']], torch.tensor(source['selected_indices'], device=geometry.device))
    bank, canonical = bank_from_query(query)
    anchor_query = encoded['seg_template']
    anchor_bank, anchor_canonical = bank_from_query(anchor_query)
    queries = dict(adapted=query, anchor=anchor_query)
    banks = dict(adapted=bank, anchor=anchor_bank)
    counts = torch.bincount(query.parents, minlength=bank.class_count).cpu()
    if counts.tolist() != source['selected_counts']:
        raise ValueError('Changed per-class alias counts.')
    records = []
    output.mkdir(parents=True)
    for number, key in enumerate(keys, 1):
        image = load_rgb(samples[key])
        h, w = image.shape[-2:]
        top, left = max(0, (h-512)//2), max(0, (w-512)//2)
        prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
        coords = tile_coordinates(top, left, geometry.device)
        valid = (coords[:, 0] < h) & (coords[:, 1] < w)
        fine_coords = tile_reference_coordinates(coords, top, left)
        maps, crops, _, count = prepare_wide(image, vip, {'text': (banks, queries)})
        adapted_map = torch.zeros_like(maps['text', 'adapted'])
        for crop in crops['text', 'adapted']:
            logits = class_logits(crop.alias_logits, crop.salience, query.parents,
                                  bank.class_count, choice['tau'], choice['tem'])
            dense = F.interpolate(logits.T.reshape(1, -1, 21, 21), (336, 336),
                                  mode='bilinear', align_corners=False)[0]
            adapted_map[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width] += dense[:, :crop.actual_height, :crop.actual_width]
        adapted_map /= count
        anchor_geo = ((prepared.geometry_projected.float() @ anchor_bank.features.T)[0]*40)[valid]
        anchor_wide = sampled_aliases(crops['text', 'anchor'], count, coords[valid], (h, w))
        labels, quality, _ = trusted_witnesses(anchor_geo, anchor_wide, anchor_canonical,
            anchor_query.parents, bank.class_count, bank.class_names[0] == 'background')
        fine, fine_count, _ = observe_fine(image[:, top:top+512, left:left+512], vip,
                                          {'adapted': query}, fine_coords, valid)
        operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
        raw = alias_class_scores((prepared.geometry_projected.float() @ bank.features.T)[0],
                                 bank.parent_indices, bank.class_count)/.07
        broad = sample_broad(adapted_map, top, left, h, w).reshape_as(raw)
        wide_crops = [replace(c, salience=c.salience/choice['tem']) for c in crops['text', 'adapted']]
        fine_crops = [replace(c, salience=c.salience/choice['tem']) for c in fine['adapted']]
        scores, _ = retained_variable_scores(raw, operator, broad, wide_crops, count,
            fine_crops, fine_count, coords, fine_coords, valid, query.parents, canonical,
            (h, w), beta=choice['tau'])
        records.append(dict(scores=scores['RivalFineHard_Exact'][valid].cpu(),
                            operator=operator[valid][:, valid].cpu(),
                            labels=labels.cpu(), quality=quality.cpu()))
        save(output/'progress.json', dict(status='running', processed_images=number,
                                         total_images=len(keys), target_masks_loaded=False))
        print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(keys),
                              trusted_tokens=int((quality > 0).sum()))), flush=True)
    decision = choose_count_strength(records, counts, choice['tau'])
    # Only the operator row mass is needed for subsequent scalar-prior replay.
    cache = [dict(scores=r['scores'], operator_row_mass=r['operator'].sum(-1),
                  labels=r['labels'], quality=r['quality']) for r in records]
    torch.save(dict(image_keys=keys, records=cache, counts=counts, tau=choice['tau']), output/'replay.pt')
    result = dict(status='complete', implementation=IMPLEMENTATION, dataset=args.dataset,
        processed_images=len(keys), total_images=len(keys), image_keys=keys,
        source_selection_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_implementation=source['implementation'], source_core=source['core_implementation'],
        source_identity=identity, selected_counts=counts.tolist(), source_choice=choice,
        count_calibration=decision, smoke_only=bool(args.max_images),
        exactness='Exact scalar broad-count-prior replay on retained central tiles, not an alias-removal counterfactual.',
        target_masks_loaded=False, target_label_tuning=False, transductive=True,
        full_image_threshold_calibrated=False, wall_seconds=time.perf_counter()-started,
        peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
        **check_frozen(states, geometry, vip))
    save(output/'selection.json', result)
    print(json.dumps(dict(dataset=args.dataset, count_calibration=decision)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'output-dir', 'source-selection'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    calibrate(parser.parse_args())
