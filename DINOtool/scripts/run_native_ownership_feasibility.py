"""Mask-free source feasibility on unchanged cached native observations."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.matched_contribution_alias import stencil_margins
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.native_class_ownership import (IMPLEMENTATION, CONFIG, FEASIBILITY_GATE,
    family_description, family_class_field, ownership_risk, risk_union)
from dinotool.stratified_soft_alias import WideCrop, crop_stencil
from dinotool.tcpr import TCPRTextBank
from eval_pair_context_reader import ORIGINAL
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import TOOL, BASE, THIRD, idle

sys.path.insert(0, str(THIRD))


DATASETS = ('vdd', 'potsdam', 'udd5', 'oem', 'loveda', 'vaihingen', 'landcoverai', 'flair1')
SOURCE = TOOL/'results/native_alias_noise_screen_r2_20261003'


def native_crops(cache, key, positions, device):
    crops, count = [], torch.zeros(512, 512, device=device)
    for i, (t, l, h, w) in enumerate(positions):
        prefix = key+f'__native_crop{i}__'
        crops.append(WideCrop(torch.from_numpy(cache[prefix+'mean_template_logits']).to(device),
                              torch.from_numpy(cache[prefix+'salience']).to(device), t, l, h, w))
        count[t:t+h, l:l+w] += 1
    return crops, count.clamp_min(1)


def accumulate(records, prior, fields, cached, valid):
    wrong = prior['vocabularies']['wrong_parent']
    for p, description in wrong['replacements'].items():
        names = [c['name'] for c in wrong['vocabularies'][p]]
        groups = records.setdefault(p, {})
        for change in description['replacements']:
            c, d = names.index(change['class']), names.index(change['declared_rival'])
            ids = [20*c+slot for slot in change['slots']]
            row = groups.setdefault(change['class'], {'declared_rival': names[d],
                'foreign_active': 0, 'foreign_risk_sum': 0., 'foreign_blind': 0, 'blind_captured': 0,
                'paraphrase_active': 0, 'paraphrase_risk_sum': 0.})
            for scene, prefix in (('wrong_parent', 'foreign'), ('paraphrase', 'paraphrase')):
                key = scene+'__'+p
                b = cached[key+'__broad_margin'][valid][:, ids, d]
                risk = fields[key+'__attachment_risk'][valid][:, ids, d]
                active = b > 0
                row[prefix+'_active'] += int(active.sum())
                row[prefix+'_risk_sum'] += float(risk[active].astype(np.float64).sum())
                if scene == 'wrong_parent':
                    fine = cached[key+'__native_margin'][valid][:, ids, d]
                    blind = active & (fine >= 0)
                    if np.any(cached[key+'__risk'][valid][:, ids, d][blind] != 0):
                        raise RuntimeError('Prior view-reversal blind identity differs.')
                    row['foreign_blind'] += int(blind.sum())
                    row['blind_captured'] += int((blind & (risk > 0)).sum())


@torch.inference_mode()
def run(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists() or not idle(0):
        raise ValueError('New isolated root and idle physical GPU0 required.')
    prior = {d: json.loads((SOURCE/d/'merged.json').read_text()) for d in DATASETS}
    if any(not r['coverage_verified'] or r['processed_images'] != 8 or len(set(r['sample_keys'])) != 8 for r in prior.values()):
        raise ValueError('Verified64 fixed prior samples required.')
    root.mkdir()
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'config': asdict(CONFIG),
        'feasibility_gate': FEASIBILITY_GATE, 'source': str(SOURCE), 'images': 64,
        'target_masks_loaded': False, 'new_image_forwards': 0, 'text_encoding_physical_gpu': 0,
        'construction_metadata_audit_only': True, 'no_parameter_fitting': True})
    started = time.perf_counter()
    upstream = BASE/'third_party/VIP_official_5bd25ee'
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise RuntimeError('Pinned VIP source changed.')
    checkpoints = make_checkpoints(SimpleNamespace(checkpoint_dir=str(BASE/'ckpt/DINO'), dinov3_repo=str(TOOL/'dinov3_hub')))
    manifest = checkpoint_manifest(checkpoints)
    if any(r['signature']['checkpoints'] != manifest for r in prior.values()):
        raise RuntimeError('Changed frozen checkpoints.')
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device='cuda'), upstream)
    device = vip.device
    torch.cuda.reset_peak_memory_stats()
    text_cache, results = {}, {}
    for dataset in DATASETS:
        directory = root/dataset
        (directory/'fields').mkdir(parents=True)
        metadata, descriptions = {}, {}
        for scene, entry in prior[dataset]['vocabularies'].items():
            for p, classes in entry['vocabularies'].items():
                names = tuple(c['name'] for c in classes)
                words = tuple(tuple(c['synonyms']) for c in classes)
                if any(len(a) != 20 for a in words):
                    raise ValueError('Changed20-count contract.')
                aliases = tuple(a for group in words for a in group)
                missing = list(dict.fromkeys(a for a in aliases if a not in text_cache))
                if missing:
                    q = vip.encode_queries(('encoding_cache',), (tuple(missing),))
                    mean = q.features.float().mean(1).cpu()
                    text_cache.update({a: mean[i] for i, a in enumerate(missing)})
                features = F.normalize(torch.stack([text_cache[a] for a in aliases]).to(device), dim=-1)
                parents = torch.arange(len(names), device=device).repeat_interleave(20)
                canonical_mask = torch.tensor([a == names[c] for c, group in enumerate(words) for a in group], dtype=torch.bool, device=device)
                bank = TCPRTextBank(features, parents, canonical_mask, names, aliases)
                families, excluded, assignments, members, canonical, conflict = family_description(bank)
                key = scene+'__'+p
                metadata[key] = (bank, excluded, assignments, members, canonical, conflict)
                descriptions[key] = {'families': [[aliases[a] for a in family] for family in families],
                    'family_indices': families, 'aliases': aliases, 'parents': parents.tolist(),
                    'canonical': canonical.tolist(), 'known_pair_fraction': float(((~excluded[:, members]).sum(-1) > 0).float().mean())}
                np.savez_compressed(directory/(key+'__text.npz'), mean_template_features=features.cpu().numpy())
        save(directory/'families.json', descriptions)
        records, invariants = {}, []
        for i, sample in enumerate(prior[dataset]['sample_keys']):
            with np.load(ORIGINAL/dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as old:
                coordinates = torch.from_numpy(old['coordinates']).to(device)
                valid = torch.from_numpy(old['valid']).to(device)
            fields, replay_errors = {}, {}
            with np.load(SOURCE/dataset/'numerical_cache'/f'{i}.npz', allow_pickle=False) as cached:
                positions = prior[dataset]['diagnostics'][sample]['native_crop_positions']
                for key, (bank, excluded, assignments, members, canonical, conflict) in metadata.items():
                    crops, count = native_crops(cached, key, positions, device)
                    old_margin = torch.zeros(len(valid), len(bank.alias_names), bank.class_count, device=device)
                    for crop in crops:
                        ids, coeff = crop_stencil(crop, count, coordinates, (512, 512))
                        old_margin += stencil_margins(profiled_logits(crop, members), ids, coeff, CONFIG.beta)
                    replay_errors[key] = float((old_margin-torch.from_numpy(cached[key+'__native_margin']).to(device)).abs().max())
                    if replay_errors[key] > 1e-4:
                        raise RuntimeError('Original native physical query replay differs.')
                    field, known = family_class_field(crops, count, coordinates, members, excluded, valid)
                    attachment = ownership_risk(field, known, assignments, bank.parent_indices, canonical, conflict, valid)
                    previous_risk = torch.from_numpy(cached[key+'__risk']).to(device)
                    union = risk_union(previous_risk, attachment)
                    if (float(attachment[:, canonical].abs().max()) != 0 or not bool(torch.isfinite(union).all())
                            or bool(((union < 0) | (union > 1)).any())):
                        raise RuntimeError('Canonical/bounded source invariants differ.')
                    if not torch.equal(union[attachment == 0], previous_risk[attachment == 0]):
                        raise RuntimeError('Unknown source does not exactly preserve preceding risk.')
                    if i == 0:
                        changed = [replace(c, alias_logits=c.alias_logits.clone(), salience=c.salience.clone()) for c in crops]
                        held = excluded[0].nonzero().flatten()
                        for crop in changed:
                            crop.alias_logits[:, held] = 10000
                            crop.salience[held] = 10000
                        other = family_class_field(changed, count, coordinates, members, excluded[:1], valid)[0]
                        if not torch.equal(other[:, 0], field[:, 0]):
                            raise RuntimeError('Held-out observations affect their own class reference.')
                    fields[key+'__reference'] = field.cpu().numpy()
                    fields[key+'__known'] = known.cpu().numpy()
                    fields[key+'__attachment_risk'] = attachment.float().cpu().numpy()
                    fields[key+'__joint_risk'] = union.cpu().numpy()
                np.savez_compressed(directory/'fields'/f'{i}.npz', **fields)
                # Construction metadata enters only this audit, after every source field is saved.
                accumulate(records, prior[dataset], fields, cached, valid.cpu().numpy())
            invariants.append(replay_errors)
            save(root/'suite_status.json', {'status': 'running', 'dataset': dataset, 'processed': i+1, 'total': 8,
                'completed': list(results), 'target_masks_loaded': False, 'elapsed_seconds': time.perf_counter()-started})
        row = {'status': 'complete', 'implementation': IMPLEMENTATION, 'config': asdict(CONFIG),
            'sample_keys': prior[dataset]['sample_keys'], 'signature': prior[dataset]['signature'],
            'processed_images': 8, 'total_images': 8, 'coverage_verified': True,
            'construction_audit': records, 'native_replay_max_errors': invariants,
            'source_fields_precede_construction_audit': True, 'heldout_independence_verified': True,
            'target_masks_loaded': False, 'new_image_forwards': 0, 'families': descriptions}
        save(directory/'results.json', row)
        results[dataset] = row
    separating, capturing, aggregate = 0, 0, {}
    for dataset, row in results.items():
        group = row['construction_audit']['D' if dataset == 'loveda' else dataset]
        totals = {k: sum(c[k] for c in group.values()) for k in next(iter(group.values())) if k != 'declared_rival'}
        foreign = totals['foreign_risk_sum']/max(1, totals['foreign_active'])
        legitimate = totals['paraphrase_risk_sum']/max(1, totals['paraphrase_active'])
        capture = totals['blind_captured']/max(1, totals['foreign_blind'])
        separating += foreign > legitimate
        capturing += capture >= FEASIBILITY_GATE['minimum_blind_spot_capture_fraction']
        aggregate[dataset] = {**totals, 'foreign_mean_risk': foreign, 'paraphrase_mean_risk': legitimate,
                              'blind_spot_capture_fraction': capture}
    decision = {'passed': separating >= FEASIBILITY_GATE['minimum_domains_separating_foreign_from_paraphrase']
                and capturing >= FEASIBILITY_GATE['minimum_domains_capturing_blind_spot'],
        'separating_domains': separating, 'capturing_domains': capturing,
        'source_only_not_model_promotion': True, 'no_automatic_full_rollout': True}
    if not all(not p.requires_grad for p in vip.backbone.model.parameters()):
        raise RuntimeError('Weights are not frozen.')
    final = {'status': 'complete', 'implementation': IMPLEMENTATION, 'config': asdict(CONFIG),
        'decision': decision, 'aggregate': aggregate, 'completed': list(results),
        'target_masks_loaded': False, 'new_image_forwards': 0, 'weights_frozen': True,
        'unique_aliases_text_encoded': len(text_cache), 'wall_seconds': time.perf_counter()-started,
        'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1048576}
    save(root/'suite_results.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
