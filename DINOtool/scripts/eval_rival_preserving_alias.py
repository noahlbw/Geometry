"""One changed writer: rival-preserving observation with frozen source/readout."""
import sys

import numpy as np
import torch

from dinotool.alias_spatial_allocation import ROUNDING_GUARD
from dinotool.contrastive_reversal_alias import CONFIG as SOURCE_CONFIG, reversal_risk
from dinotool.pair_context_reader import pair_observation
from dinotool.rival_preserving_alias import (IMPLEMENTATION, PRIMARY, METHODS, ALIAS_SHUFFLES,
    CONFIG, competitive_potential, directional_controls)
from dinotool.target_context_alias import alias_permutations
from eval_contrastive_reversal_alias import predict as previous_predict
from eval_geometry_semantic_innovation import parse_args, SETTINGS
from eval_pair_context_reader import main


@torch.inference_mode()
def extend(*, bank, crops, count, coordinates, image_size, members, canonical, valid, cap,
           operator, local, broad, baseline, risk, full, kept, removed, text, source, protocol, supported):
    risks = {PRIMARY: risk,
        'RivalShuffledSupport_Exact': reversal_risk(full, *[torch.from_numpy(source[protocol+'__source_shuffle_'+name]).to(risk.device)
            for name in ('kept', 'removed')], bank.parent_indices, canonical, supported, SOURCE_CONFIG)['reversal'],
        'RivalTextOnly_Exact': text[None].expand_as(risk).masked_fill(~valid[:, None, None], 0).clone()}
    risks['RivalTextOnly_Exact'].scatter_(-1, bank.parent_indices[None, :, None].expand(len(valid), -1, 1), 0)
    for name, permutation in zip(ALIAS_SHUFFLES, alias_permutations(members, canonical, SOURCE_CONFIG.random_seed)):
        changed = torch.empty_like(risk)
        grouped = risk[:, members]
        changed[:, members] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
        risks[name] = changed
    endpoints, fields, diagnostics = {}, {}, {'sources': {}, 'controls': {}}
    primary = None
    for name, gamma in risks.items():
        _, _, directed = pair_observation(crops, count, coordinates, image_size, members, gamma,
            valid, SETTINGS.tau, CONFIG.query_chunk)
        excess = float((-directed-cap[..., None]).clamp_min(0).max())
        if excess > ROUNDING_GUARD or float(gamma[:, canonical].abs().max()) != 0:
            raise RuntimeError('Per-rival original writer protection failed.')
        potential, margins, stats = competitive_potential(directed, valid, CONFIG)
        lower = -(bank.class_count-1)*cap.double()/bank.class_count
        upper = (cap.double().sum(-1, keepdim=True)-cap.double())/bank.class_count
        bound_error = float(torch.maximum((lower-potential).clamp_min(0), (potential-upper).clamp_min(0)).max())
        if bound_error > ROUNDING_GUARD:
            raise RuntimeError('Directed-capacity-induced potential bounds failed.')
        endpoints[name] = baseline+operator @ potential
        fields[name+'__directed'] = directed.cpu().numpy()
        fields[name+'__potential'] = potential.cpu().numpy()
        fields[name+'__scores'] = endpoints[name].cpu().numpy()
        diagnostics['sources'][name] = {**stats, 'directed_capacity_excess_max': excess,
            'potential_bound_excess_max': bound_error, 'canonical_risk_max': float(gamma[:, canonical].abs().max()),
            'mean_rival_risk': float(gamma[valid].mean()),
            'risk_spectrum_max_error': float((gamma[:, members].sort(2).values-risk[:, members].sort(2).values).abs().max())
                if name in ALIAS_SHUFFLES else None}
        if name == PRIMARY:
            primary = directed
            fields['pair_risk'] = gamma.cpu().numpy()
            fields['requested_pair_margins'] = margins.cpu().numpy()
            endpoints['RivalMeanLogit'] = .5*(local.double()+broad.double()+potential)
            fields['RivalMeanLogit__scores'] = endpoints['RivalMeanLogit'].cpu().numpy()
    for name, directed in directional_controls(primary.cpu().numpy(), cap.cpu().numpy(), valid.cpu().numpy()).items():
        action = torch.from_numpy(directed).to(operator.device)
        potential, _, stats = competitive_potential(action, valid, CONFIG)
        endpoints[name] = baseline+operator @ potential
        fields[name+'__directed'] = directed
        fields[name+'__potential'] = potential.cpu().numpy()
        fields[name+'__scores'] = endpoints[name].cpu().numpy()
        diagnostics['controls'][name] = {**stats,
            'pair_budget_max_error': float((action.sum(0)-primary.double().sum(0)).abs().max()),
            'potential_budget_max_error': float((potential.sum(0)-torch.from_numpy(fields[PRIMARY+'__potential']).to(potential.device).sum(0)).abs().max()),
            'pair_spectrum_max_error': float((action[valid].sort(0).values-primary.double()[valid].sort(0).values).abs().max()),
            'capacity_excess_max': float((-action-cap.double()[..., None]).clamp_min(0).max())}
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise RuntimeError('Nonfinite competitive reader fields.')
    return endpoints, fields, diagnostics


def predict(*args):
    return previous_predict(*args, extension=extend)


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
    main(parse_args(), smoke, predictor=predict, methods=METHODS, implementation=IMPLEMENTATION, config=CONFIG)
