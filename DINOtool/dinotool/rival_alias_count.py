"""Nested vocabulary-size controls for the unchanged RivalFineHard reader."""
import json

import torch

from .fine_alias_view import CONFIG, physical_margins
from .native_alias_noise import hard_pair_observation, signed_potential
from .native_query_alias import native_risk
from .prompts import clean_phrase
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'rival-fine-hard-nested-alias-count-study-v1-20261003'
COUNTS = (20, 30, 40)
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact',
    'ObserverAll', 'ObserverHard', 'CountMatchedRandom_Exact')


def canonical_indices(class_names, aliases, parents):
    indices = []
    for c, name in enumerate(class_names):
        matches = [i for i, alias in enumerate(aliases) if alias == name and int(parents[i]) == c]
        if len(matches) != 1:
            raise ValueError('Each class requires exactly one canonical alias in its own group.')
        indices.append(matches[0])
    return torch.tensor(indices, device=parents.device)


def parse_additions(response, excluded, maximum=20):
    """Only format/nonempty/duplicate checks; never a semantic-quality filter."""
    start = response.find('[')
    if start < 0:
        raise ValueError('No JSON array in language-model response.')
    values, _ = json.JSONDecoder().raw_decode(response[start:])
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError('Expected a JSON array of phrase strings.')
    seen = {clean_phrase(v).casefold() for v in excluded}
    accepted = []
    for value in values:
        if not value.strip():
            continue
        phrase = clean_phrase(value)
        if phrase.casefold() not in seen:
            seen.add(phrase.casefold())
            accepted.append(phrase)
        if len(accepted) == maximum:
            break
    return accepted


def nested_classes(base, additions, count):
    if count not in COUNTS or len(base) != len(additions):
        raise ValueError('Matching class entries and declared20/30/40 count required.')
    result = []
    for cls, extra in zip(base, additions):
        old = cls['synonyms']
        complete = old+extra
        if (len(old) != 20 or len(extra) != 20 or cls['name'] not in old
                or len({clean_phrase(v).casefold() for v in complete}) != 40):
            raise ValueError('Require40 distinct phrases with exact historical20 prefix.')
        result.append({'name': cls['name'], 'synonyms': complete[:count]})
    return result


def count_predictions(local, operator, broad, wide, wide_count, fine, fine_count,
                      coordinates, valid, members, canonical, parents, image_size):
    from .calibrated_competitive_alias import profiled_logits
    from .matched_contribution_alias import stencil_margins
    from .stratified_soft_alias import crop_stencil

    full = torch.zeros(len(valid), members.numel(), len(members), device=broad.device)
    for crop in wide:
        ids, coeff = crop_stencil(crop, wide_count, coordinates, image_size)
        full += stencil_margins(profiled_logits(crop, members), ids, coeff, CONFIG.beta)
    fine_margin = physical_margins(fine, fine_count, coordinates, members, valid, CONFIG)
    pair = native_risk(full, fine_margin, fine_margin, parents, canonical, valid, CONFIG)
    args = (wide, wide_count, coordinates, image_size, members)
    directed = hard_pair_observation(*args, pair, valid, CONFIG.beta, CONFIG.query_chunk)
    potential, _ = signed_potential(directed, valid)
    baseline = local.double()+operator.double() @ (broad.double()-local.double())
    values = {'Geometry': local.double(), 'NoAdmission_Exact': baseline,
        'RivalFineHard_Exact': baseline+operator.double() @ potential,
        'ObserverAll': broad.double(), 'ObserverHard': broad.double()+potential}
    permutation = alias_permutations(members, canonical, CONFIG.random_seed)[0]
    grouped = pair[:, members]
    shuffled = torch.empty_like(pair)
    shuffled[:, members] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
    random_directed = hard_pair_observation(*args, shuffled, valid, CONFIG.beta, CONFIG.query_chunk)
    random_potential, _ = signed_potential(random_directed, valid)
    values['CountMatchedRandom_Exact'] = baseline+operator.double() @ random_potential
    error = int(((shuffled[:, members] > 0).sum(2)-(grouped > 0).sum(2)).abs().max())
    if error or bool((pair[:, canonical] != 0).any()):
        raise RuntimeError('Canonical protection or count-matched deletion changed.')
    retained = (grouped[valid] == 0).sum(2)
    active_rivals = ~torch.eye(len(members), device=pair.device, dtype=torch.bool)
    chosen = retained[:, active_rivals]
    stats = {'random_count_max_error': error, 'canonical_risk_max': float(pair[:, canonical].abs().max()),
        'retained_count_min': int(chosen.min()), 'retained_count_max': int(chosen.max()),
        'retained_count_mean': float(chosen.double().mean()),
        'retained_count_histogram': torch.bincount(chosen.flatten(), minlength=members.shape[1]+1).cpu().tolist(),
        'per_class': []}
    for c in range(len(members)):
        other = torch.arange(len(members), device=pair.device) != c
        rejection = (grouped[valid, c, :, :][:, :, other] > 0).double().mean((0, 2))
        stats['per_class'].append({'alias_rejection_fractions': rejection.cpu().tolist(),
            'old20_retention_fraction': float((1-rejection[:20]).mean()),
            'added_retention_fraction': float((1-rejection[20:]).mean()) if members.shape[1] > 20 else None})
    return values, pair, stats
