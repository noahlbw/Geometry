"""Cross-position alias information tests, not a selector or a weight rule."""
from dataclasses import dataclass, asdict
import numpy as np


IMPLEMENTATION = 'geometry-cross-support-alias-increment-diagnostic-v1-20261008'


@dataclass(frozen=True)
class DiagnosticConfig:
    ridge: float = .01
    near_duplicate_cosine: float = .95
    minimum_positions: int = 32
    minimum_effective_support: float = 32.
    epsilon: float = 1e-8
    seed: int = 20261008
    shuffles: int = 3


CONFIG = DiagnosticConfig()
PROTOCOL = dict(implementation=IMPLEMENTATION, config=asdict(CONFIG),
    panel='40 complete UDD5 images; eight evenly spaced complete images in each other RS domain',
    folds='32x32 grid: columns0..13 vs18..31; columns14..17 excluded; bidirectional fit/test',
    support='mean original nonnegative Geometry W rows of frozen coupled top2 competing queries; diagonal removed',
    target='native-head canonical c-minus-rival contrast, not an input to the class-only regressor',
    baseline='intercept + local owner/rival and wide owner/rival class log-mean-exp',
    alias='one wide profiled/interpolated alias logit; not a new visual observation',
    exclusion='candidate and owner aliases at text cosine>=.95 removed from BOTH baseline pools; canonical-near candidates ineligible',
    null='three within-fold spatial permutations of alias responses; independently break fit/test correspondence',
    audit='masks only after source statistics persisted; held-out two-class Brier, exact old full-image single-deletion joins',
    calibration='train-only class ridge .01 frozen; weighted OLS alias residualization; scalar residual ridge .01',
    limitations='spatial folds share backbone attention, Geometry and wide crops; native canonical is a pseudo-reference, not truth')


def effective_count(weights):
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum()**2/(w@w)) if w@w > 0 else 0.


def fit_predict(x_train, y_train, weights, x_test, config=CONFIG):
    """Fit normalization only on training positions; intercept is unpenalized."""
    x, y, w, test = [np.asarray(v, dtype=np.float64) for v in (x_train, y_train, weights, x_test)]
    if x.ndim != 2 or test.ndim != 2 or test.shape[1] != x.shape[1] or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError('Matching training and held-out observations required.')
    if not all(np.isfinite(v).all() for v in (x, y, w, test)) or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Finite observations and nonnegative nonempty weights required.')
    w = w/w.sum()
    mean = w@x
    scale = np.sqrt(w@((x-mean)**2))
    scale = np.where(scale > config.epsilon, scale, 1.)
    train = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    heldout = np.column_stack((np.ones(len(test)), (test-mean)/scale))
    penalty = np.diag([0.]+[config.ridge]*x.shape[1])
    coefficients = np.linalg.solve(train.T@(w[:, None]*train)+penalty, train.T@(w*y))
    return heldout@coefficients, coefficients


def fold_indices(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (1024,):
        raise ValueError('Frozen32x32 tile required.')
    columns = np.arange(1024)%32
    return np.flatnonzero(valid & (columns < 14)), np.flatnonzero(valid & (columns >= 18))


def weighted_mse(target, prediction, weights):
    return float(np.average((target-prediction)**2, weights=weights))


def increment_predict(base_train, target, weights, base_test, alias_train, alias_test, config=CONFIG):
    before, _ = fit_predict(base_train, target, weights, base_test, config)
    fitted, _ = fit_predict(base_train, target, weights, base_train, config)
    w = weights/weights.sum()
    mean = w@base_train
    scale = np.sqrt(w@((base_train-mean)**2))
    scale = np.where(scale > config.epsilon, scale, 1.)
    x = np.column_stack((np.ones(len(base_train)), (base_train-mean)/scale))
    t = np.column_stack((np.ones(len(base_test)), (base_test-mean)/scale))
    projection = np.linalg.lstsq(x*np.sqrt(w[:, None]), alias_train*np.sqrt(w), rcond=config.epsilon)[0]
    residual = alias_train-x@projection
    residual_scale = np.sqrt(w@(residual**2))
    if residual_scale <= config.epsilon:
        return before, before.copy(), 0.
    z = residual/residual_scale
    coefficient = float((w*z)@(target-fitted)/(w@(z*z)+config.ridge))
    return before, before+coefficient*(alias_test-t@projection)/residual_scale, coefficient


def cross_fit(base, alias, target, support, valid, seed, config=CONFIG):
    folds, rows = fold_indices(valid), []
    for fold in range(2):
        train, test = folds[fold], folds[1-fold]
        wt, we = support[train], support[test]
        neff = [effective_count(w) for w in (wt, we)]
        if min(len(train), len(test)) < config.minimum_positions or min(neff) < config.minimum_effective_support:
            rows.append(dict(fold=fold, status='insufficient_support', effective_support=neff))
            continue
        if np.average((target[train]-np.average(target[train], weights=wt))**2, weights=wt) < config.epsilon:
            rows.append(dict(fold=fold, status='constant_reference', effective_support=neff))
            continue
        before, after, coefficient = increment_predict(base[train], target[train], wt, base[test],
                                                      alias[train], alias[test], config)
        nulls = []
        for number in range(config.shuffles):
            rng = np.random.default_rng(seed+number)
            nulls.append(increment_predict(base[train], target[train], wt, base[test],
                alias[train][rng.permutation(len(train))], alias[test][rng.permutation(len(test))], config)[1])
        baseline_error = weighted_mse(target[test], before, we)
        extra_error = weighted_mse(target[test], after, we)
        null_error = [weighted_mse(target[test], p, we) for p in nulls]
        rows.append(dict(fold=fold, status='tested', effective_support=neff, baseline_mse=baseline_error,
            alias_mse=extra_error, increment=baseline_error-extra_error,
            increment_fraction=(baseline_error-extra_error)/max(baseline_error, config.epsilon),
            spatial_null_mse=null_error, null_increments=[baseline_error-e for e in null_error],
            alias_coefficient_standardized=coefficient, position=test, weight=we,
            before=before, after=after, nulls=nulls))
    return rows


def logmean(values):
    maximum = values.max(-1)
    return maximum+np.log(np.exp(values-maximum[:, None]).mean(-1))


def numpy(value):
    return value.detach().float().cpu().numpy().astype(np.float64)


def source_diagnostic(source, text_features, config=CONFIG):
    text = numpy(text_features)
    text /= np.linalg.norm(text, axis=-1, keepdims=True)
    members = source.members.cpu().numpy()
    canonical = source.canonical.cpu().numpy()
    classes, count = members.shape
    similarity = np.einsum('ckd,cjd->ckj', text[members], text[members])
    records, pending = [], []
    for tile_number, tile in enumerate(source.tiles):
        local, wide, native, relation = [numpy(v) for v in
            (tile.local_aliases, tile.wide_aliases, tile.native_aliases, tile.relation)]
        valid = tile.valid.cpu().numpy()
        paired = np.sort(numpy(tile.scores['Geometry_PatchOnly2Coupled']).argsort(-1)[:, -2:], axis=-1)
        relation[~valid, :] = 0.
        relation[:, ~valid] = 0.
        np.fill_diagonal(relation, 0.)
        if (relation < 0).any() or not np.isfinite(relation).all():
            raise ValueError('Original nonnegative Geometry W required, not signed H.')
        for c, rival in np.unique(paired[valid], axis=0):
            queries = valid & (paired[:, 0] == c) & (paired[:, 1] == rival)
            support = relation[queries].mean(0)
            for owner, other in ((int(c), int(rival)), (int(rival), int(c))):
                target = native[:, owner, canonical[owner]]-native[:, other, canonical[other]]
                for k in range(count):
                    info = dict(tile=tile_number, owner=owner, rival=other, alias=k,
                                flat_alias=int(members[owner, k]), competing_queries=int(queries.sum()))
                    if similarity[owner, k, canonical[owner]] >= config.near_duplicate_cosine:
                        records.append(dict(**info, status='canonical_or_near_duplicate'))
                        continue
                    keep = similarity[owner, k] < config.near_duplicate_cosine
                    if keep.sum() < 2:
                        records.append(dict(**info, status='insufficient_reference_aliases'))
                        continue
                    base = np.column_stack((logmean(local[:, owner, keep]), logmean(local[:, other]),
                        logmean(wide[:, owner, keep]), logmean(wide[:, other])))
                    seed = config.seed+tile_number*100000+owner*1000+other*20+k
                    for result in cross_fit(base, wide[:, owner, k], target, support, valid, seed, config):
                        predictions = {name: result.pop(name) for name in ('position', 'weight', 'before', 'after', 'nulls')
                                       if name in result}
                        record = dict(**info, excluded_aliases=np.flatnonzero(~keep).tolist(), **result)
                        records.append(record)
                        if predictions:
                            pending.append((record, predictions))
    return records, pending


def semantic_audit(pending, donor_labels):
    """No label is used by the predictor; audit held-out owner/rival discrimination."""
    for row, predictions in pending:
        labels = donor_labels[row['tile']][predictions['position']]
        known = (labels == row['owner']) | (labels == row['rival'])
        weights = predictions['weight'][known]
        labels = labels[known] == row['owner']
        counts = [int(labels.sum()), int((~labels).sum())]
        row['audit_owner_rival_positions'] = counts
        if min(counts) < 4 or any(weights[labels == side].sum() <= 0 for side in (0, 1)):
            row['audit_status'] = 'insufficient_two_class_labels'
            continue
        balanced = weights.copy()
        for side in (0, 1):
            balanced[labels == side] /= 2*weights[labels == side].sum()
        def brier(value):
            probability = 1/(1+np.exp(-np.clip(value[known], -30, 30)))
            return float(balanced@((probability-labels)**2))
        before = brier(predictions['before'])
        after = brier(predictions['after'])
        row.update(audit_status='tested', brier_baseline=before, brier_alias=after,
            brier_increment=before-after,
            brier_null_increments=[before-brier(p) for p in predictions['nulls']])
    return pending
