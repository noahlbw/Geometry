"""Label-free input calibration around the unchanged natural-image reader."""
import math

import numpy as np


IMPLEMENTATION = 'natural-sense-actual-reader-calibration-v1-20261004'
FOREGROUND_REJECTION_BUDGET = .05
MINIMUM_IMAGES = 8


def choose_profile(records, minimum_images=MINIMUM_IMAGES):
    pairs = [(r['default_error'], r['frozen_error']) for r in records
             if r['default_error'] is not None and r['frozen_error'] is not None]
    differences = np.asarray([b-a for a, b in pairs], dtype=np.float64)
    upper = None
    if len(differences) >= max(2, minimum_images):
        upper = float(differences.mean()+1.96*differences.std(ddof=1)/math.sqrt(len(differences)))
    return dict(profile='frozen' if upper is not None and upper < 0 else 'default',
                paired_images=len(pairs), mean_frozen_minus_default_error=float(differences.mean()) if len(pairs) else None,
                upper95_frozen_minus_default_error=upper,
                rule='Default tau=tem=1 unless repeated actual-reader pseudo-witness disagreement favors the source profile.')


def weighted_quantile(values, weights, quantile):
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0 or not 0 <= quantile <= 1):
        raise ValueError('Finite samples and nonnegative nonempty weights required.')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])/weights.sum()
    index = min(int(np.searchsorted(cumulative, quantile, side='left')), len(values)-1)
    return float(values[order[index]])


def foreground_thresholds(records, classes, background, source_threshold,
                          budget=FOREGROUND_REJECTION_BUDGET, minimum_images=MINIMUM_IMAGES):
    if classes < 2 or not math.isfinite(source_threshold) or not 0 <= source_threshold <= 1:
        raise ValueError('Valid class count and bounded source threshold required.')
    thresholds = np.zeros(classes, dtype=np.float64)
    if not background or source_threshold == 0:
        return dict(thresholds=thresholds.tolist(), budget=budget, per_class=[],
                    rule='No residual background rejection in this protocol.')
    thresholds.fill(source_threshold)
    thresholds[0] = 0
    samples, image_counts = [[] for _ in range(classes)], np.zeros(classes, dtype=int)
    for row in records:
        prob = np.asarray(row['probabilities'])
        labels, quality = np.asarray(row['labels']), np.asarray(row['quality'])
        if (prob.ndim != 2 or prob.shape != (len(labels), classes) or quality.shape != labels.shape
                or not np.isfinite(prob).all() or not np.isfinite(quality).all()
                or np.any(quality < 0) or np.any(labels < 0) or np.any(labels >= classes)
                or np.any(prob < 0) or not np.allclose(prob.sum(1), 1, atol=1e-4)):
            raise ValueError('Matching finite pseudo-witness probabilities required.')
        prediction = prob.argmax(1)
        confidence = prob.max(1)
        known = (quality > 0) & (prediction == labels)
        for c in range(1, classes):
            use = known & (labels == c)
            if not np.any(use):
                continue
            samples[c].append((confidence[use], quality[use]/quality[use].sum()))
            image_counts[c] += 1
    global_values, global_weights = [], []
    for c in range(1, classes):
        if samples[c]:
            global_values.extend(np.concatenate([s[0] for s in samples[c]]))
            global_weights.extend(np.concatenate([s[1] for s in samples[c]])/len(samples[c]))
    foreground_images = sum(bool(np.any((np.asarray(r['quality']) > 0)
                            & (np.asarray(r['labels']) > 0)
                            & (np.asarray(r['probabilities']).argmax(1) == np.asarray(r['labels'])))) for r in records)
    global_guard = source_threshold
    if foreground_images >= minimum_images and global_values:
        global_guard = min(source_threshold, weighted_quantile(global_values, global_weights, budget))
    details = []
    for c in range(1, classes):
        reason = 'global foreground guard for sparse class witnesses'
        value = global_guard
        if image_counts[c] >= minimum_images:
            values = np.concatenate([s[0] for s in samples[c]])
            weights = np.concatenate([s[1] for s in samples[c]])
            value = min(source_threshold, weighted_quantile(values, weights, budget))
            reason = 'class-specific image-balanced foreground guard'
        thresholds[c] = value
        details.append(dict(class_index=c, witness_images=int(image_counts[c]), threshold=float(value), reason=reason))
    return dict(thresholds=thresholds.tolist(), budget=budget, global_guard=global_guard,
                foreground_images=foreground_images, per_class=details,
                rule='Source threshold capped by the weighted lower5% confidence of correctly predicted image-only foreground witnesses; no GT fitting.')


def reject_labels(labels, confidence, thresholds, background):
    labels, confidence = np.asarray(labels), np.asarray(confidence)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (labels.shape != confidence.shape or not np.isfinite(confidence).all()
            or not np.isfinite(thresholds).all() or np.any(labels < 0)
            or np.any(labels >= len(thresholds)) or np.any(thresholds < 0) or np.any(thresholds > 1)):
        raise ValueError('Matching labels/confidences and bounded class thresholds required.')
    result = labels.copy()
    if background:
        result[confidence < thresholds[labels]] = 0
    elif np.any(thresholds):
        raise ValueError('A no-background protocol cannot reject valid pixels.')
    return result


def witness_error(probabilities, labels, quality):
    probabilities, labels, quality = map(np.asarray, (probabilities, labels, quality))
    if probabilities.shape != (len(labels), probabilities.shape[-1]) or quality.shape != labels.shape:
        raise ValueError('Matching pseudo-witness arrays required.')
    prediction = probabilities.argmax(1)
    errors = []
    for c in np.unique(labels[quality > 0]):
        use = (labels == c) & (quality > 0)
        errors.append(np.average(prediction[use] != c, weights=quality[use]))
    return float(np.mean(errors)) if errors else None


def sample_accumulated_probabilities(accumulator, coordinates):
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError('Pixel row/column coordinates required.')
    y, x = coordinates.astype(np.int64).T
    if np.any(y < 0) or np.any(y >= accumulator.height) or np.any(x < 0) or np.any(x >= accumulator.width):
        raise ValueError('Witness coordinates outside the image.')
    normalizer = accumulator.normalizer[y, x]
    if np.any(normalizer <= 0):
        raise ValueError('Uncovered image witness.')
    return (accumulator.probabilities[:, y, x] / normalizer[None]).T.copy()


def finalize_thresholds(accumulator, thresholds, background):
    labels = np.empty((accumulator.height, accumulator.width), dtype=np.uint8)
    rows = max(1, min(1024, (64 * 1024 * 1024)//max(accumulator.classes*accumulator.width*4, 1)))
    for top in range(0, accumulator.height, rows):
        bottom = min(top+rows, accumulator.height)
        normalizer = accumulator.normalizer[top:bottom]
        if np.any(normalizer <= 0):
            raise ValueError('Uncovered image pixels.')
        probabilities = accumulator.probabilities[:, top:bottom]/normalizer[None]
        labels[top:bottom] = reject_labels(probabilities.argmax(0), probabilities.max(0), thresholds, background)
    return labels
