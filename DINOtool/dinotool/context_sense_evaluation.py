"""Scoring contract for the frozen Context word-transfer evaluation."""
import numpy as np


IMPLEMENTATION = 'context-sense-frozen-reader-transfer-v1-20261004'


def frozen_source_input(candidate, vocabulary, verified_identity, source_implementation):
    dataset, names = candidate['dataset'], candidate['class_names']
    groups = candidate['source_identity']['groups']
    if (dataset not in ('context59', 'context60')
            or names != [c['name'] for c in vocabulary['classes']]
            or candidate['source_identity']['dataset'] != dataset
            or candidate['source_identity']['classes'] != names
            or len(groups) != len(names)
            or any(not group or group[0] != name for name, group in zip(names, groups))):
        raise ValueError('Aligned fixed Context taxonomy/source groups required.')
    # Only these fields form the deployed model identity; metadata is not calibration evidence.
    identity = dict(checkpoints=verified_identity['checkpoints'],
        upstream_commit=verified_identity['upstream_commit'], dataset=dataset,
        classes=names, groups=groups, implementation=source_implementation)
    counts = [len(group) for group in groups]
    return dict(status='complete', dataset=dataset, identity=identity,
        source_type='text_only_frozen_preset', source_image_selection_exists=False,
        selected_indices=list(range(sum(counts))), pool_counts=counts, selected_counts=counts,
        chosen=dict(template='openai_imagenet_template', tau=1., tem=1.,
                    prob_thd=.1 if dataset == 'context60' else 0.),
        image_keys=[], processed_images=0, total_images=0,
        target_masks_loaded=False, target_label_tuning=False, transductive=False,
        threshold_reason='Fixed transfer preset, not fitted on Context images or masks.')


def confusion(target, prediction, classes, reject=False):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError('Matching full-image target and prediction required.')
    valid = (target >= 0) & (target < classes)
    pred = np.where(prediction == 255, classes, prediction) if reject else prediction
    columns = classes+int(reject)
    if np.any(pred[valid] < 0) or np.any(pred[valid] >= columns):
        raise ValueError('Unexpected predicted/rejected category.')
    matrix = np.bincount(target[valid]*columns+pred[valid], minlength=classes*columns).reshape(classes, columns)
    return matrix, int((~valid).sum())
