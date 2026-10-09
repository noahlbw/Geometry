"""Fixed-setting comparison contract; no learned or visual readout changes."""
import math


IMPLEMENTATION = 'natural-fixed-reader-qualified-queries-v1-20261004'
METHODS = ('V2_Words', 'Qualified_Words', 'V2_Words_NoThreshold', 'Qualified_Words_NoThreshold')


def paired_inputs(candidate, baseline, threshold):
    if candidate.class_names != baseline.class_names or not candidate.class_names:
        raise ValueError('Identical nonempty annotation class order required.')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Frozen bounded source threshold required.')
    if candidate.class_names[0] != 'background' and threshold != 0:
        raise ValueError('A foreground-only protocol cannot invent rejection.')
    profile = dict(template='seg_template', tau=1., tem=1., prob_thd=threshold)
    return dict(default=candidate, frozen=baseline), dict(default=dict(profile), frozen=dict(profile))


def rename_predictions(predictions):
    names = dict(Sense_Words='V2_Words', Sense_Default='Qualified_Words',
                 Sense_Words_NoThreshold='V2_Words_NoThreshold',
                 Sense_Default_NoThreshold='Qualified_Words_NoThreshold')
    if set(predictions) != set(names):
        raise ValueError('The paired word study must contain only the two fixed-setting arms.')
    return {method: predictions[next(k for k, v in names.items() if v == method)] for method in METHODS}
