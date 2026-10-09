"""Input contracts for paired, fixed-reader alias calibration experiments."""
from itertools import accumulate


IMPLEMENTATION = 'natural-frozen-alias-count-paired-evaluation-v1-20261004'


def validate_selections(source, count, curated, source_sha):
    if (source['status'] != 'complete' or source['target_masks_loaded']
            or source['target_label_tuning'] or count['status'] != 'complete'
            or count['dataset'] != source['dataset']
            or count['target_masks_loaded'] or count['target_label_tuning']
            or count['smoke_only'] or not count['weights_frozen']
            or not count['head_weights_unchanged']
            or count['source_selection_sha256'] != source_sha
            or count['source_identity'] != source['identity']
            or count['selected_counts'] != source['selected_counts']
            or count['source_choice'] != source['chosen']
            or count['image_keys'] != source['image_keys']
            or count['processed_images'] != count['total_images']
            or count['total_images'] != len(set(count['image_keys']))):
        raise ValueError('Complete, unchanged, image-only source and count selection required.')
    strength = count['count_calibration']['strength']
    if strength not in (0., .5, 1.):
        raise ValueError('Unexpected frozen count-calibration strength.')
    if curated is not None:
        for field in ('identity', 'chosen', 'image_keys', 'target_masks_loaded', 'target_label_tuning'):
            if curated[field] != source[field]:
                raise ValueError('Text curation changed non-alias input: '+field)
        indices = curated['selected_indices']
        parents = [c for c, group in enumerate(source['identity']['groups']) for _ in group]
        if not set(indices).issubset(source['selected_indices']):
            raise ValueError('Curation cannot introduce unselected source aliases.')
        counts = [sum(parents[i] == c for i in indices) for c in range(len(source['identity']['classes']))]
        anchors = [0, *accumulate(source['pool_counts'][:-1])]
        if (curated['status'] != 'complete' or indices != sorted(set(indices))
                or not set(anchors).issubset(indices) or not all(counts)
                or counts != curated['selected_counts']):
            raise ValueError('Curation must remove only noncanonical source aliases.')
    return strength
