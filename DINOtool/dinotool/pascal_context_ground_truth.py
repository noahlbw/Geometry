"""MMSeg's fixed Pascal Context category mapping for original Stanford MAT masks."""
import numpy as np


# IDs from OpenMMLab's pascal_context.py converter, in its sorted class order.
SOURCE_IDS = tuple(sorted((
    0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
    158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
    440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
    85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115)))
CLASSES = (
    'background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
    'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car',
    'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain',
    'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
    'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
    'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep', 'shelves',
    'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train',
    'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood')


def map_full_category_mask(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype.kind not in 'iu' or raw.size == 0 or raw.min() < 0 or raw.max() > 459:
        raise ValueError('Original Pascal Context 0..459 integer LabelMap required.')
    lookup = np.zeros(460, dtype=np.uint8)
    lookup[list(SOURCE_IDS)] = np.arange(len(SOURCE_IDS), dtype=np.uint8)
    return lookup[raw]


def validate_label_dictionary(text):
    labels = {int(line.split(':', 1)[0]): line.split(':', 1)[1].strip()
              for line in text.splitlines() if line.strip()}
    if len(labels) != 459 or any(labels.get(raw_id) != name for raw_id, name in zip(SOURCE_IDS[1:], CLASSES[1:])):
        raise ValueError('Stanford label IDs/names do not match the fixed MMSeg taxonomy.')
    return labels
