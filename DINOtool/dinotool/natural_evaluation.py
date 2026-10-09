"""Image-only vocabulary expansion and locked natural segmentation protocols."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


TOTALS = dict(voc20=1449, voc21=1449, context59=5105, context60=5105,
              ade150=2000, coco_stuff171=5000, coco_object81=5000, cityscapes19=500)
CLASS_COUNTS = dict(voc20=20, voc21=21, context59=59, context60=60,
                    ade150=150, coco_stuff171=171, coco_object81=81, cityscapes19=19)
ROOTS = dict(voc20='VOCdevkit/VOC2012', voc21='VOCdevkit/VOC2012',
             context59='VOCdevkit/VOC2010', context60='VOCdevkit/VOC2010',
             ade150='ADEChallengeData2016', coco_stuff171='COCOStuff164k',
             coco_object81='COCOObject', cityscapes19='Cityscapes')
LEXICON = {
    'aeroplane': ('airplane', 'plane', 'aircraft'),
    'bicycle': ('bike', 'pedal bicycle', 'pedal bike'),
    'bird': ('avian animal', 'feathered bird'),
    'boat': ('watercraft', 'small boat', 'sailing boat'),
    'bottle': ('drink bottle', 'glass bottle', 'plastic bottle'),
    'bus': ('passenger bus', 'public transit bus', 'city bus'),
    'car': ('automobile', 'passenger car', 'motorcar'),
    'cat': ('feline', 'domestic cat', 'house cat'),
    'chair': ('seat chair', 'seating chair', 'individual chair'),
    'cow': ('cattle animal', 'bovine', 'domestic cow'),
    'diningtable': ('dining table', 'meal table', 'dinner table'),
    'dining table': ('meal table', 'dinner table'),
    'dog': ('canine', 'domestic dog', 'pet dog'),
    'horse': ('equine animal', 'domestic horse'),
    'motorbike': ('motorcycle', 'motorized bike'),
    'motorcycle': ('motorbike', 'motorized bike'),
    'person': ('human', 'human being', 'individual person', 'people'),
    'pottedplant': ('potted plant', 'plant in a pot', 'pot plant'),
    'potted plant': ('plant in a pot', 'pot plant'),
    'sheep': ('ovine animal', 'domestic sheep'),
    'sofa': ('couch', 'settee', 'upholstered sofa'),
    'couch': ('sofa', 'settee', 'upholstered couch'),
    'train': ('railway train', 'railroad train', 'rail vehicle'),
    'tvmonitor': ('television', 'TV screen', 'television monitor'),
    'tv': ('television', 'TV screen', 'television set'),
    'wall': ('wall surface', 'vertical wall', 'interior wall'),
    'building': ('building structure', 'constructed building', 'building exterior'),
    'sky': ('open sky', 'sky above', 'overhead sky'),
    'floor': ('floor surface', 'indoor floor', 'room floor'),
    'tree': ('woody tree', 'tree canopy', 'branching tree'),
    'ceiling': ('room ceiling', 'overhead ceiling', 'indoor ceiling'),
    'road': ('roadway', 'street road', 'paved road'),
    'grass': ('grassy ground', 'grass cover', 'grass surface'),
    'sidewalk': ('footpath', 'pedestrian walkway', 'street sidewalk'),
    'earth': ('bare earth', 'soil surface', 'exposed soil'),
    'door': ('door panel', 'doorway door', 'entry door'),
    'mountain': ('mountain terrain', 'mountain peak', 'mountain slope'),
    'plant': ('growing plant', 'leafy plant', 'green plant'),
    'water': ('body of water', 'water surface', 'open water'),
    'sea': ('ocean', 'seawater', 'ocean water'),
    'river': ('river water', 'flowing river', 'river channel'),
    'sand': ('sandy ground', 'sand surface', 'sandy area'),
    'rock': ('stone', 'natural rock', 'rock surface'),
    'snow': ('snow cover', 'snow surface', 'snowy ground'),
    'fence': ('fencing', 'fence structure', 'boundary fence'),
    'bed': ('sleeping bed', 'bed furniture', 'room bed'),
    'cabinet': ('storage cabinet', 'cupboard', 'cabinet furniture'),
    'windowpane': ('window pane', 'window glass', 'glass window'),
    'curtain': ('window curtain', 'hanging curtain', 'drapery'),
    'table': ('table furniture', 'tabletop furniture'),
    'desk': ('work desk', 'writing desk', 'desk furniture'),
    'shelf': ('shelving', 'storage shelf'),
    'armchair': ('arm chair', 'chair with armrests'),
    'wardrobe': ('clothes wardrobe', 'clothing cabinet'),
    'lamp': ('lighting lamp', 'light fixture', 'illuminating lamp'),
    'rug': ('floor rug', 'area rug', 'floor carpet'),
    'bench': ('seating bench', 'long bench'),
    'bag': ('carry bag', 'carrying bag'),
    'backpack': ('rucksack', 'knapsack', 'back pack'),
    'umbrella': ('rain umbrella', 'handheld umbrella'),
    'handbag': ('purse', 'hand bag', 'carried handbag'),
    'tie': ('necktie', 'neck tie'),
    'suitcase': ('travel suitcase', 'luggage case'),
    'truck': ('lorry', 'cargo truck', 'motor truck'),
    'airplane': ('aeroplane', 'aircraft', 'plane'),
    'traffic light': ('traffic signal', 'road traffic light'),
    'fire hydrant': ('hydrant', 'street fire hydrant'),
    'stop sign': ('road stop sign', 'stop traffic sign'),
    'parking meter': ('street parking meter', 'parking payment meter'),
    'elephant': ('elephant animal', 'large elephant'),
    'bear': ('bear animal', 'wild bear'),
    'zebra': ('striped zebra', 'zebra animal'),
    'giraffe': ('giraffe animal', 'long-necked giraffe'),
    'frisbee': ('flying disc', 'throwing disc'),
    'skis': ('ski equipment', 'snow skis'),
    'snowboard': ('snowboarding board', 'snow board'),
    'sports ball': ('sport ball', 'game ball'),
    'kite': ('flying kite', 'toy kite'),
    'baseball bat': ('bat for baseball', 'baseball hitting bat'),
    'baseball glove': ('baseball mitt', 'catching glove'),
    'skateboard': ('skate board', 'wheeled skateboard'),
    'surfboard': ('surf board', 'surfing board'),
    'tennis racket': ('tennis racquet', 'racket for tennis'),
    'wine glass': ('wineglass', 'glass for wine'),
    'cup': ('drinking cup', 'beverage cup'),
    'fork': ('dining fork', 'eating fork'),
    'knife': ('table knife', 'cutting knife'),
    'spoon': ('eating spoon', 'table spoon'),
    'bowl': ('food bowl', 'serving bowl'),
    'banana': ('banana fruit', 'ripe banana'),
    'apple': ('apple fruit', 'whole apple'),
    'orange': ('orange fruit', 'citrus orange'),
    'sandwich': ('bread sandwich', 'sandwich food'),
    'broccoli': ('broccoli vegetable', 'broccoli florets'),
    'carrot': ('carrot vegetable', 'carrot root'),
    'hot dog': ('hotdog', 'sausage in a bun'),
    'pizza': ('pizza pie', 'pizza food'),
    'donut': ('doughnut', 'ring doughnut'),
    'cake': ('cake dessert', 'baked cake'),
    'toilet': ('toilet fixture', 'toilet bowl'),
    'laptop': ('laptop computer', 'notebook computer'),
    'mouse': ('computer mouse', 'desktop mouse'),
    'remote': ('remote control', 'handheld remote control'),
    'keyboard': ('computer keyboard', 'typing keyboard'),
    'cell phone': ('mobile phone', 'cellphone', 'smartphone'),
    'microwave': ('microwave oven', 'kitchen microwave'),
    'oven': ('cooking oven', 'kitchen oven'),
    'toaster': ('bread toaster', 'kitchen toaster'),
    'sink': ('wash basin', 'sink basin'),
    'refrigerator': ('fridge', 'kitchen refrigerator'),
    'book': ('printed book', 'bound book'),
    'clock': ('timepiece', 'time clock'),
    'vase': ('flower vase', 'decorative vase'),
    'scissors': ('cutting scissors', 'pair of scissors'),
    'teddy bear': ('stuffed bear', 'toy bear'),
    'hair drier': ('hair dryer', 'blow dryer'),
    'toothbrush': ('tooth brush', 'dental brush'),
}
PARAPHRASES = (
    '{}', 'a {}', 'the {}', 'visible {}', '{} in a scene', '{} in the scene',
    '{} in view', '{} in the image', '{} in a picture', '{} seen in a scene',
    'a visible {}', 'the visible {}', '{} as seen in an image',
    '{} visible in the image', '{} visible in the scene', '{} within a scene',
    '{} appearing in the scene', '{} in a natural image', '{} seen in the image',
    '{} present in the scene',
)
BACKGROUND = (
    'background', 'scene background', 'image background', 'non-target region',
    'non-target objects', 'other objects', 'other scene content', 'unclassified region',
    'unclassified objects', 'unlabeled scene content', 'background region',
    'background content', 'background area', 'background parts of the scene',
    'non-target scene content', 'unspecified objects', 'unspecified scene content',
    'remaining scene content', 'other image content', 'non-category region',
)


def expand_aliases(name):
    if name == 'background':
        return BACKGROUND
    literal = name.replace('-', ' ').replace('_', ' ')
    roots = tuple(dict.fromkeys((name, literal, *LEXICON.get(name, ()))))
    result = []
    for phrase in PARAPHRASES:
        for root in roots:
            value = phrase.format(root)
            if value.casefold() not in {v.casefold() for v in result}:
                result.append(value)
            if len(result) == 20:
                return tuple(result)
    raise ValueError('Could not produce 20 distinct candidates: '+name)


@dataclass(frozen=True)
class NaturalSample:
    key: str
    image_path: Path
    mask_path: Path


def discover_samples(dataset, data_root):
    root = Path(data_root)
    if dataset.startswith('voc'):
        keys = (root/'ImageSets/Segmentation/val.txt').read_text().split()
        samples = [NaturalSample(k, root/'JPEGImages'/(k+'.jpg'), root/'SegmentationClass'/(k+'.png')) for k in keys]
    elif dataset.startswith('context'):
        keys = (root/'ImageSets/SegmentationContext/val.txt').read_text().split()
        samples = [NaturalSample(k, root/'JPEGImages'/(k+'.jpg'), root/'SegmentationClassContext'/(k+'.png')) for k in keys]
    elif dataset == 'ade150':
        samples = [NaturalSample(p.stem, p, root/'annotations/validation'/(p.stem+'.png'))
                   for p in sorted((root/'images/validation').glob('*.jpg'))]
    elif dataset.startswith('coco'):
        suffix = '_labelTrainIds.png' if dataset == 'coco_stuff171' else '_instanceTrainIds.png'
        samples = [NaturalSample(p.stem, p, root/'annotations/val2017'/(p.stem+suffix))
                   for p in sorted((root/'images/val2017').glob('*.jpg'))]
    elif dataset == 'cityscapes19':
        samples = []
        for p in sorted((root/'leftImg8bit/val').glob('*/*_leftImg8bit.png')):
            stem = p.name.removesuffix('_leftImg8bit.png')
            mask = root/'gtFine/val'/p.parent.name/(stem+'_gtFine_labelTrainIds.png')
            if not mask.is_file():
                mask = mask.with_name(stem+'_gtFine_labelIds.png')
            samples.append(NaturalSample(p.parent.name+'/'+stem, p, mask))
    else:
        raise ValueError('Unknown natural dataset: '+dataset)
    if len(samples) != TOTALS[dataset] or len({s.key for s in samples}) != len(samples):
        raise ValueError('Incorrect full validation coverage: '+dataset)
    if any(not s.image_path.is_file() or not s.mask_path.is_file() for s in samples):
        raise FileNotFoundError('Missing validation image/mask pair: '+dataset)
    return samples


def load_rgb(sample):
    import torch

    with Image.open(sample.image_path) as image:
        array = np.asarray(image.convert('RGB')).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def map_target(raw, dataset, city_raw=False):
    if raw.ndim != 2:
        raise ValueError('Indexed 2D ground truth is required.')
    result = np.full(raw.shape, -1, dtype=np.int64)
    if dataset == 'cityscapes19' and city_raw:
        for index, value in enumerate((7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)):
            result[raw == value] = index
        return result
    shifted = dataset in ('voc20', 'context59', 'ade150')
    lower = 1 if shifted else 0
    upper = CLASS_COUNTS[dataset] if shifted else CLASS_COUNTS[dataset]-1
    allowed = ((raw >= lower) & (raw <= upper)) | (raw == 255)
    if shifted:
        allowed |= raw == 0
    if not allowed.all():
        raise ValueError('Unexpected raw label IDs for '+dataset)
    valid = (raw >= lower) & (raw <= upper)
    result[valid] = raw[valid].astype(np.int64)-int(shifted)
    return result


def load_target(sample, dataset, shape):
    with Image.open(sample.mask_path) as image:
        target = map_target(np.asarray(image), dataset, sample.mask_path.name.endswith('_labelIds.png'))
    if target.shape != shape:
        raise ValueError('Original image/mask shapes differ: '+sample.key)
    return target
