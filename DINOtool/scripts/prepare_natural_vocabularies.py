"""Freeze semantic-only twenty-candidate vocabularies without loading images/masks."""
import argparse
import ast
import json
from pathlib import Path

import requests

from dinotool.natural_evaluation import CLASS_COUNTS, expand_aliases


SOURCES = {key: 'https://raw.githubusercontent.com/open-mmlab/mmsegmentation/v1.2.2/mmseg/datasets/'+file
           for key, file in dict(voc='voc.py', context='pascal_context.py', ade='ade.py',
                                coco='coco_stuff.py', city='cityscapes.py').items()}
COUNTS = dict(voc=21, context=59, ade=150, coco=171, city=19)


def classes_from_source(text, expected):
    tree = ast.parse(text)
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'METAINFO' for t in node.targets):
            if isinstance(node.value, ast.Call):
                candidates.extend(ast.literal_eval(k.value) for k in node.value.keywords if k.arg == 'classes')
            elif isinstance(node.value, ast.Dict):
                candidates.append(ast.literal_eval(node.value)['classes'])
    matching = [tuple(c) for c in candidates if len(c) == expected]
    if not matching or any(c != matching[0] for c in matching):
        raise ValueError('Could not identify the locked class taxonomy.')
    return matching[0]


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    taxonomies = {}
    for key, url in SOURCES.items():
        cached = output/(key+'_mmseg_source.py')
        if cached.exists():
            text = cached.read_text()
        else:
            response = requests.get(url, timeout=(15, 30))
            response.raise_for_status()
            text = response.text
            classes_from_source(text, COUNTS[key])
            cached.write_text(text)
        taxonomies[key] = classes_from_source(text, COUNTS[key])
    names = dict(voc20=taxonomies['voc'][1:], voc21=taxonomies['voc'],
                 context59=taxonomies['context'], context60=('background', *taxonomies['context']),
                 ade150=taxonomies['ade'], coco_stuff171=taxonomies['coco'],
                 coco_object81=('background', *taxonomies['coco'][:80]), cityscapes19=taxonomies['city'])
    for dataset, classes in names.items():
        if len(classes) != CLASS_COUNTS[dataset] or len(set(classes)) != len(classes):
            raise ValueError('Class count/order error: '+dataset)
        result = dict(dataset=dataset, expansion='natural-semantic20-v1-20261003',
                      aliases_per_class=20, target_images_loaded=False, target_masks_loaded=False,
                      source_taxonomies=SOURCES, lexical_policy='Canonical names, curated synonyms where available, then neutral linguistic paraphrases; not twenty distinct visual concepts.',
                      classes=[dict(name=name, synonyms=expand_aliases(name)) for name in classes])
        path = output/(dataset+'_20.json')
        if path.exists():
            if json.loads(path.read_text()) != json.loads(json.dumps(result)):
                raise RuntimeError('Refusing to change frozen vocabulary: '+dataset)
        else:
            path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(dataset=dataset, classes=len(classes), aliases=20*len(classes))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
