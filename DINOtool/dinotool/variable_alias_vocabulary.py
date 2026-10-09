"""Metadata-only variable alias generation contract; no visual quality claim."""
import json
import copy

MAX_ADDITIONS=12
KINDS=('synonym','subtype')


def key(phrase):
    return ' '.join(phrase.replace('_',' ').replace('-',' ').casefold().split())


def prompt(name,anchor,competitors,family):
    return ('Generate optional English visual queries for semantic segmentation using only category metadata. '
        'Image domain: '+family+'. Category label: '+json.dumps(name)+'. Intended visual concept: '+json.dumps(anchor)+'. '
        'Other scored concepts: '+json.dumps(competitors)+'. '
        'Return between0 and12 additions, not a fixed quota. Never pad the list with plural forms or cosmetic paraphrases. '
        'Each phrase must denote the intended category itself (synonym) or an unambiguous subtype belonging to it (subtype). '
        'Do not return another scored category, a broader superclass, a part, a co-occurring object, scene, color alone, '
        'generic material, or invented annotation rule. Distinguish ambiguous word senses. '
        'Avoid subtypes assigned to another scored category. Do not repeat the intended concept. '
        'If no safe addition exists return an empty array. Short phrases, at most8 words. '
        'Return only a JSON array of objects with exactly phrase and kind keys; kind is synonym or subtype. '
        'No explanation, confidence, markdown, images or performance assumptions.')


def parse_reply(reply,anchor,competitors):
    start=reply.find('[')
    if start<0:raise ValueError('No JSON array.')
    items,_=json.JSONDecoder().raw_decode(reply[start:])
    if not isinstance(items,list):raise ValueError('JSON array required.')
    seen={key(anchor)};rivals={key(p) for p in competitors};accepted=[];rejected=[]
    for item in items:
        if not isinstance(item,dict) or set(item)!= {'phrase','kind'} or not isinstance(item['phrase'],str) or item['kind'] not in KINDS:
            raise ValueError('Typed phrase/kind objects required.')
        phrase=' '.join(item['phrase'].split());normalized=key(phrase);reason=None
        if not phrase or len(phrase.split())>8:reason='invalid_length'
        elif normalized in seen:reason='duplicate_or_anchor'
        elif normalized in rivals:reason='exact_competitor_concept'
        elif len(accepted)>=MAX_ADDITIONS:reason='uniform_upper_budget'
        if reason:rejected.append(dict(**item,reason=reason));continue
        seen.add(normalized);accepted.append(dict(phrase=phrase,kind=item['kind']))
    return accepted,rejected


def remove_shared_additions(records):
    owners={}
    for i,record in enumerate(records):
        for item in record['accepted']:owners.setdefault(key(item['phrase']),set()).add(i)
    for record in records:
        keep=[]
        for item in record['accepted']:
            if len(owners[key(item['phrase'])])>1:
                record['rejected'].append(dict(**item,reason='exact_shared_generated_phrase'))
            else:keep.append(item)
        record['accepted']=keep


def freeze_pool(protocol,source):
    """Fail closed per invalid class, preserving the failed raw generation."""
    if source['target_images_loaded'] or source['target_masks_loaded'] or source['target_label_tuning']:
        raise ValueError('Metadata-only source required.')
    records=copy.deepcopy(source['records']);datasets={};fallback=[]
    for d,e in protocol['datasets'].items():
        bank_name='semantic_segmentation' if e['family']=='natural' else 'original_imagenet'
        names=e['banks'][bank_name]['classes']
        cohort=[r for r in records if r['dataset']==d]
        expected={i for i in range(len(names)) if i!=e['background_index']}
        if {r['index'] for r in cohort}!=expected or len(cohort)!=len(expected):
            raise ValueError('Complete distinct foreground requests required.')
        for r in cohort:
            if r['name']!=names[r['index']]['name'] or r['anchor']!=names[r['index']]['synonyms'][0]:
                raise ValueError('Changed class/anchor mapping.')
            if not r.get('format_complete',False):
                r['accepted']=[];fallback.append(dict(dataset=d,index=r['index'],name=r['name']))
        remove_shared_additions(cohort)
        banks=copy.deepcopy(e['banks'])
        for bank in banks.values():
            for r in cohort:
                i=r['index'];anchor=bank['classes'][i]['synonyms'][0]
                bank['classes'][i]['synonyms']=[anchor]+[x['phrase'] for x in r['accepted']]
        datasets[d]=banks
    return dict(status='candidate_pool_ready',generation_status=source['status'],
        fallback_rule='Any schema-failed class retains canonical anchor only; no selective item salvage or regeneration.',
        canonical_only_format_fallbacks=fallback,datasets=datasets,records=records,
        semantic_quality_verified=False,target_images_loaded=False,target_masks_loaded=False,target_label_tuning=False)
