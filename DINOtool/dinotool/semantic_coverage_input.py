"""One metadata-only replacement proposal for a fixed twenty-slot bank.

These checks establish bookkeeping and lexical eligibility. They do not prove
semantic ownership or visual usefulness; an explicit pre-evaluation text
review must approve each surviving proposal before compiling candidates.
"""
from collections import Counter, defaultdict
import json


IMPLEMENTATION='fixed20-single-generation-semantic-coverage-input-v1-20261009'
RELATIONS=('synonym','subtype','visual_variant')
FIELDS={'slot_id','phrase','relation','meaning_added','basis'}


def key(value):return ' '.join(value.replace('_',' ').replace('-',' ').casefold().split())


def prompt(record):
    metadata={name:record[name] for name in ('domain','class_index','class_name','official_name_field',
        'sense_clarification','competitor_names','queries','protected_slots','eligible_slots')}
    return ('Construct optional replacement queries for a frozen semantic-segmentation vocabulary using text metadata only. '
        'The official name field is category-name/synonym metadata, not a complete annotation definition. '
        'No image, label, class frequency or model score is available.\n'+json.dumps(metadata,ensure_ascii=True)+'\n'
        'Keep the target meaning and every protected concept. Propose zero to M replacements, where M is the number '
        'of eligible slots. There is no requirement to fill the budget. Bind each phrase to a distinct eligible slot_id; '
        'do not offer alternatives for the same slot. An empty replacements array is valid.\n'
        'Allowed relations: synonym (another ordinary name of the same intended concept), subtype (a genuine narrower '
        'kind), or visual_variant (a familiar concrete appearance of that same object/surface, explicitly named). '
        'Do not collapse subtypes into generic synonyms or repeat a concept already adequately expressed by the current queries. '
        'Exclude another scored category, broader superclass, separate part, nearby object, scene description or unsupported '
        'annotation claim. Do not infer precedence between overlapping categories. Omit uncertain ownership. '
        'Do not pad with plurals, visible/in-the-image/photographic wrappers, or systematic lists of colors/materials. '
        'An appearance qualifier must denote a real familiar distinction. Do not infer frequency or segmentation usefulness.\n'
        'Return only JSON with exactly the key replacements, an array of objects with exactly slot_id (integer), phrase '
        '(English noun phrase, at most8 words), relation (synonym|subtype|visual_variant), meaning_added and basis '
        '(each a concise explanation, at most12 words). No markdown or confidence scores.')


def parse_reply(reply,record):
    value=json.loads(reply)
    if not isinstance(value,dict) or set(value)!={'replacements'} or not isinstance(value['replacements'],list):
        raise ValueError('Exact replacements schema required.')
    entries=value['replacements']
    if len(entries)>len(record['eligible_slots']):raise ValueError('Proposal exceeds eligible-slot budget.')
    for item in entries:
        if (not isinstance(item,dict) or set(item)!=FIELDS or type(item['slot_id']) is not int
                or any(not isinstance(item[k],str) for k in FIELDS-{'slot_id'}) or item['relation'] not in RELATIONS):
            raise ValueError('Invalid typed replacement schema.')
    slots=Counter(item['slot_id'] for item in entries);phrases=Counter(key(item['phrase']) for item in entries)
    old={key(word) for word in record['queries']};rivals={key(word) for word in record['competitor_names']}
    accepted=[];rejected=[]
    for raw in entries:
        item={k:(' '.join(v.split()) if isinstance(v,str) else v) for k,v in raw.items()}
        reason=None;phrase=item['phrase'];normalized=key(phrase)
        if item['slot_id'] not in record['eligible_slots']:reason='protected_or_invalid_slot'
        elif slots[item['slot_id']]!=1:reason='multiple_alternatives_for_slot'
        elif not phrase or len(phrase.split())>8:reason='invalid_phrase_length'
        elif not item['meaning_added'] or not item['basis'] or max(len(item[k].split()) for k in ('meaning_added','basis'))>12:
            reason='missing_or_long_explanation'
        elif normalized in old:reason='already_present_query'
        elif phrases[normalized]!=1:reason='duplicate_generated_phrase'
        elif normalized in rivals:reason='exact_competitor_name'
        elif any(token in normalized for token in ('in the image','in a scene','in the scene','in view','photograph','pictured','depicted')) or normalized.startswith(('visible ','the visible ','a visible ')):
            reason='cosmetic_wrapper'
        (rejected if reason else accepted).append(dict(item,reason=reason) if reason else item)
    return accepted,rejected


def lexical_filter(records):
    """Reject both owners of a shared addition, with no winner selection."""
    old_owners=defaultdict(set);new_owners=defaultdict(set)
    for record in records:
        owner=(record['dataset'],record['class_index'])
        for phrase in record['queries']:old_owners[(record['dataset'],key(phrase))].add(owner)
        for item in record['accepted']:new_owners[(record['dataset'],key(item['phrase']))].add(owner)
    for record in records:
        owner=(record['dataset'],record['class_index']);keep=[]
        for item in record['accepted']:
            identifier=(record['dataset'],key(item['phrase']));reason=None
            if len(new_owners[identifier])>1:reason='shared_new_phrase'
            elif old_owners[identifier]-{owner}:reason='exact_rival_inherited_phrase'
            if reason:record['rejected'].append(dict(item,reason=reason))
            else:keep.append(item)
        record['accepted']=keep


def compile_banks(records,review):
    if review.get('target_images_loaded') or review.get('target_masks_loaded') or not review.get('complete'):
        raise ValueError('Complete metadata-only review required.')
    indices=defaultdict(list)
    for record in records:indices[record['dataset']].append(record['class_index'])
    if any(values!=list(range(len(values))) for values in indices.values()):
        raise ValueError('Restore complete frozen class-index order before compilation.')
    decisions={(row['dataset'],row['class_index'],row['slot_id']):row for row in review['decisions']}
    if len(decisions)!=len(review['decisions']):raise ValueError('Duplicate review decision.')
    expected={(r['dataset'],r['class_index'],item['slot_id']) for r in records for item in r['accepted']}
    if set(decisions)!=expected:raise ValueError('Review must cover every mechanically eligible proposal exactly.')
    output=defaultdict(lambda:{'Existing20':[],'Coverage20':[],'SynonymOnly20':[]});audit=[]
    for record in records:
        old=list(record['queries']);coverage=old.copy();synonyms=old.copy();kept=[]
        if len(old)!=20 or len(set(map(key,old)))!=20:raise ValueError('Twenty distinct inherited queries required.')
        for item in record['accepted']:
            decision=decisions[(record['dataset'],record['class_index'],item['slot_id'])]
            if (decision.get('phrase')!=item['phrase'] or decision.get('relation')!=item['relation']
                    or type(decision.get('accept')) is not bool or not decision.get('reason')
                    or type(decision.get('relation_confirmed')) is not bool):
                raise ValueError('Review phrase/relation/decision/reason differs.')
            if decision['accept']:
                if not decision['relation_confirmed']:
                    raise ValueError('Accepted proposal requires confirmed original relation.')
                coverage[item['slot_id']]=item['phrase'];kept.append(item)
                if item['relation']=='synonym':synonyms[item['slot_id']]=item['phrase']
        for group in (coverage,synonyms):
            if len(set(map(key,group)))!=20:raise ValueError('Replacement introduced a duplicate.')
            if any(group[i]!=old[i] for i in record['protected_slots']):raise ValueError('Protected semantic slot changed.')
        for arm,group in (('Existing20',old),('Coverage20',coverage),('SynonymOnly20',synonyms)):
            output[record['dataset']][arm].append(dict(name=record['class_name'],synonyms=group))
        audit.append(dict(dataset=record['dataset'],class_index=record['class_index'],class_name=record['class_name'],
            eligible_slots=record['eligible_slots'],protected_slots=record['protected_slots'],
            approved=kept,inherited_count=20-len(kept),generated_semantic_count=len(kept)))
    for dataset,arms in output.items():
        for arm,classes in arms.items():
            owners={}
            for row in classes:
                for phrase in row['synonyms']:
                    normalized=key(phrase)
                    if normalized in owners and owners[normalized]!=row['name']:
                        raise ValueError('Cross-class duplicate after compilation: '+dataset+'/'+arm+'/'+phrase)
                    owners[normalized]=row['name']
    return dict(datasets=dict(output),audit=audit,implementation=IMPLEMENTATION,
        target_images_loaded=False,target_masks_loaded=False,target_label_tuning=False)
