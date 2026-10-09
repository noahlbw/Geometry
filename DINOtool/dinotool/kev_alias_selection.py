"""Text-only variable-length admission; no pixels, masks, or task scores here."""
import math

IMPLEMENTATION = 'frozen-kev-taxonomy-alias-admission-v1-20261009'
CHECKPOINT = 'jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52'
THRESHOLDS = (.4, .6, .8)
OPTIONS = ('Specific to the target class and excludes the other listed classes',
           'Can describe the target but also another listed class, or is too broad',
           'Describes a different class or contradicts the target',
           'Cannot determine a meaningful visual interpretation')


def unique(words):
    out, seen = [], set()
    for word in words:
        word = word.strip()
        key = word.casefold()
        if word and key not in seen:
            out.append(word); seen.add(key)
    return out


def candidate_groups(entry):
    original = entry['banks']['original']['classes']
    names = [c['name'] for c in original]
    for bank in entry['banks'].values():
        if [c['name'] for c in bank['classes']] != names:
            raise ValueError('Candidate banks changed the scored class order.')
    return [unique([name]+[a for b in entry['banks'].values()
                          for a in b['classes'][i]['synonyms']]) for i,name in enumerate(names)]


def context(entry, index):
    names = [c['name'] for c in entry['banks']['original']['classes']]
    family = 'overhead remote sensing' if entry['family']=='remote_sensing' else 'natural photographs'
    return ('Task: pixel-level semantic segmentation of '+family+'.\n'
            'The entire mutually competing output taxonomy is: '+', '.join(names)+'.\n'
            'Target class: '+names[index]+'.\n'
            'Judge literal visual meaning relative to every competing label. A subtype of the target '
            'is valid if its pixels belong to that target in this taxonomy. Co-occurrence, part of '
            'another object, a place containing an object, and visual similarity alone are not synonyms. '
            'Respect explicit class granularity: roof is distinct from wall when both are labels; '
            'a car can be a vehicle, but not all vehicles are cars. Background/other is residual. '
            'Do not estimate accuracy, use benchmark scores, or assume that high image response makes a word valid.')


def question(alias, rotation=0):
    order = tuple((j+rotation)%4 for j in range(4))
    return dict(instr='Classify this proposed target-class alias: '+repr(alias)+'.',
                options=[OPTIONS[j] for j in order], label=0), order


def restored(probabilities, order):
    p = [float(v) for v in probabilities]
    if len(p)!=4 or any(not math.isfinite(v) or v<0 for v in p) or abs(sum(p)-1)>1e-3:
        raise ValueError('Judge returned invalid four-way probabilities.')
    result=[0.]*4
    for j,k in enumerate(order):result[k]=p[j]
    return result


def admit(names, groups, scores, threshold, background):
    if len(names)!=len(groups) or len(scores)!=len(names):
        raise ValueError('Judgments do not cover the complete class order.')
    classes=[]
    for i,(name,words,rows) in enumerate(zip(names,groups,scores)):
        if i==background:
            selected=words
        else:
            selected=[name]
            for alias in words:
                if alias.casefold()==name.casefold():continue
                probability=rows[alias]['probabilities']
                # Shared evidence may still be useful, but is weaker than exclusivity.
                if probability[0]+.5*probability[1]>=threshold:
                    selected.append(alias)
        classes.append(dict(name=name,synonyms=unique(selected)))
    return classes
