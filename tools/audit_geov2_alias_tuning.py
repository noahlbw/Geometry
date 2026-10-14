"""Audit frozen alias provenance and recombine already evaluated gain settings."""
import json
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
root = workspace / 'research/geov2_g0_sweep_20261014'
summary = json.loads((root / 'SUMMARY.json').read_text())
snapshot = workspace / 'research/query_slot_best_alias_20261013'
if not (snapshot / 'configs/voc21.yaml').exists():
    snapshot = workspace / 'artifacts/geometry_github/research/query_slot_best_alias_20261013'
rows = []
for dataset in summary['datasets']:
    config = json.loads((snapshot / 'configs' / (dataset + '.yaml')).read_text())
    vocabulary = json.loads((snapshot / 'vocabularies' / (dataset + '.json')).read_text())
    bank = vocabulary['banks'][config['profile']['bank']]
    counts = [len(c['synonyms']) for c in bank['classes']]
    rows.append({'dataset':dataset,'bank':config['profile']['bank'],'aliases':sum(counts),
                 'counts':counts,'template':bank['template'],
                 'source_trial':config['provenance']['source_trial'],
                 'labeled_development':config['provenance']['labeled_development'],
                 'full_validation_viewed':config['provenance']['full_validation_viewed']})

audit = []
source_path = workspace / 'tmp/vip_source_audit.txt'
archive_path = root / 'ALIAS_TUNING_AUDIT.json'
if source_path.exists():
    for line in source_path.read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('{'):
            item = json.loads(line)
            if item['path'].startswith('configs/cfg_') or item['path'] == 'dinosegmentor.py':
                audit.append(item)
elif archive_path.exists():
    audit = json.loads(archive_path.read_text())['vip_source_excerpts']
else:
    raise FileNotFoundError('Pinned VIP source excerpts are required for the configuration audit.')

policies = [('shared_half', {}),('voc21_one', {'voc21':1}),
            ('vdd_voc21_one', {'vdd':1,'voc21':1}),
            ('four_exceptions', {'vdd':1,'voc21':1,'oem':.25,'vaihingen':.25})]
computed = []
for name, overrides in policies:
    choices = {d:overrides.get(d,.5) for d in summary['datasets']}
    per_dataset = {d:summary['datasets'][d]['metrics']['retained__g'+format(g,'g')] for d,g in choices.items()}
    computed.append({'name':name,'choices':choices,'macro_miou':sum(per_dataset.values())/len(per_dataset)})
data = {'scope':'Code/provenance audit and arithmetic recombination of existing full-sweep results. No new model, alias, inference, or independent validation.',
        'current_aliases':rows,'vip_commit':'5bd25ee03ec25c1538622cf7da661e8c0461e769',
        'vip_source':'https://github.com/MiSsU-HH/VIP','vip_source_excerpts':audit,
        'post_hoc_policy_recombinations':computed}
(root / 'ALIAS_TUNING_AUDIT.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'aliases':{r['dataset']:{'bank':r['bank'],'count':r['aliases']} for r in rows},
                  'macro_miou':{r['name']:r['macro_miou'] for r in computed}},indent=2))
