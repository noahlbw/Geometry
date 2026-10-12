"""Bounded PC59/60 alias and scalar search; preserve every incumbent artifact."""
import argparse
import copy
import json
from pathlib import Path
import shlex

from training_free_baseline_experiment import WORKSPACE, REMOTE, PYTHON, checked_connect, execute, mkdir, remote_json, alive

LOCAL=WORKSPACE/'research/context_corrclip_tuning_20261012'
ROOT=REMOTE+'/results/context_corrclip_tuning_20261012'
SESSION='gpcc12_controller'
FILES=('scripts/eval_context_corrclip_tuning.py','scripts/run_context_corrclip_tuning.py')
TARGETS=dict(context59=47.7036,context60=45.1164)
PRECISE={
    'wood':['wood','timber','lumber','wooden plank','piece of wood','wooden material'],
    'track':['railway track','railroad track','train track','rail tracks','railroad rails'],
    'platform':['railway platform','station platform','train platform','raised platform'],
    'light':['light','lamp','light fixture','ceiling lamp','streetlight'],
    'ground':['ground','soil','bare ground','dirt','earth surface'],
    'cloth':['cloth','fabric','textile','piece of fabric'],
    'computer':['computer','desktop computer','laptop computer','computer system'],
    'shelves':['shelf','shelves','shelving','bookshelf','storage shelves'],
}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare():
    from curated_variable_natural_experiment import CURATED
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Prepared trial is preserved.')
    old=read(WORKSPACE/'research/kev_alias_search_20261009/protocol.json')
    retained=WORKSPACE/'research/efficient_geometry_20261012'
    entries={}
    for d in TARGETS:
        e=copy.deepcopy(old['datasets'][d]);config=read(retained/'configs'/f'{d}.yaml')
        vocab=read(retained/'vocabularies'/f'{d}.json')
        current=vocab['banks'][config['profile']['bank']]
        other='context60' if d=='context59' else 'context59'
        peer=next(iter(read(retained/'vocabularies'/f'{other}.json')['banks'].values()))
        names=[c['name'] for c in current['classes']];bg=e['background_index']
        lookup={c['name']:c['synonyms'] for c in peer['classes']}
        groups={
            'peer_best':[lookup.get(n,current['classes'][i]['synonyms']) for i,n in enumerate(names)],
            'literal':[current['classes'][i]['synonyms'] if i==bg else CURATED[n] for i,n in enumerate(names)],
            'precise':[current['classes'][i]['synonyms'] if i==bg else PRECISE.get(n,CURATED[n]) for i,n in enumerate(names)],
            'natural_anchor':[current['classes'][i]['synonyms'] if i==bg else [CURATED[n][0]] for i,n in enumerate(names)],
        }
        e['banks']['incumbent']=copy.deepcopy(current)
        for label,g in groups.items():
            for suffix,template in (('segmentation','seg_template'),('imagenet','ImageNet80')):
                e['banks'][label+'_'+suffix]=dict(template=template,
                    classes=[dict(name=n,synonyms=list(dict.fromkeys(a))) for n,a in zip(names,g)])
        e['search_banks']=[k for k in e['banks'] if k!='original']
        e['coordinate_banks']=['incumbent','peer_best_segmentation','literal_segmentation',
            'precise_segmentation','natural_anchor_segmentation','official_segmentation','semantic_segmentation']
        for suffix,template in (('segmentation','seg_template'),('imagenet','ImageNet80')):
            candidates=list(e['banks'].values())
            all_groups=[list(dict.fromkeys(a for b in candidates for a in b['classes'][i]['synonyms'])) for i in range(len(names))]
            e['banks']['candidate_pool_'+suffix]=dict(template=template,
                classes=[dict(name=n,synonyms=a) for n,a in zip(names,all_groups)])
        e.update(incumbent_config=config,incumbent_vocabulary=vocab,incumbent_profile={**config['profile'],'bank':'incumbent'},
            incumbent_bias=config['background_bias'],incumbent_threshold=config['background_threshold'],
            incumbent_metrics=read(retained/'merged'/f'{d}.json')['metrics']['efficient'][d],
            corrclip_target=TARGETS[d],coarse_count=12,search_count=64,confirmation_count=256,shards=4,
            fixed_background_rule=config['profile']['background_rule'])
        entries[d]=e
    protocol=dict(implementation='pc-alias-scalar-only-v1-20261012',datasets=entries,order=list(entries),
        allowed='Aliases/templates and scalar readout/coupling/tau/tem/local-temperature/background calibration only.',
        frozen='Visual weights, Geometry relation/depth, local+wide operators, coupling equation, background rule/residual ontology, cap4+4/zero fine and image resizing.',
        evaluation='Exploratory labeled development on previously viewed validation images. Freeze a development-selected choice before one full run; preserve failures and negative results.',
        search=dict(strengths=[1,1.5,2,2.5,3,3.5,4],coupling=[.25,.5,.625,.75,.875,1,1.25],
            tau=[.5,.75,1,1.5,2.2,3],tem=[.3,.5,1,2,3,10],temperature=[.04,.05,.06,.07,.08,.1,.12],
            biases=[0,-.25,-.5,.25,.5],thresholds=[0,.025,.05,.075,.1,.125,.15,.2],class_sweeps=1),
        cost=dict(samples=7,repeats=7),gpu_order=[4,5,6,7],expected_protocol_images=10210)
    write(LOCAL/'protocol.json',protocol)
    write(LOCAL/'cost_keys.json',{d:read(retained/'cost_keys.json')[d] for d in entries})
    print(json.dumps(dict(protocol=str(LOCAL/'protocol.json'),datasets=list(entries),full_images=10210)))


def deploy(client,sftp):
    if alive(client,SESSION):raise RuntimeError('Controller is already active.')
    mkdir(sftp,ROOT)
    for name in FILES:
        target=REMOTE+'/'+name;source=(WORKSPACE/'DINOtool'/name).read_bytes()
        try:
            with sftp.open(target,'rb') as stream:existing=stream.read()
        except FileNotFoundError:existing=None
        if existing is not None and existing!=source:raise RuntimeError('Preserve existing source: '+name)
        if existing is None:sftp.put(str(WORKSPACE/'DINOtool'/name),target)
    for name in ('protocol.json','cost_keys.json'):
        path=ROOT+'/'+name
        existing=remote_json(sftp,path)
        if existing is not None and existing!=read(LOCAL/name):raise RuntimeError('Preserve remote frozen protocol.')
        if existing is None:sftp.put(str(LOCAL/name),path)


def start(client,sftp):
    deploy(client,sftp)
    if remote_json(sftp,ROOT+'/state.json'):raise RuntimeError('Existing trial state preserved; inspect status.')
    third=str(Path(REMOTE).parent).replace('\\','/')+'/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c'
    environment=['env','PYTHONPATH='+REMOTE+':'+REMOTE+'/scripts:'+third,'OMP_NUM_THREADS=2']
    execute(client,shlex.join(environment+[PYTHON,'-c','import eval_context_corrclip_tuning; import run_context_corrclip_tuning; print("worker imports ok")']))
    command=environment+[PYTHON,'-u',
        'scripts/run_context_corrclip_tuning.py','--suite-root',ROOT]
    shell='cd '+shlex.quote(REMOTE)+' && exec '+shlex.join(command)+' > '+shlex.quote(ROOT+'/controller.log')+' 2>&1'
    execute(client,shlex.join(['tmux','new-session','-d','-s',SESSION,shell]))
    print(json.dumps(dict(controller=SESSION,root=ROOT)))


def status(client,sftp,collect=False):
    state=remote_json(sftp,ROOT+'/state.json') or {}
    write(LOCAL/'state.json',state)
    print(json.dumps(dict(controller_alive=alive(client,SESSION),status=state.get('status'),
        completed=sum(j['status']=='complete' for j in state.get('jobs',{}).values()),
        jobs=len(state.get('jobs',{})),failures={k:v for k,v in state.get('jobs',{}).items() if v['status']=='failed'})))
    for key,job in state.get('jobs',{}).items():
        if job['status'] in ('running','failed'):
            row=remote_json(sftp,job['output']+'/progress.json') or {}
            print(json.dumps(dict(key=key,alive=alive(client,job['session']),progress=row)))
            if job['mode']=='search':
                try:passes=sorted(n for n in sftp.listdir(job['output']) if n.startswith('pass_') and n.endswith('.json'))
                except FileNotFoundError:passes=[]
                if passes:
                    last=remote_json(sftp,job['output']+'/'+passes[-1])
                    print(json.dumps(dict(key=key,last_pass=passes[-1],stage=last['stage'],images=len(last['sample_keys']),
                        development_miou=last['best']['development_miou'],profile=last['best']['profile'])))
            if not alive(client,job['session']) and not remote_json(sftp,job['output']+'/results.json'):
                print(execute(client,shlex.join(['tail','-n','45',job['log']])))
    for d in TARGETS:
        for remote,local in ((f'search/{d}/selection.json',f'{d}/selection.json'),
            (f'merged/{d}.json',f'{d}/merged.json'),(f'cost/{d}/results.json',f'{d}/cost.json'),
            (f'configs/{d}.yaml',f'configs/{d}.yaml'),(f'vocabularies/{d}.json',f'vocabularies/{d}.json')):
            row=remote_json(sftp,ROOT+'/'+remote)
            if row is not None:write(LOCAL/local,row)
        choice_path=LOCAL/d/'selection.json'
        if choice_path.exists():
            choice=read(choice_path)
            last=remote_json(sftp,ROOT+f'/search/{d}/pass_{choice["search_passes"]:03d}.json')
            if last:write(LOCAL/d/'final_confirmation.json',last)
    suite=remote_json(sftp,ROOT+'/suite_results.json')
    if suite:write(LOCAL/'suite_results.json',suite)
    if collect:
        if not suite or suite['status']!='complete':raise RuntimeError('Full trial is not complete.')
        lines=['# PC59/PC60 alias and scalar tuning versus CorrCLIP','',read(LOCAL/'protocol.json')['evaluation'],'',
            '|Protocol|Images|Incumbent|Tuned|CorrCLIP|Tuned minus CorrCLIP|Median ms|Peak MiB|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        comparisons={}
        for d,target in TARGETS.items():
            row=read(LOCAL/d/'merged.json');cost=read(LOCAL/d/'cost.json');entry=read(LOCAL/'protocol.json')['datasets'][d]
            if not all(row[f] for f in ('coverage_verified','per_image_sums_verified','scored_target_counts_match','development_replay_verified')):
                raise RuntimeError('Unverified result '+d)
            value=row['metrics']['mean_iou_percent'];old=entry['incumbent_metrics']['mean_iou_percent']
            comparisons[d]=dict(previous=old,tuned=value,corrclip=target,delta_corrclip_pp=value-target,
                delta_previous_pp=value-old,beats_corrclip=value>target,improves_incumbent=value>old,
                retained_recommendation='trial' if value>old else 'incumbent',
                cost=cost,selection=read(LOCAL/d/'selection.json'),full=row)
            lines.append(f'|{d}|5105|{old:.4f}|{value:.4f}|{target:.4f}|{value-target:+.4f}|{cost["median_ms"]:.2f}|{cost["peak_allocated_mib"]:.2f}|')
        lines+=['','Only words/templates and declared scalars change. Geometry/local/wide/coupled equations, '
            'residual ontology, image views and pretrained weights are frozen. Variable alias counts also '
            'change inherited wide aggregation mass. CorrCLIP uses MetaCLIP plus online SAM2; these are '
            'local full-method comparisons on the same scored inputs, not published-paper numbers.',
            '', 'Timing uses the same seven fixed inputs, two warmups and seven synchronized repeats; '
            'whole RGB to CPU IDs, excluding decode/init/offline search. Original incumbents remain preserved.',
            '', 'Development selection averages classes with ground-truth support in the development subset. '
            'The split summaries below use the standard union-valid mean, which also scores false-positive-only '
            'classes. Their mIoU values can differ even though the development confusion matrix replays exactly.']
        for d,c in comparisons.items():
            entry=read(LOCAL/'protocol.json')['datasets'][d]
            lines+=['','## '+d,'','Frozen choice: `'+json.dumps(c['selection']['profile'])+'`.',
                '',f'Development 256; complement 4849. Bias {c["selection"]["background_bias"]}; threshold {c["selection"]["background_threshold"]}.',
                '',f'Development mIoU {c["full"]["splits"]["development"]["mean_iou_percent"]:.4f}; '
                f'complement mIoU {c["full"]["splits"]["complement"]["mean_iou_percent"]:.4f}.',
                '', '|Parameter|Incumbent|Frozen trial|','|---|---|---|']
            for key in ('strength','coupling','temperature','tau','tem','wide_policy','background_rule'):
                lines.append(f'|{key}|{entry["incumbent_profile"][key]}|{c["selection"]["profile"][key]}|')
            lines += [f'|background bias|{entry["incumbent_bias"]}|{c["selection"]["background_bias"]}|',
                f'|background threshold|{entry["incumbent_threshold"]}|{c["selection"]["background_threshold"]}|',
                '', '|Class|Incumbent IoU|Tuned IoU|Delta pp|Selected aliases|','|---|---:|---:|---:|---|']
            bank=next(iter(c['selection']['vocabulary'].values()))
            for a,b,words in zip(read(LOCAL/'protocol.json')['datasets'][d]['incumbent_metrics']['per_class'],c['full']['metrics']['per_class'],bank['classes']):
                if a['iou_percent'] is not None and b['iou_percent'] is not None:
                    lines.append(f'|{a["name"]}|{a["iou_percent"]:.4f}|{b["iou_percent"]:.4f}|{b["iou_percent"]-a["iou_percent"]:+.4f}|'+', '.join(words['synonyms'])+'|')
        write(LOCAL/'comparison_summary.json',dict(status='complete',comparisons=comparisons))
        report=WORKSPACE/'research/CONTEXT_CORRCLIP_TUNING_20261012.md'
        report.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','start','status','collect'))
    action=p.parse_args().action
    if action=='prepare':prepare()
    else:
        c=checked_connect()
        try:
            with c.open_sftp() as s:
                if action=='start':start(c,s)
                else:status(c,s,action=='collect')
        finally:c.close()
