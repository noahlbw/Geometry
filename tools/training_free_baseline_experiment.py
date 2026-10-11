"""Prepare, deploy, launch and collect pinned native training-free baselines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import tarfile

import paramiko

WORKSPACE=Path(__file__).resolve().parents[1]
REMOTE=os.environ.get('GEOMETRY_BASELINE_REMOTE_ROOT','/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926')
PYTHON=os.environ.get('GEOMETRY_BASELINE_CONTROLLER_PYTHON','/data/miniconda3/envs/pfu/bin/python')


def checked_connect():
    client=paramiko.SSHClient();client.load_system_host_keys()
    known=Path('C:/Users/CodexSandboxOffline/.ssh/known_hosts')
    if known.exists():client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    askpass=WORKSPACE/'.a800_askpass.cmd'
    password=subprocess.check_output(['cmd.exe','/c',str(askpass)],text=True).strip() if askpass.exists() else None
    client.connect(os.environ.get('GEOMETRY_BASELINE_SSH_HOST','180.76.111.71'),
        username=os.environ.get('GEOMETRY_BASELINE_SSH_USER','root'),password=password,
        look_for_keys=password is None,allow_agent=password is None,timeout=30,
        banner_timeout=30,auth_timeout=30,disabled_algorithms={'kex':['curve25519-sha256@libssh.org']})
    return client


def execute(client,command):
    _,stdout,stderr=client.exec_command(command,timeout=180)
    out=stdout.read().decode('utf-8',errors='replace');err=stderr.read().decode('utf-8',errors='replace')
    if stdout.channel.recv_exit_status():raise RuntimeError(err or out)
    return out


def alive(client,session):
    _,stdout,_=client.exec_command(shlex.join(['tmux','has-session','-t',session]))
    return stdout.channel.recv_exit_status()==0


def mkdir(sftp,path):
    from pathlib import PurePosixPath
    current=''
    for part in PurePosixPath(path).parts:
        if part=='/':current='/';continue
        current=current.rstrip('/')+'/'+part
        try:sftp.stat(current)
        except FileNotFoundError:sftp.mkdir(current)


def remote_json(sftp,path):
    try:
        with sftp.open(path) as stream:return json.loads(stream.read().decode())
    except FileNotFoundError:return None

LOCAL=WORKSPACE/'research/training_free_baselines_full_20261011'
ROOT=REMOTE+'/results/training_free_baselines_full_20261011'
SOURCE=WORKSPACE/'third_party/training_free_ovss_20261010'
SESSION='gtf11_controller'
FILES=('export_training_free_samples.py','eval_training_free_baseline.py',
       'run_training_free_baselines.py','prepare_training_free_weights.py',
       'training_free_baseline_runtime.py','setup_training_free_baselines.sh','setup_training_free_extra.sh')
METHODS={
    'PEARL':('PEARL','pearl_ovss','PEARL'),
    'GLA-CLIP':('GLA-CLIP','gla_clip_segmentor','GLA_CLIPSegmentation'),
    'PTC':('PTC','ptc_segmentor','PTCSegmentation'),
    'CorrCLIP':('CorrCLIP/CorrCLIPv1','corrclip_segmentor','CorrCLIPSegmentation'),
    'SegEarth-OV':('SegEarth-OV','segearth_segmentor','SegEarthSegmentation'),
}
NATURAL=dict(voc20='voc20',voc21='voc21',context59='context59',context60='context60',
    ade150='ade20k',coco_object81='coco_object',coco_stuff171='coco_stuff164k')
RS=dict(vdd='vdd',potsdam='potsdam',udd5='udd5',vaihingen='vaihingen',oem='openearthmap',loveda='loveda')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def prepare():
    if (LOCAL/'protocol.json').exists():
        prepare_cost_manifest()
        print('Frozen protocol already prepared; preserved.');return
    source=read(WORKSPACE/'research/vip_kev_same_vocabulary_20261010/protocol.json')
    manifest=read(WORKSPACE/'research/training_free_baselines_20261010/source_manifest.json')
    datasets={d:dict(family=e['family'],background_index=e['background_index'],
        total_images=e['total_images'],data_root=e['data_root'],
        class_names=[c['name'] for c in e['banks']['original']['classes']]) for d,e in source['datasets'].items()}
    methods={}
    for method,(directory,module,classname) in METHODS.items():
        repo_name=directory.split('/')[0]
        provenance=next(x for x in manifest if Path(x['local_path']).name==repo_name)
        repo=SOURCE/directory
        base=runpy.run_path(str(repo/'configs/base_config.py'))['model']
        configs={}
        for d,e in datasets.items():
            native=(method!='SegEarth-OV' and d in NATURAL) or (method=='SegEarth-OV' and d in RS)
            suffix=NATURAL.get(d) if method!='SegEarth-OV' else RS.get(d)
            path=repo/'configs'/('cfg_'+str(suffix)+'.py')
            value=runpy.run_path(str(path)) if native else {}
            model={**base,**value.get('model',{})}
            if native:
                word=(repo/model.pop('name_path')).read_bytes()
                resize=next(t for t in value['test_pipeline'] if t['type']=='Resize')
                resize={k:resize[k] for k in ('scale','keep_ratio')}
                word_origin=str(path.relative_to(SOURCE))
            else:
                # No selected/tuned Geometry words. Natural transfer uses the
                # same public PEARL task ontology, not a generated alias pool.
                if d in NATURAL:
                    natural_cfg=runpy.run_path(str(SOURCE/'PEARL/configs'/('cfg_'+NATURAL[d]+'.py')))
                    word=(SOURCE/'PEARL'/natural_cfg['model']['name_path']).read_bytes()
                    resize=dict(scale=[2048,448],keep_ratio=True)
                    word_origin='PEARL/configs/cls_'+NATURAL[d]+'.txt (fixed transfer ontology)'
                else:
                    delimiter='; ' if method=='CorrCLIP' else ', '
                    word=('\n'.join(e['class_names'])+'\n').encode()
                    resize=dict(scale=[448,448],keep_ratio=True)
                    word_origin='one locked canonical dataset class per line; no alias tuning'
                model.pop('name_path',None)
                model['prob_thd']=0.0
                if method=='SegEarth-OV':model['bg_idx']=e['background_index'] or 0
            # Exact original file bytes and native parser are retained.
            lines=word.decode().splitlines()
            prediction_map=None
            if method=='SegEarth-OV' and d=='oem' and len(lines)==9:
                prediction_map=[8,*range(8)]
            if method=='SegEarth-OV' and d=='vaihingen' and len(lines)==6:
                prediction_map=[*range(5),5]
            if len(lines)!=len(e['class_names']) and prediction_map is None:
                raise RuntimeError(f'Native query classes mismatch {method}/{d}: {len(lines)}')
            word_file=f'words/{method}/{d}.txt'
            destination=LOCAL/word_file;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(word)
            cfg=dict(method=method,module=module,**{'class':classname},
                source_path=ROOT+'/sources/'+directory,model=model,
                word_file=word_file,word_sha256=hashlib.sha256(word).hexdigest(),
                word_origin=word_origin,resize=resize,
                native_protocol_config=native,config_origin=str(path.relative_to(SOURCE)) if native else 'fixed transfer',
                parser='semicolon-space' if method=='CorrCLIP' else 'native comma-space',
                full_flags={},prediction_map=prediction_map)
            if method=='PEARL':cfg['model']['use_prop']='on'
            if method=='GLA-CLIP':cfg['full_flags']=dict(CLIP_type='ProxyCLIP',token_norm=True,
                KV_token_extension=True,proxy_sim=True,mini_iters=2,initial_crit_pos=.6,
                dynamic_beta=True,beta_alpha=.3,dynamic_gamma=True,gamma_alpha=30.)
            if method=='CorrCLIP':
                cfg['model']['mask_generator']='sam2'
                cfg['mask_note']='Native online SAM2 HieraLarge generation, 8 points/side, IoU/stability .4; included in inference cost. Official precomputed-mask scores are a separate resource setting.'
            if d=='loveda':
                foreground=('\n'.join(lines[1:])+'\n').encode()
                cfg['foreground_word_file']=f'words/{method}/loveda_P.txt'
                (LOCAL/cfg['foreground_word_file']).write_bytes(foreground)
                cfg['foreground_note']='Separate six-class P inference. Rejection disabled because P has no background slot. D retains native config when available.'
            configs[d]=cfg
        methods[method]=dict(repository=provenance['url'],commit=provenance['commit'],
            source_subdirectory=directory,configs=configs)
    protocol=dict(implementation='pinned-native-training-free-baseline-suite-v1-20261011',
        datasets=datasets,methods=methods,method_order=list(METHODS),
        dataset_order=['vdd','potsdam','udd5','vaihingen','oem','loveda','landcoverai','flair1',
            'voc20','voc21','ade150','context59','context60','coco_object81','coco_stuff171'],
        expected_protocol_images=45200,
        provenance='Authors native official config where provided; fixed untuned transfer elsewhere. '
            'Matched existing images/targets. Different pretrained backbones and external resources remain part of each full method; '
            'this is not an operator ablation. Geometry references used prior labeled development optimization.')
    write(LOCAL/'protocol.json',protocol)
    # Only metrics/coverage are copied into comparison references, not caches.
    for d in datasets:
        reference=read(WORKSPACE/'research/vip_kev_same_vocabulary_20261010'/d/'merged.json')
        write(LOCAL/'references'/(d+'.json'),dict(coverage=dict(
            sample_keys=reference['signature']['sample_keys'],
            global_sample_keys_sha256=reference['signature']['global_sample_keys_sha256']),
            results=reference['metrics'],source='verified local finite-row-repaired native VIP reimplementation, not paper numbers'))
    print('Prepared frozen configurations: 5 methods x 15 protocols.')
    prepare_cost_manifest()


def prepare_cost_manifest():
    if (LOCAL/'cost_keys.json').exists():return
    source=read(WORKSPACE/'research/vip_kev_same_vocabulary_20261010/comparison_summary.json')['comparisons']
    keys={d:[r['key'] for r in source[d+'__'+('D' if d=='loveda' else d)]['cost']['images']]
        for d in read(LOCAL/'protocol.json')['datasets']}
    if any(len(v)!=7 or len(set(v))!=7 for v in keys.values()):raise RuntimeError('Matched VIP timing keys missing')
    write(LOCAL/'cost_keys.json',keys)


def deploy(client,sftp):
    mkdir(sftp,ROOT)
    for name in FILES:
        sftp.put(str(WORKSPACE/'DINOtool/scripts'/name),REMOTE+'/scripts/'+name)
    for path in [LOCAL/'protocol.json',LOCAL/'cost_keys.json',*(LOCAL/'words').rglob('*.txt'),*(LOCAL/'references').glob('*.json')]:
        target=ROOT+'/'+path.relative_to(LOCAL).as_posix();mkdir(sftp,str(Path(target).parent).replace('\\','/'))
        existing=remote_json(sftp,target) if path.suffix=='.json' else None
        if existing is not None and existing!=read(path):raise RuntimeError('Frozen remote file differs: '+target)
        sftp.put(str(path),target)
    archive=LOCAL/'sources.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in METHODS:
            for path in (SOURCE/name).rglob('*'):
                if path.is_file() and not any(p in ('.git','__pycache__') for p in path.parts) and path.suffix not in ('.pt','.pth','.ckpt','.safetensors'):
                    tar.add(path,arcname=path.relative_to(SOURCE).as_posix(),recursive=False)
    sftp.put(str(archive),ROOT+'/sources.tar.gz')
    command='cd '+shlex.quote(REMOTE)+' && mkdir -p '+shlex.quote(ROOT+'/sources')+' && tar -xzf '+shlex.quote(ROOT+'/sources.tar.gz')+' -C '+shlex.quote(ROOT+'/sources')
    print(execute(client,command))
    args=[PYTHON,'scripts/export_training_free_samples.py','--source',
        ROOT+'/protocol.json','--output',ROOT+'/samples.json']
    print(execute(client,'cd '+shlex.quote(REMOTE)+' && PYTHONPATH=.:scripts '+shlex.join(args)))


def start(client,sftp):
    if alive(client,SESSION):print('Existing controller preserved.');return
    if remote_json(sftp,ROOT+'/suite_status.json'):raise RuntimeError('Controller already has state; inspect before recovery')
    cmd=[PYTHON,'-u','scripts/run_training_free_baselines.py','--root',ROOT]
    shell='cd '+shlex.quote(REMOTE)+' && exec env PYTHONPATH=.:scripts '+shlex.join(cmd)+' > '+shlex.quote(ROOT+'/controller.log')+' 2>&1'
    print(execute(client,shlex.join(['tmux','new-session','-d','-s',SESSION,shell])))


def resume(client,sftp):
    if alive(client,SESSION):print('Existing controller preserved.');return
    state=remote_json(sftp,ROOT+'/suite_status.json')
    if not state:raise RuntimeError('Use start for a new suite')
    if state['status']=='complete':print('Complete suite preserved.');return
    cmd=[PYTHON,'-u','scripts/run_training_free_baselines.py','--root',ROOT,'--resume']
    shell='cd '+shlex.quote(REMOTE)+' && exec env PYTHONPATH=.:scripts '+shlex.join(cmd)+' > '+shlex.quote(ROOT+'/controller_resume.log')+' 2>&1'
    print(execute(client,shlex.join(['tmux','new-session','-d','-s',SESSION,shell])))


def status(client,sftp):
    state=remote_json(sftp,ROOT+'/suite_status.json') or {}
    print(json.dumps(dict(controller_alive=alive(client,SESSION),status=state.get('status'),
        current_method=state.get('current_method'),completed_jobs=len(state.get('completed',[])),
        pending_jobs=len(state.get('pending',[])),merged=state.get('merged',{}),failures=state.get('failures',{})),ensure_ascii=False))
    for job in state.get('active',[]):
        row=remote_json(sftp,job['output']+'/results.json') or {}
        print(json.dumps(dict(method=job['method'],dataset=job['dataset'],phase=job['phase'],shard=job['shard'],
            gpu=job['gpu'],alive=alive(client,job['session']),processed=row.get('processed_images'),total=row.get('total_images'))))


def collect(client,sftp):
    state=remote_json(sftp,ROOT+'/suite_status.json') or {}
    write(LOCAL/'suite_status.json',state)
    protocol=read(LOCAL/'protocol.json')
    lines=['# Pinned native training-free baseline comparison','',protocol['provenance'],'',
        'Status: '+str(state.get('status','preparing'))+'. Only verified complete protocols receive scores.','']
    scores={}
    costs={}
    for method in METHODS:
        scores[method]={}
        for d,e in protocol['datasets'].items():
            merged=remote_json(sftp,f'{ROOT}/{method}/{d}/merged.json')
            cost=remote_json(sftp,f'{ROOT}/{method}/{d}/cost/s0/cost.json')
            if cost:
                write(LOCAL/method/d/'cost.json',cost);costs[method+'/'+d]=cost
            if not merged:continue
            write(LOCAL/method/d/'merged.json',merged);scores[method][d]=merged['results']
    old=read(WORKSPACE/'research/vip_kev_same_vocabulary_20261010/comparison_summary.json')['comparisons']
    references={}
    lines+=['|Protocol|Images|Geometry best measured|Strong verified local VIP|'+ '|'.join(METHODS)+'|',
            '|---|---:|---:|---:|'+ '|'.join('---:' for _ in METHODS)+'|']
    for d in protocol['dataset_order']:
        for p in ('P','D') if d=='loveda' else (d,):
            row=old[d+'__'+p]
            our=row['ours']
            vip=max(v for v in (row.get('historical_vip'),row.get('same_text'),row.get('official_template')) if v is not None)
            if d in ('context60','voc20','voc21'):
                latest=read(WORKSPACE/'research/natural_alias_best_20261010'/d/'merged.json')
                our=latest['metrics'][p]['CuratedTuned']['mean_iou_percent']
            references[d+'__'+p]=dict(ours=our,vip=vip,
                provenance='Best previously measured labeled-development/full-validation-viewed Geometry configuration; strongest of verified local VIP references. Neither is a new evaluation or a published paper number.')
            values=[f"{scores[m][d][p]['mean_iou_percent']:.4f}" if d in scores[m] else 'pending' for m in METHODS]
            label=d+(' '+p if d=='loveda' else '')
            lines.append(f"|{label}|{protocol['datasets'][d]['total_images']}|{our:.4f}|{vip:.4f}|"+'|'.join(values)+'|')
    rs=[d for d,e in protocol['datasets'].items() if e['family']=='remote_sensing']
    mean=lambda values:sum(values)/len(values)
    our_mean=mean([references[d+'__'+('D' if d=='loveda' else d)]['ours'] for d in rs])
    vip_mean=mean([references[d+'__'+('D' if d=='loveda' else d)]['vip'] for d in rs])
    means=[f"{mean([scores[m][d]['D' if d=='loveda' else d]['mean_iou_percent'] for d in rs]):.4f}" if all(d in scores[m] for d in rs) else 'pending' for m in METHODS]
    lines.append(f'|Eight remote mean|20092|{our_mean:.4f}|{vip_mean:.4f}|'+ '|'.join(means)+'|')
    lines+=['','## Standalone whole-image cost','',
        '|Method|Protocol|Median ms|Peak allocated MiB|Native config|',
        '|---|---|---:|---:|---|']
    for key,cost in costs.items():
        method,d=key.split('/')
        peak=max(x['peak_allocated_bytes'] for x in cost['images'])/1024**2
        lines.append(f"|{method}|{d}|{cost['median_ms']:.2f}|{peak:.2f}|{protocol['methods'][method]['configs'][d]['native_protocol_config']}|")
    lines+=['','Reproduce: `python tools/training_free_baseline_experiment.py prepare`, `deploy`, `start`, `status`, `collect`.',
        'Pinned commits, exact query files, official settings and fixed transfer policies are in `protocol.json` and `words/`.',
        'CorrCLIP uses native online SAM2 masks, including their cost; SegEarth-OV retains pretrained SimFeatUp.',
        'No evaluation labels select a baseline vocabulary, threshold or transfer setting. LoveDA P is read independently.',
        'VOC20/21 and PC59/60 reuse photographs. Eight remote domains count LoveDA D once; LandCover.ai substitutes for unavailable labeled iSAID.',
        'Runtime uses a dedicated Torch2.4/MMCV2.2 environment and an explicitly recorded MMSeg version-guard compatibility change.']
    report=WORKSPACE/'research/TRAINING_FREE_BASELINES_FULL_20261011.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(LOCAL/'comparison_summary.json',dict(status=state.get('status','preparing'),methods=scores,previous_references=references,
        completed_protocols=sum(len(v) for v in scores.values()),expected_protocols=75,costs=costs))
    print(str(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','deploy','start','resume','status','collect'))
    args=parser.parse_args()
    if args.action=='prepare':prepare();return
    client=checked_connect()
    try:
        with client.open_sftp() as sftp:globals()[args.action](client,sftp)
    finally:client.close()


if __name__=='__main__':main()
