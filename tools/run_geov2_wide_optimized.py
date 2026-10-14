"""Run the opt-in wide/readout optimization screen on A800."""
import argparse
import json
import shlex
import time
from pathlib import Path

from a800_region_experiment import connect
from pointwise_context_experiment import WORKSPACE, REMOTE, PYTHON, THIRD

LOCAL = WORKSPACE/'research/geov2_wide_optimized_20261015'
REMOTE_ROOT = REMOTE+'/results/geov2_wide_optimized_20261015'
SOURCE = WORKSPACE/'research/response_local_wide_20261013_v3/protocol.json'
DATASETS=('vdd','potsdam','ade150','voc20')
GPUS=(1,2,3,4,5,6,7)
VARIANTS=('baseline','shared_canvas','dual_direct','shared_direct','shared_native','shared_regression','shared_short336','shared_fp32','shared_slots48','shared_visual','shared_visual_canvas','optimized48','optimized24','ridge01','ridge03','ridge1')


def retry():
    last=None
    for i in range(8):
        try:return connect()
        except Exception as exc:last=exc;time.sleep(1.5*(i+1))
    raise last


def prepare():
    source=json.loads(SOURCE.read_text())
    p=json.loads(json.dumps(source))
    root_remote=REMOTE+'/results/query_slot_best_alias_20261013'
    p['implementation']='geov2-wide-optimized-20261015'
    p['checkpoint_dir']='/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO'
    p['dinov3_repo']=REMOTE+'/dinov3_hub'
    p['upstream_root']='/data/test/code/ovss_cafe_ped_v2_20260919/third_party/VIP_official_5bd25ee'
    p['screen_images']=16
    cost_keys=json.loads((WORKSPACE/'research/query_slot_best_alias_20261013/cost_keys.json').read_text())
    p['cost_keys']=cost_keys
    complete=json.loads((WORKSPACE/'research/query_slot_best_alias_20261013/protocol.json').read_text())
    for d in ('voc21','coco_object81'):
        q=complete['datasets'][d]
        p['datasets'][d]=dict(loader=dict(dataset=d,data_root=q['data_root'],sample_seed=20260923,vdd_ontology='official'),
                              screen_images=64)
    for d,r in p['datasets'].items():
        config=json.loads((WORKSPACE/'research/query_slot_best_alias_20261013/configs'/(d+'.yaml')).read_text())
        r['profile']=config['profile']
        r['total_images']=config['total_images']
        r['config_path']=root_remote+'/configs/'+d+'.yaml'
        r['text_cache']=root_remote+'/text_cache/'+d+'.pt'
        r['gain']=float(r['profile'].get('coupling',.5))
        r['profile']['strength']='original'
        # The evaluator resolves the fixed cost set from the dataset record;
        # keep the root mapping too because it is useful provenance.
        r['cost_keys']=cost_keys.get(d,[])
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'optimization_protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    c=retry()
    try:
        _,o,e=c.exec_command('mkdir -p '+shlex.quote(REMOTE_ROOT+'/source'))
        if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
        with c.open_sftp() as s:
            s.put(str(LOCAL/'optimization_protocol.json'),REMOTE_ROOT+'/optimization_protocol.json')
            # Upload the opt-in module and evaluator into the remote checkout
            # used by PYTHONPATH.  The experiment directory stores only the
            # protocol and outputs, so it stays resumable and self-contained.
            for rel in ('dinotool/frozen_deployment.py','dinotool/geov2_wide_optimized.py','scripts/eval_geov2_wide_optimization.py'):
                s.put(str(WORKSPACE/'DINOtool'/rel),REMOTE+'/'+rel)
    finally:c.close()
    print('Prepared',LOCAL)


def launch(variant,mode='screen'):
    if variant not in VARIANTS:raise ValueError(variant)
    c=retry()
    try:
        _,o,e=c.exec_command('mkdir -p '+shlex.quote(REMOTE_ROOT+'/'+mode+'/'+variant))
        if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    finally:c.close()
    jobs=[]
    for gpu,d in zip(GPUS,DATASETS):
        out=REMOTE_ROOT+'/'+mode+'/'+variant+'/'+d
        args=[PYTHON,'-u',REMOTE+'/scripts/eval_geov2_wide_optimization.py','--experiment',REMOTE_ROOT,
              '--dataset',d,'--variant',variant,'--mode',mode,'--num-shards','1','--shard-index','0']
        env='cd '+shlex.quote(REMOTE)+' && export CUDA_VISIBLE_DEVICES='+str(gpu)+' && export PYTHONPATH='+shlex.quote(REMOTE+':'+REMOTE+'/scripts:'+THIRD)+' && '
        command=env+shlex.join(args)+' > '+shlex.quote(out+'.log')+' 2>&1'
        session='g2opt15_'+mode+'_'+variant+'_'+d
        c=retry()
        try:
            _,o,e=c.exec_command('tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote(command))
            if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
        finally:c.close()
        jobs.append(dict(dataset=d,gpu=gpu,session=session,output=out+'_s0',log=out+'.log'))
    (LOCAL/(mode+'_'+variant+'_jobs.json')).write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs))


def suite(variants, mode='screen', datasets=DATASETS, gpus=GPUS):
    if len(gpus)<len(datasets):
        raise ValueError('Supply a distinct GPU for every dataset in a suite.')
    c=retry()
    jobs={v:[] for v in variants}
    try:
        for gpu,d in zip(gpus,datasets):
            session='g2opt15_suite_'+mode+'_'+d
            commands=[]
            for v in variants:
                out=REMOTE_ROOT+'/'+mode+'/'+v+'/'+d
                args=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={REMOTE}:{REMOTE}/scripts:{THIRD}',
                      PYTHON,'-u',REMOTE+'/scripts/eval_geov2_wide_optimization.py',
                      '--experiment',REMOTE_ROOT,'--dataset',d,'--variant',v,'--mode',mode]
                command='mkdir -p '+shlex.quote(REMOTE_ROOT+'/'+mode+'/'+v)+' && '+shlex.join(args)+' > '+shlex.quote(out+'.log')+' 2>&1'
                commands.append(command)
                jobs[v].append(dict(dataset=d,gpu=gpu,session=session,output=out+'_s0',log=out+'.log'))
            shell='cd '+shlex.quote(REMOTE)+' && '+'; '.join(commands)
            _,o,e=c.exec_command('tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote(shell))
            if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    finally:c.close()
    for v,rows in jobs.items():
        path=LOCAL/(mode+'_'+v+'_jobs.json')
        old=json.loads(path.read_text()) if path.exists() else []
        rows=[j for j in old if j['dataset'] not in datasets]+rows
        path.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(mode=mode,variants=variants,datasets=datasets,serialized_on_each_gpu=True)))


def full_pipeline():
    # Seven independent GPUs; two COCO shards bound the long pole. Each cost
    # comparison stays on one GPU, with a fresh process for every arm.
    plans=[(1,'vdd',0,1),(2,'potsdam',0,1),(3,'ade150',0,1),
           (4,'voc20',0,1),(5,'voc21',0,1),
           (6,'coco_object81',0,2),(7,'coco_object81',1,2)]
    jobmap={}
    c=retry()
    try:
        for gpu,d,shard,parts in plans:
            session=f'g2opt15_final_{d}_s{shard}'
            commands=[]
            if gpu==7:
                commands.append('while tmux has-session -t g2opt15_suite_cost_vdd 2>/dev/null; do sleep 2; done')
            steps=[]
            if gpu==7:
                steps.append(('cost','shared_slots48','vdd',0,1))
            for v in ('baseline','shared_slots48'):
                if d!='vdd' and shard==0:steps.append(('cost',v,d,0,1))
                steps.append(('full',v,d,shard,parts))
            for mode,v,dataset,s,n in steps:
                out=REMOTE_ROOT+'/'+mode+'/'+v+'/'+dataset+f'_s{s}'
                log=out+'.log'
                args=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={REMOTE}:{REMOTE}/scripts:{THIRD}',
                      PYTHON,'-u',REMOTE+'/scripts/eval_geov2_wide_optimization.py',
                      '--experiment',REMOTE_ROOT,'--dataset',dataset,'--variant',v,'--mode',mode,
                      '--num-shards',str(n),'--shard-index',str(s)]
                commands.append('mkdir -p '+shlex.quote(REMOTE_ROOT+'/'+mode+'/'+v)+' && '+shlex.join(args)+' > '+shlex.quote(log)+' 2>&1')
                jobmap.setdefault((mode,v),[]).append(dict(dataset=dataset,gpu=gpu,session=session,
                    output=out,log=log,shard=s,num_shards=n))
            shell='cd '+shlex.quote(REMOTE)+' && '+'; '.join(commands)
            _,o,e=c.exec_command('tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote(shell))
            if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    finally:c.close()
    for (mode,v),rows in jobmap.items():
        path=LOCAL/(mode+'_'+v+'_jobs.json')
        old=json.loads(path.read_text()) if path.exists() else []
        keys={(r['dataset'],r.get('shard',0)) for r in rows}
        rows=[r for r in old if (r['dataset'],r.get('shard',0)) not in keys]+rows
        path.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(variants=['baseline','shared_slots48'],full_datasets=list(DATASETS)+['voc21','coco_object81'],
                          coco_shards=2,cost_images=7,cost_warmups=2,cost_repeats=7)))


def optimized_pipeline():
    plans=[(1,'vdd',0,1),(2,'potsdam',0,1),(3,'ade150',0,1),
           (4,'voc20',0,1),(5,'voc21',0,1),(6,'coco_object81',0,2),(7,'coco_object81',1,2)]
    rows=[]; sessions=[];cost_jobs={v:[] for v in ('baseline','optimized48')}
    env=lambda gpu:['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={REMOTE}:{REMOTE}/scripts:{THIRD}',PYTHON,'-u',REMOTE+'/scripts/eval_geov2_wide_optimization.py','--experiment',REMOTE_ROOT,'--cpu-threads','1']
    c=retry()
    try:
        for gpu,d,shard,parts in plans:
            session=f'g2opt15_optimized_{d}_s{shard}';sessions.append(session)
            old=f'g2opt15_final_{d}_s{shard}';sessions.append(old)
            out=REMOTE_ROOT+'/full/optimized48/'+d+f'_s{shard}'
            args=env(gpu)+['--dataset',d,'--variant','optimized48','--mode','full','--num-shards',str(parts),'--shard-index',str(shard)]
            commands=[f'while tmux has-session -t {old} 2>/dev/null; do sleep 5; done',
                      'mkdir -p '+shlex.quote(REMOTE_ROOT+'/full/optimized48'),
                      shlex.join(args)+' > '+shlex.quote(out+'.log')+' 2>&1']
            if d=='vdd':
                # A declared coupling control: use the already known VOC21
                # step g0=1, with exactly the same optimized visual model.
                args=env(gpu)+['--dataset','voc21','--variant','optimized48','--mode','full','--gain','1',
                               '--output-root',REMOTE_ROOT+'/gain1']
                commands.append(shlex.join(args)+' > '+shlex.quote(REMOTE_ROOT+'/gain1_voc21.log')+' 2>&1')
            shell='cd '+shlex.quote(REMOTE)+' && '+'; '.join(commands)
            _,o,e=c.exec_command('tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote(shell))
            if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
            rows.append(dict(dataset=d,gpu=gpu,session=session,output=out,log=out+'.log',shard=shard,num_shards=parts))
        commands=[]
        for session in sessions:
            commands.append(f'while tmux has-session -t {session} 2>/dev/null; do sleep 10; done')
        # Benchmark only once all full evaluations have released their CPU
        # pools. All arms then use one CPU thread and independent GPUs.
        for gpu,d in zip(GPUS,DATASETS+('voc21','coco_object81')):
            session='g2opt15_stable_'+d
            calls=[]
            for v in ('baseline','optimized48'):
                out=REMOTE_ROOT+'/stable/cost/'+v+'/'+d+'_s0'
                args=env(gpu)+['--dataset',d,'--variant',v,'--mode','cost','--output-root',REMOTE_ROOT+'/stable']
                calls.append(shlex.join(args)+' > '+shlex.quote(REMOTE_ROOT+'/stable_'+v+'_'+d+'.log')+' 2>&1')
                cost_jobs[v].append(dict(dataset=d,gpu=gpu,session=session,output=out,log=REMOTE_ROOT+'/stable_'+v+'_'+d+'.log'))
            shell='cd '+shlex.quote(REMOTE)+' && '+'; '.join(calls)
            commands.append('tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote(shell))
        _,o,e=c.exec_command('tmux new-session -d -s g2opt15_stable_coordinator '+shlex.quote('; '.join(commands)))
        if o.channel.recv_exit_status():raise RuntimeError(e.read().decode())
    finally:c.close()
    (LOCAL/'full_optimized48_jobs.json').write_text(json.dumps(rows,indent=2)+'\n')
    for v,values in cost_jobs.items():
        (LOCAL/('stable_cost_'+v+'_jobs.json')).write_text(json.dumps(values,indent=2)+'\n')
    print('Queued final full model, VOC21 gain=1 control, and uncontended paired costs.')


def collect(variant,mode):
    jobs=json.loads((LOCAL/(mode+'_'+variant+'_jobs.json')).read_text())
    c=retry()
    try:
        with c.open_sftp() as s:
            for j in jobs:
                remote=j['output'] if j['output'].rsplit('_s',1)[-1].isdigit() else j['output']+'_s0'
                target=LOCAL/mode/variant/Path(remote).name
                target.mkdir(parents=True,exist_ok=True)
                for name in ('results.json','statistics.npz','progress.json'):
                    try:s.get(remote+'/'+name,str(target/name))
                    except FileNotFoundError:pass
    finally:c.close()
    print('Collected',mode,variant)


def status(variant,mode):
    jobs=json.loads((LOCAL/(mode+'_'+variant+'_jobs.json')).read_text())
    rows=[]
    c=retry()
    try:
        for j in jobs:
            c.exec_command('true')
            result_path=j['output'] if j['output'].rsplit('_s',1)[-1].isdigit() else j['output']+'_s0'
            _,o,e=c.exec_command('python3 -c '+shlex.quote(
                'import json,pathlib,subprocess; p=pathlib.Path('+repr(result_path)+')/"results.json"; r=json.loads(p.read_text()) if p.exists() else {}; a=subprocess.run(["tmux","has-session","-t",'+repr(j['session'])+'],capture_output=True).returncode==0; print(json.dumps(dict(dataset='+repr(j['dataset'])+',status=r.get("status","starting" if a else "failed"),processed=r.get("processed_images",0),active=a,miou=r.get("metrics",{}).get("mean_iou_percent"),median_ms=r.get("median_ms"),peak_allocated_mib=r.get("peak_allocated_mib"),invariants=r.get("invariants",{}))))'))
            rows.append(o.read().decode())
    finally:c.close()
    print(''.join(rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('prepare','launch','suite','pipeline','optimized-pipeline','status','collect'))
    p.add_argument('--variant',choices=VARIANTS,default='baseline')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['baseline','shared_canvas','dual_direct','shared_direct'])
    p.add_argument('--mode',choices=('smoke','screen','full','cost','stable_cost'),default='screen')
    p.add_argument('--datasets',nargs='+',default=list(DATASETS))
    p.add_argument('--gpus',nargs='+',type=int,default=list(GPUS))
    a=p.parse_args()
    if a.action=='optimized-pipeline':optimized_pipeline()
    elif a.action=='pipeline':full_pipeline()
    elif a.action=='prepare':prepare()
    elif a.action=='launch':launch(a.variant,a.mode)
    elif a.action=='suite':suite(a.variants,a.mode,a.datasets,a.gpus)
    elif a.action=='collect':collect(a.variant,a.mode)
    else:status(a.variant,a.mode)


if __name__=='__main__':main()
