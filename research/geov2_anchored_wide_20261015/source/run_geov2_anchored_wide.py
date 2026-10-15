"""Freeze, launch, monitor and collect the matched anchored-wide experiment."""
import argparse
import json
from pathlib import Path
import shlex
import stat

from a800_region_experiment import WORKSPACE, REMOTE, PYTHON, connect
from pointwise_context_experiment import THIRD


NAME = 'geov2_anchored_wide_20261015'
LOCAL = WORKSPACE/'research'/NAME
ROOT = REMOTE+'/results/'+NAME
DATASETS = ('vdd', 'potsdam', 'udd5', 'oem', 'vaihingen', 'ade150', 'voc21')
GAINS = (.25, .5, .75, 1.)
PRIMARY = ('Coordinate', 'Legacy', 'Anchored')
CONTROLS = ('NoPrior', 'NoReference', 'AbsoluteMix', 'Shuffled')
SOURCES = ('eval_geov2_anchored_wide.py', 'eval_geov2_multidataset_components.py',
    'eval_geov2_fixed05_components.py', 'eval_geov2_g0_sweep.py')


def cells(gain):
    rows = [dict(key=method+'__g'+format(g, 'g'), method=method, g0=g, alpha=None)
        for method in PRIMARY for g in GAINS]
    rows += [dict(key=method+'__g'+format(gain, 'g'), method=method, g0=gain, alpha=None)
        for method in CONTROLS]
    rows += [dict(key='UniformBlend__a'+format(a, 'g'), method='UniformBlend', g0=None, alpha=a)
        for a in (0., .125, .25, .375, .5, .75, 1.)]
    return rows


def command(client, value):
    _, out, err = client.exec_command(value)
    result = out.read().decode()
    error = err.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError(error or result)
    return result


def prepare(client):
    previous = WORKSPACE/'research/geov2_multidataset_ablation_20261015'
    old = json.loads((previous/'protocol.json').read_text())
    expected = json.loads((previous/'expected_keys.json').read_text())
    p = {k:old[k] for k in ('checkpoint_dir', 'dinov3_repo', 'upstream_root')}
    p.update(status='frozen_before_new_evaluation', architecture='Geov2 anchored residual wide / matched shared_canvas',
        slots=24, beta=.5, query_temperature=.1, iterations=5, gains=GAINS,
        primary_arms=PRIMARY, secondary_arms=CONTROLS, datasets={},
        prediction_target_access=False, bootstrap_repeats=2000, bootstrap_seed=20261015,
        grouping='Potsdam/Vaihingen original aerial scenes; OEM region path prefix; others images.',
        selection='All seven complete validation sets are declared before evaluation. Screens are integration/development diagnostics, not an architecture search. One proposed candidate; report negative outcomes. Three primary arms have the same four g0 choices; anchored-observation ordinary fusion has seven alpha choices.',
        promotion=dict(macro_gain='Positive paired grouped 95% lower bounds versus both Coordinate and Legacy at frozen reference gains',
            voc21_noninferiority_margin_pp=-.2, remote_sensing='Nonnegative point gain over Coordinate on both Potsdam and Vaihingen',
            latency_ratio_max=1.05, peak_allocated_increment_mib_max=50),
        limitations='Previously inspected validation; historical aliases/calibration used development labels. Full-set scalar selection is development analysis. Group bootstrap is conditional on selected parameters; no untouched-test claim. P spatial mapping matches incumbent sampler, including its padding behavior. Same head/scoring/normalization for all controls. Only local raw query is shuffled by a half-sequence roll, not a multiseed random control. No new adaptive gate.',
        formulas=dict(e='(normalize(q_local)-normalize(sample(normalize(raw_global)))) @ centers.T / 0.1',
            posterior='softmax(log(normalize(sample(P))) + e)',
            feature='normalize(sample(semantic) + 0.5*(posterior-prior)@values)',
            writer='L+g0*(I+A.T@A)^(-1)@A.T@A@(W-L)'))
    for d in DATASETS:
        r = old['datasets'][d]
        p['datasets'][d] = {k:r[k] for k in ('config_path', 'text_cache', 'data_root', 'total_images', 'reference_g0', 'cost_keys')}
        n = min(len(expected[d]), 128 if d in ('ade150', 'voc21') else 64)
        indices = [i*(len(expected[d])-1)//(n-1) for i in range(n)] if n>1 else [0]
        p['datasets'][d].update(cells=cells(r['reference_g0']), screen_keys=[expected[d][i] for i in indices])
    LOCAL.mkdir(parents=True, exist_ok=False)
    (LOCAL/'source').mkdir()
    (LOCAL/'protocol.json').write_text(json.dumps(p, indent=2)+'\n', encoding='utf-8')
    (LOCAL/'expected_keys.json').write_text(json.dumps(expected, indent=2)+'\n', encoding='utf-8')
    command(client, PYTHON+' -c '+shlex.quote('from pathlib import Path; p=Path('+repr(ROOT)+'); p.mkdir(); (p/"source").mkdir()'))
    deploy(client)
    with client.open_sftp() as sftp:
        for name in ('protocol.json', 'expected_keys.json'):
            sftp.put(str(LOCAL/name), ROOT+'/'+name)
    print('Prepared', LOCAL)


def deploy(client):
    with client.open_sftp() as sftp:
        for name in SOURCES:
            local = WORKSPACE/'DINOtool/scripts'/name
            sftp.put(str(local), ROOT+'/source/'+name)
            (LOCAL/'source'/name).write_bytes(local.read_bytes())
        local = WORKSPACE/'DINOtool/dinotool/anchored_slot_memory.py'
        sftp.put(str(local), REMOTE+'/dinotool/anchored_slot_memory.py')
        sftp.put(str(local), ROOT+'/source/anchored_slot_memory.py')
        (LOCAL/'source/anchored_slot_memory.py').write_bytes(local.read_bytes())
        test = WORKSPACE/'DINOtool/tests/test_anchored_slot_memory.py'
        if test.exists():
            sftp.put(str(test), ROOT+'/source/test_anchored_slot_memory.py')
            (LOCAL/'source/test_anchored_slot_memory.py').write_bytes(test.read_bytes())
    print('Deployed new experiment sources only')


def free_gpus(client, chosen):
    rows = command(client, 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits')
    for row in rows.splitlines():
        gpu, memory, use = [int(x.strip()) for x in row.split(',')]
        if gpu in chosen and (memory>256 or use>5):
            raise RuntimeError('GPU occupied: '+row)


def launch(client, mode):
    free_gpus(client, range(1,8))
    if mode in ('smoke', 'screen'):
        plans = {i+1:[(d,0,1)] for i,d in enumerate(DATASETS)}
    else:
        plans = {1:[('vdd',0,1), ('oem',0,1)],
            2:[('potsdam',0,1), ('vaihingen',0,1), ('udd5',0,1)],
            3:[('ade150',0,3)],4:[('ade150',1,3)],5:[('ade150',2,3)],
            6:[('voc21',0,2)],7:[('voc21',1,2)]}
    jobs=[]
    for gpu,steps in plans.items():
        script='set -e\ncd '+shlex.quote(REMOTE)+'\n'
        script+=f'export CUDA_VISIBLE_DEVICES={gpu}\nexport OMP_NUM_THREADS=1\nexport OPENBLAS_NUM_THREADS=1\n'
        script+='export PYTHONPATH='+shlex.quote(ROOT+'/source:'+REMOTE+':'+REMOTE+'/scripts:'+THIRD)+'\n'
        for d,shard,parts in steps:
            args=[PYTHON,'-u',ROOT+'/source/eval_geov2_anchored_wide.py','--experiment',ROOT,
                '--dataset',d,'--mode',mode,'--shard-index',str(shard),'--num-shards',str(parts)]
            log=ROOT+f'/{mode}_{d}_s{shard}.log'
            script+=shlex.join(args)+' > '+shlex.quote(log)+' 2>&1\n'
            jobs.append(dict(dataset=d,shard=shard,shards=parts,gpu=gpu,log=log))
        launch_script(client, mode, gpu, script)
    (LOCAL/(mode+'_jobs.json')).write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs))


def launch_script(client, mode, gpu, script):
    file=ROOT+f'/{mode}_gpu{gpu}.sh'
    with client.open_sftp() as sftp:
        with sftp.file(file,'w') as stream: stream.write(script)
    shell='bash '+shlex.quote(file)+'; result=$?; echo "$result" > '+shlex.quote(ROOT+f'/{mode}_gpu{gpu}.exit')+'; exit "$result"'
    command(client,'tmux new-session -d -s '+shlex.quote(f'g2anchor15_{mode}_{gpu}')+' '+shlex.quote(shell))


def cost(client):
    # Other authorized work may occupy a device after our precision job ends.
    gpu_rows=command(client,'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits')
    available=[]
    for row in gpu_rows.splitlines():
        gpu,memory,use=[int(x.strip()) for x in row.split(',')]
        if gpu in range(1,8) and memory<=256 and use<=5:
            available.append(gpu)
    if not available:
        raise RuntimeError('No idle GPU is available for independent costs.')
    p=json.loads((LOCAL/'protocol.json').read_text())
    jobs=[]
    plans={gpu:[] for gpu in available}
    for i,d in enumerate(DATASETS): plans[available[i%len(available)]].append(d)
    for gpu,datasets in plans.items():
        if not datasets: continue
        script='set -e\ncd '+shlex.quote(REMOTE)+'\n'
        script+=f'export CUDA_VISIBLE_DEVICES={gpu}\nexport OMP_NUM_THREADS=1\nexport OPENBLAS_NUM_THREADS=1\n'
        script+='export PYTHONPATH='+shlex.quote(ROOT+'/source:'+REMOTE+':'+REMOTE+'/scripts:'+THIRD)+'\n'
        for d in datasets:
            for method in PRIMARY:
                cell=method+'__g'+format(p['datasets'][d]['reference_g0'],'g')
                args=[PYTHON,'-u',ROOT+'/source/eval_geov2_anchored_wide.py','--experiment',ROOT,
                    '--dataset',d,'--mode','cost','--cell',cell]
                log=ROOT+f'/cost_{d}_{method}.log'
                script+=shlex.join(args)+' > '+shlex.quote(log)+' 2>&1\n'
                jobs.append(dict(dataset=d,method=method,gpu=gpu,log=log))
        launch_script(client,'cost',gpu,script)
    (LOCAL/'cost_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Started',len(jobs),'independent cost processes')


def status(client):
    code='from pathlib import Path; import json; r=Path('+repr(ROOT)+'); '
    code+="print('exits',[(p.name,p.read_text().strip()) for p in sorted(r.glob('*.exit'))]); "
    code+="print('progress',[(str(p.relative_to(r)),json.loads(p.read_text())) for p in sorted(r.glob('*/*/s*/progress.json'))]); "
    code+="print('errors',[(p.name,p.read_text()[-4000:]) for p in r.glob('*.log') if 'Traceback' in p.read_text()]); "
    code+="print('costs_complete',len(list(r.glob('cost/*/*/results.json'))))"
    print(command(client,PYTHON+' -c '+shlex.quote(code)))


def collect(client):
    with client.open_sftp() as sftp:
        def walk(src,dest):
            dest.mkdir(parents=True,exist_ok=True)
            for row in sftp.listdir_attr(src):
                if row.filename=='__pycache__': continue
                source,target=src+'/'+row.filename,dest/row.filename
                if stat.S_ISDIR(row.st_mode): walk(source,target)
                elif not target.exists() or target.stat().st_size!=row.st_size:
                    sftp.get(source,str(target),prefetch=False)
        walk(ROOT,LOCAL)
    print('Collected',LOCAL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','deploy','smoke','screen','full','cost','status','collect'),required=True)
    args=parser.parse_args()
    client=connect()
    try:
        if args.stage=='prepare': prepare(client)
        elif args.stage=='deploy': deploy(client)
        elif args.stage in ('smoke','screen','full'): launch(client,args.stage)
        elif args.stage=='cost': cost(client)
        elif args.stage=='status': status(client)
        else: collect(client)
    finally: client.close()
