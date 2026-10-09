"""Queue only idle GPUs; preserve completed outputs and stop on exact failures."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
from eval_alias_finalization import METHODS
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_evidence_adaptive_readout import TOOL,BASE,THIRD,PYTHON,OLD,idle,alive,read_json


def launch(root,d,entry,gpu,phase,shard=0,mode=None):
    smoke=phase.startswith('smoke')
    output=root/phase/d if smoke else root/phase/d/('s'+str(shard))
    if phase=='cost':output=root/phase/d/(mode+'.json')
    log=output.with_suffix('.log');session=f'gaf09_{phase}_{d}_'+(mode or 's'+str(shard))
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or GPU occupied: '+session)
    output.parent.mkdir(parents=True,exist_ok=True)
    common=['--dataset',d,'--suite-root',str(root),'--data-root',entry['data_root'],
        '--original-cache',str(OLD/'text_cache'/(d+'.pt')),'--dinov3-repo',str(TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(BASE/'ckpt/DINO'),'--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee')]
    if phase=='cost':
        args=[PYTHON,'-u','scripts/benchmark_alias_finalization.py',*common,'--output',str(output),
              '--cost-mode',mode,'--samples','7','--repeats','7']
    else:
        args=[PYTHON,'-u','scripts/eval_alias_finalization.py',*common,'--mode','smoke' if smoke else phase,'--output-dir',str(output),
              '--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,phase=phase,shard=shard,mode=mode,output=str(output),log=str(log),session=session,gpu=gpu)


def verify_job(job):
    row=read_json(Path(job['output']) if job['phase']=='cost' else Path(job['output'])/'results.json')
    if not row or row['status']!='complete':raise RuntimeError('Worker did not complete: '+job['session'])
    if job['phase']=='cost':
        if not row['standalone_process'] or row['target_masks_loaded']:raise RuntimeError('Non-isolated cost/mask access.')
        return
    if row['processed_images']!=row['total_images'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
        raise RuntimeError('Incomplete or mutated model.')
    if job['phase'].startswith('smoke'):
        if row['target_masks_loaded'] or not row['scalar_agreement_verified']:raise RuntimeError('Smoke failed.')
        return
    with np.load(Path(job['output'])/'per_image_confusions.npz',allow_pickle=False) as data:
        if data['sample_keys'].tolist()!=row['signature']['sample_keys']:raise RuntimeError('Per-image key mismatch.')
        for p,group in row['metrics'].items():
            targets=None
            for m in METHODS:
                cm=data[p+'__'+m].sum(0)
                if not np.array_equal(cm,group[m]['confusion_matrix']):raise RuntimeError('Per-image sum mismatch.')
                if targets is not None and not np.array_equal(targets,cm.sum(1)):raise RuntimeError('Different scored target counts.')
                targets=cm.sum(1)


def verify_dataset(root,d,entry,preserve_existing=False,reference_override=None):
    inputs=[root/'full'/d/('s'+str(i)) for i in range(entry['shards'])]
    target=root/'full'/d/'merged.json'
    reuse_verified=False
    if target.exists():
        if not preserve_existing:raise RuntimeError('Existing merge; preserve it.')
        row=read_json(target)
        original=str(target)
        target=target.with_name('verified_merged.json')
        if target.exists():
            row=read_json(target);reuse_verified=True
        else:row['original_merge_preserved']=original
    else:row=merge([str(f) for f in inputs],str(target))
    if not row['coverage_verified'] or row['processed_images']!=entry['total_images']:raise RuntimeError('Incomplete global coverage.')
    checks=[]
    for folder in inputs:checks.extend(read_json(folder/'deployment_checks.json')['samples'])
    if len(checks)!=entry['total_images'] or len({r['key'] for r in checks})!=len(checks):raise RuntimeError('Missing deployment checks.')
    historical={}
    if entry['family']=='remote_sensing':
        prior=read_json(OLD/'full'/d/'merged.json')
        if row['signature']['global_sample_keys_sha256']!=prior['signature']['global_sample_keys_sha256']:
            raise RuntimeError('Historical RS sample sequence differs.')
        if reference_override is not None:
            if (d!='loveda' or reuse_verified
                    or not reference_override.get('exact_historical_per_image_confusions')
                    or reference_override['protocol']!='P'
                    or reference_override['global_sample_keys_sha256']!=row['signature']['global_sample_keys_sha256']):
                raise RuntimeError('Invalid scoped historical reference override.')
            metric=reference_override['metric']
            if not np.array_equal(metric['confusion_matrix'],prior['metrics']['P__original20']['Geometry_PatchOnly2Coupled']['confusion_matrix']):
                raise RuntimeError('Replacement reference is not the exact historical confusion.')
            original=row['metrics']['P']['Reference20SameViews']
            if not np.array_equal(np.asarray(original['confusion_matrix']).sum(1),np.asarray(metric['confusion_matrix']).sum(1)):
                raise RuntimeError('Replacement reference scored targets differ.')
            row['reference_recovery']=dict(reference_override,raw_reference_metric=original,
                scope='Only P Reference20SameViews uses its independent historical P text cache. All candidate metrics and raw per-image outputs are unchanged.')
            row['metrics']['P']['Reference20SameViews']=metric
        for p,group in row['metrics'].items():
            wanted=prior['metrics'][p+'__original20']['Geometry_PatchOnly2Coupled']['confusion_matrix']
            if not np.array_equal(group['Reference20SameViews']['confusion_matrix'],wanted):
                raise RuntimeError('Historical same-view original20 reference differs: '+d+'/'+p)
            historical[p]=True
    scalar_replay=None
    if 'expected_retained_scalar' in entry:
        # NaturalShortEdge historical output was the packed deployment API, not
        # the independently implemented scalar reduction. The latter is already
        # checked per image with a strict probability/argmax-tie tolerance.
        scalar_replay=bool(np.array_equal(row['metrics'][d]['Retained']['confusion_matrix'],entry['expected_retained_scalar']))
        if not scalar_replay:raise RuntimeError('Frozen deployed selected-scale endpoint differs: '+d)
    for p,g in row['metrics'].items():
        target_counts=np.asarray(g['Retained']['confusion_matrix']).sum(1)
        if any(not np.array_equal(target_counts,np.asarray(g[m]['confusion_matrix']).sum(1)) for m in METHODS):
            raise RuntimeError('Paired target counts differ.')
    row.update(paired_scored_targets_equal=True,scalar_agreement_verified=True,
        exact_historical_original20_replay=historical,exact_selected_deployment_replay=scalar_replay,
        historical_expected_field_note='Frozen manifest field expected_retained_scalar is the historical NaturalShortEdge packed deployment confusion; its spelling is preserved. Independent scalar agreement permits only verified numerical ties.',
        deployment_changed_pixels={m:sum(c['protocols'][p][m]['changed_pixels'] for c in checks for p in c['protocols'])
            for m in ('retained','uniform')})
    if not reuse_verified:save(target,row)
    return str(target)


def main(root,resume=False,resume_verification=False):
    root.resolve().relative_to((TOOL/'results').resolve())
    status_path=root/('verification_status.json' if resume_verification else 'recovery_status.json' if resume else 'suite_status.json')
    result_path=root/('verification_results.json' if resume_verification else 'recovery_results.json' if resume else 'suite_results.json')
    if status_path.exists():raise RuntimeError('Existing controller state; no duplicate resume.')
    protocol=read_json(root/'protocol.json');entries=protocol['datasets']
    active,completed,failures,finished={}, {}, {}, set()
    pending=[dict(dataset=d,phase='smoke',shard=0) for d in protocol['order']]
    if resume:
        prior=read_json(root/'suite_results.json')
        if not prior or prior['status']!='failed':raise RuntimeError('Terminal original failure required.')
        if set(prior['failures'])!={'gaf09_smoke_loveda_s0'}:
            raise RuntimeError('Recovery is scoped only to the recorded LoveDA path-loader failure.')
        pending=[]
        for d in protocol['order']:
            if d=='loveda':
                pending.append(dict(dataset=d,phase='smoke_loader_repair',shard=0));continue
            job=dict(dataset=d,phase='smoke',output=str(root/'smoke'/d),session=f'gaf09_smoke_{d}_s0')
            verify_job(job)
            for mode in protocol['cost']['modes']:
                path=root/'cost'/d/(mode+'.json')
                if path.exists():verify_job(dict(job,phase='cost',output=str(path)))
                else:pending.append(dict(dataset=d,phase='cost',shard=0,mode=mode))
            for i in range(entries[d]['shards']):
                path=root/'full'/d/('s'+str(i))
                if path.exists():raise RuntimeError('Unexpected pre-recovery full output; inspect it.')
                pending.append(dict(dataset=d,phase='full',shard=i))
    if resume_verification:
        prior=read_json(root/'recovery_results.json')
        if not prior or prior['status']!='failed' or not prior['failures']:
            raise RuntimeError('Terminal verifier failure required.')
        if any(not f['error'].startswith('Frozen scalar selected-scale endpoint differs:') for f in prior['failures'].values()):
            raise RuntimeError('Recovery scoped only to historical packed/scalar confusion mixup.')
        pending=[]
        for d in protocol['order']:
            smoke_phase='smoke_loader_repair' if d=='loveda' else 'smoke'
            job=dict(dataset=d,phase=smoke_phase,output=str(root/smoke_phase/d),session=f'gaf09_{smoke_phase}_{d}_s0')
            if d=='loveda' and not (root/smoke_phase/d/'results.json').exists():
                # Earlier queue stopped on VDD before reaching the repaired
                # LoveDA smoke. Queue it once; its success will queue its arms.
                pending.append(dict(dataset=d,phase=smoke_phase,shard=0))
                continue
            verify_job(job)
            for mode in protocol['cost']['modes']:
                path=root/'cost'/d/(mode+'.json')
                if path.exists():verify_job(dict(job,phase='cost',output=str(path)))
                else:pending.append(dict(dataset=d,phase='cost',shard=0,mode=mode))
            for i in range(entries[d]['shards']):
                path=root/'full'/d/('s'+str(i))
                if path.exists():
                    verify_job(dict(job,phase='full',output=str(path)))
                    finished.add((d,i))
                else:pending.append(dict(dataset=d,phase='full',shard=i))
            if all((d,i) in finished for i in range(entries[d]['shards'])):
                completed[d]=verify_dataset(root,d,entries[d],preserve_existing=True)
    started=time.time()
    while active or pending:
        for gpu,job in list(active.items()):
            if alive(job['session']):continue
            d=job['dataset']
            try:
                verify_job(job)
                if job['phase'].startswith('smoke'):
                    pending.extend(dict(dataset=d,phase='cost',shard=0,mode=m) for m in protocol['cost']['modes'])
                    pending.extend(dict(dataset=d,phase='full',shard=i) for i in range(entries[d]['shards']))
                elif job['phase']=='full':
                    finished.add((d,job['shard']))
                    if all((d,i) in finished for i in range(entries[d]['shards'])):
                        completed[d]=verify_dataset(root,d,entries[d])
            except Exception as e:
                failures[job['session']]=dict(error=str(e),log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
            del active[gpu]
        if failures:pending.clear()
        for job in list(pending):
            gpu=next((g for g in range(8) if g not in active and idle(g)),None)
            if gpu is None:break
            try:active[gpu]=launch(root,job['dataset'],entries[job['dataset']],gpu,job['phase'],job['shard'],job.get('mode'))
            except Exception as e:
                failures['launch_'+job['dataset']]=dict(error=str(e));pending.clear();break
            pending.remove(job)
        save(status_path,dict(status='running' if active or pending else 'failed' if failures else 'complete',
            active=list(active.values()),pending=pending,completed=completed,failures=failures))
        if pending or active:time.sleep(15)
    save(result_path,dict(status='failed' if failures else 'complete',completed=completed,
        failures=failures,suite_wall_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-verification',action='store_true')
    args=p.parse_args()
    if args.resume and args.resume_verification:p.error('Choose one recovery only.')
    main(args.root,args.resume,args.resume_verification)
