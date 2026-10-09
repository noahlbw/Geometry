"""Complete the corrected natural VIP cost protocol after the finite main queue."""
import argparse
from pathlib import Path
import time
from eval_rival_fine_full import save
from run_alias_finalization import launch,verify_job,idle,alive,read_json


def main(root,prerequisite='verification_results.json',state_prefix='natural_vip_cost'):
    state_path=root/(state_prefix+'_status.json')
    if state_path.exists():raise RuntimeError('Existing cost correction; no duplicate.')
    # The main queue owns GPUs while active. This is a finite subsequent stage,
    # never a second competing GPU scheduler on the same pool.
    while True:
        terminal=read_json(root/prerequisite)
        if terminal:
            if terminal['status']!='complete':raise RuntimeError('Main queue failed; preserve it and inspect.')
            break
        save(state_path,dict(status='waiting_for_main_queue',active=[],pending=[]))
        time.sleep(15)
    protocol=read_json(root/'protocol.json')
    pending=[d for d in protocol['order'] if protocol['datasets'][d]['family']=='natural']
    active={};completed=[];failures={}
    while active or pending:
        for gpu,job in list(active.items()):
            if alive(job['session']):continue
            try:verify_job(job);completed.append(job['dataset'])
            except Exception as e:failures[job['dataset']]=dict(error=str(e),log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
            del active[gpu]
        if failures:pending.clear()
        for d in list(pending):
            gpu=next((g for g in range(8) if g not in active and idle(g)),None)
            if gpu is None:break
            active[gpu]=launch(root,d,protocol['datasets'][d],gpu,'cost',mode='VIPOfficialProtocol')
            pending.remove(d)
        save(state_path,dict(status='running' if active or pending else 'failed' if failures else 'complete',
            active=list(active.values()),pending=pending,completed=completed,failures=failures,
            repair='Only natural VIP timing preprocessing; candidate evaluations/words/settings and old cost outputs preserved.'))
        if active or pending:time.sleep(15)
    save(root/(state_prefix+'_results.json'),dict(status='failed' if failures else 'complete',completed=completed,failures=failures))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prerequisite',default='verification_results.json')
    p.add_argument('--state-prefix',default='natural_vip_cost')
    args=p.parse_args()
    main(args.root,args.prerequisite,args.state_prefix)
