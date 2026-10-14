"""Collect completed fixed-g0 shards without interrupting A800 workers."""
import argparse
import json
from pathlib import Path

from training_free_baseline_experiment import checked_connect as connect, REMOTE, WORKSPACE


def main(args):
    remote = REMOTE + '/results/geov2_g0_sweep_20261014'
    local = WORKSPACE / args.output
    client = connect()
    try:
        with client.open_sftp() as sftp:
            def get(relative, skip=False):
                path = local/relative
                size = sftp.stat(remote+'/'+relative).st_size
                if skip and path.exists() and path.stat().st_size == size:
                    return
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix+'.part')
                sftp.get(remote+'/'+relative, str(temporary), prefetch=False)
                if temporary.stat().st_size != size:
                    raise RuntimeError('Incomplete transfer: '+relative)
                temporary.replace(path)

            for name in ('protocol.json', 'expected_keys.json', 'state.json'):
                get(name)
            protocol = json.loads((local/'protocol.json').read_text())
            state = json.loads((local/'state.json').read_text())
            for ds in protocol['datasets']:
                get('configs/'+ds+'.yaml', skip=True)
                get('vocabularies/'+ds+'.json', skip=True)
            for name in ('eval_geov2_g0_sweep.py', 'run_geov2_g0_sweep.py', 'eval_residual_ontology.py'):
                get('source/scripts/'+name, skip=True)
            complete = 0
            for key, job in state['jobs'].items():
                folder = 'full/'+key+'/'
                if job['status'] == 'complete':
                    for name in ('results.json', 'statistics.npz'):
                        get(folder+name, skip=True)
                    complete += 1
                elif job['status'] == 'running':
                    try:
                        get(folder+'progress.json')
                    except FileNotFoundError:
                        pass
                elif job['status'] == 'failed':
                    get('full/'+key+'.log')
            print(json.dumps(dict(status=state['status'], completed_shards=complete,
                total_shards=len(state['jobs']),
                progress={key: json.loads((local/'full'/key/'progress.json').read_text())
                    for key,job in state['jobs'].items()
                    if job['status']=='running' and (local/'full'/key/'progress.json').exists()}), indent=2), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='research/geov2_g0_sweep_20261014')
    main(p.parse_args())
