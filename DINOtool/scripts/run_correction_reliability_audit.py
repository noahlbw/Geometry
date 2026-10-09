"""Five fixed diagnostic cohorts, using idle GPUs without inference changes."""
import argparse
from pathlib import Path
import time
import run_evidence_envelope as launcher

q=launcher.queue
launcher.EVALUATOR='scripts/eval_correction_reliability_audit.py'
launcher.PREFIX='gcra07'


def main(root):
    root.resolve().relative_to((q.TOOL/'results').resolve())
    if (root/'suite_status.json').exists():raise RuntimeError('Existing suite; do not restart.')
    entries=q.read_json(root/'protocol.json')['datasets']
    pending=list(q.ORDER);active={};completed={};failures={}
    while pending or active:
        for gpu,job in list(active.items()):
            if q.alive(job['session']):continue
            try:
                row=q.read_json(Path(job['output'])/'results.json')
                entry=entries[job['dataset']]
                if not row or row['status']!='complete' or row['processed_images']!=8 or row['total_images']!=8:
                    raise RuntimeError('Incomplete eight-image audit.')
                if row['sample_keys']!=entry['audit_keys'] or len(set(row['sample_keys']))!=8:
                    raise RuntimeError('Frozen cohort mismatch.')
                if not all(row[k] for k in ('weights_frozen','head_weights_unchanged','labels_used_for_audit_only','labels_not_used_to_select_samples','factory_direct_probability_witness')):
                    raise RuntimeError('Audit invariant failed.')
                prior=q.read_json(Path(entry['audit_reference']))['signature']
                if row['checkpoints']!=prior['checkpoints'] or row['classes']!=prior['classes'][job['dataset']]:
                    raise RuntimeError('Checkpoint/class mismatch.')
                completed[job['dataset']]=job['output']
            except Exception as e:
                failures[job['session']]=dict(error=str(e),log_tail=Path(job['log']).read_text(errors='replace')[-6000:])
            del active[gpu]
        if not failures:
            for gpu in range(8):
                if pending and gpu not in active and q.idle(gpu):
                    d=pending.pop(0);active[gpu]=launcher.launch(root,d,entries[d],gpu,'audit')
        else:pending=[]
        state=dict(status='running',active=list(active.values()),pending=pending,completed=completed,failures=failures)
        q.save(root/'suite_status.json',state)
        if pending or active:time.sleep(5)
    state['status']='failed' if failures else 'complete'
    q.save(root/'suite_status.json',state);q.save(root/'suite_results.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
