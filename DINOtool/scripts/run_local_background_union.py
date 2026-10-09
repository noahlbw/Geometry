"""Matched control suite, exact baseline and unchanged arm verification."""
import argparse
from pathlib import Path
import numpy as np
import run_evidence_envelope as runner
import run_semantic_mass_wide as reference
from eval_local_background_union import METHODS


def verify_dataset(root,d,entry):
    path=reference.verify_dataset(root,d,entry)
    row=runner.queue.read_json(Path(path))
    applies=entry['family']=='natural' and entry['background_index'] is not None and entry.get('residual_identity') is None and len(entry['banks']['semantic_segmentation']['classes'][entry['background_index']]['synonyms'])>1
    if not applies and not np.array_equal(row['metrics'][d]['Frozen']['confusion_matrix'],row['metrics'][d]['LocalBackgroundUnion']['confusion_matrix']):
        raise RuntimeError('Unchanged full control differs.')
    row.update(local_background_union_applied=applies,unchanged_control_confusion_replay=not applies)
    runner.queue.save(Path(path),row)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    runner.EVALUATOR='scripts/eval_local_background_union.py';runner.PREFIX='glbu07';runner.METHODS=METHODS
    reference.METHODS=METHODS;runner.queue.METHODS=METHODS;runner.queue.launch=runner.launch
    runner.queue.verify_job=runner.verify_job;runner.queue.verify_dataset=verify_dataset
    runner.queue.main(p.parse_args().root)
