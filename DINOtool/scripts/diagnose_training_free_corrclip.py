"""Image-only CorrCLIP decoder compatibility and native-image replay checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from eval_training_free_baseline import load_model, load_rgb, predict, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepared-repair',action='store_true')
    parser.add_argument('--wait-idle-gpu',type=int)
    args=parser.parse_args()
    root=Path(args.root);output=Path(args.output)
    if output.exists():raise RuntimeError('Existing diagnosis preserved')
    if args.wait_idle_gpu is not None:
        from run_training_free_baselines import idle
        while not idle(args.wait_idle_gpu):time.sleep(10)
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    suite=json.loads((root/'protocol.json').read_text())
    manifests=json.loads((root/'samples.json').read_text())
    outcomes=[]
    for dataset in ('vdd','flair1'):
        config=suite['methods']['CorrCLIP']['configs'][dataset]
        model=load_model(config,root,allow_prepared_repair=args.prepared_repair)
        row=manifests[dataset]['samples'][0]
        try:
            prediction=predict(model,load_rgb(row,dataset),row,config)
            outcome=dict(dataset=dataset,key=row['key'],status='finite',
                prediction_sha256=hashlib.sha256(prediction.cpu().numpy().tobytes()).hexdigest())
        except Exception as error:
            outcome=dict(dataset=dataset,key=row['key'],status='failed',error=repr(error))
        outcomes.append(outcome)
        print(json.dumps(outcome),flush=True)
        del model
        torch.cuda.empty_cache()
    save(output,dict(target_masks_loaded=False,outcomes=outcomes))


if __name__=='__main__':main()
