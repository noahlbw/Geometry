"""Image-only trace of the observed PTC failure; never loads target masks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from eval_training_free_baseline import load_model, load_rgb, predict, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepared-repair', action='store_true')
    parser.add_argument('--wait-idle-gpu', type=int)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError('Preserve existing diagnosis: '+str(output))
    if args.wait_idle_gpu is not None:
        from run_training_free_baselines import idle
        while not idle(args.wait_idle_gpu):
            time.sleep(10)
    torch.set_num_threads(4)
    suite = json.loads((root/'protocol.json').read_text())
    manifest = json.loads((root/'samples.json').read_text())['loveda']
    config = suite['methods']['PTC']['configs']['loveda']
    model = load_model(config, root, allow_prepared_repair=args.prepared_repair)
    events = []
    original = F.softmax

    def traced_softmax(value, *a, **kw):
        result = original(value, *a, **kw)
        if not torch.isfinite(result).all():
            dim = kw.get('dim', a[0] if a else -1)
            empty = torch.isneginf(value).all(dim=dim)
            event = dict(operator='softmax', shape=list(value.shape), dtype=str(value.dtype),
                all_negative_infinity_rows=int(empty.sum()), input_nan=int(torch.isnan(value).sum()),
                input_positive_infinity=int(torch.isposinf(value).sum()),
                output_nan=int(torch.isnan(result).sum()))
            events.append(event)
            print(json.dumps(event), flush=True)
        return result

    F.softmax = traced_softmax
    rows = {row['key']: row for row in manifest['samples']}
    outcomes = []
    for key in (manifest['samples'][0]['key'], '3868.png', '3785.png', '3479.png'):
        row = rows[key]
        before = len(events)
        try:
            prediction = predict(model, load_rgb(row, 'loveda'), row, config)
            outcome = dict(key=key, status='finite',
                prediction_sha256=hashlib.sha256(prediction.cpu().numpy().tobytes()).hexdigest())
        except Exception as error:
            outcome = dict(key=key, status='failed', error=repr(error))
        outcome['events'] = events[before:]
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    if args.prepared_repair:
        config = suite['methods']['PTC']['configs']['udd5']
        row = json.loads((root/'samples.json').read_text())['udd5']['samples'][0]
        model = load_model(config, root, allow_prepared_repair=True)
        predict(model, load_rgb(row, 'udd5'), row, config)
        outcomes.append(dict(key='udd5/'+row['key'], status='finite', rectangular_grid_verified=True))
    save(output, dict(target_masks_loaded=False, outcomes=outcomes))


if __name__ == '__main__':
    main()
