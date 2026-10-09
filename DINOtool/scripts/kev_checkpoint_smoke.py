"""Load a pinned, frozen text judge; no segmentation or language-model training."""
import argparse
import json
from pathlib import Path
import time

import torch
from kev.checkpoint import Checkpoint, LoadOptions


def main(output):
    started = time.perf_counter()
    checkpoint = Checkpoint('jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52')
    tokenizer, model = checkpoint.load('cuda', LoadOptions(
        dtype=torch.bfloat16, merge=True, fused=False, cuda_graphs=False))
    model.eval().requires_grad_(False)
    record = {'state': 'Semantic segmentation labels: wall, roof, road, vehicle, vegetation, water. '
                        'Wall means an exterior vertical building surface; roof is the top covering.',
              'questions': [{'instr': 'For the target label wall, classify the alias rooftop.',
                             'options': ['Specific to wall', 'Shared or ambiguous', 'Belongs to another label', 'Insufficient information'],
                             'label': 0}]}
    with torch.inference_mode():
        encoded = model.encode(tokenizer, record, max_state=4096, max_branch=1024, strict=True)
        result = model.probs(encoded)
    probabilities = [p.detach().float().cpu().tolist() if torch.is_tensor(p) else p for p in result]
    row = dict(status='complete', checkpoint=checkpoint.ref if hasattr(checkpoint, 'ref') else str(checkpoint),
               base=checkpoint.meta.base, base_revision=checkpoint.meta.base_revision,
               temperature=checkpoint.meta.temperature, probabilities=probabilities,
               wall_seconds=time.perf_counter()-started,
               peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1048576,
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
