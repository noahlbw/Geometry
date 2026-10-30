"""Compare frozen Anchored labels with the execution-only infrastructure path."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dinotool.anchored_infra import AnchoredInfraConfig
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.geov2_wide_optimized import OptimizedWideReadout, wide_readout_config
from eval_alias_finalization import load_samples
from eval_geov2_anchored_wide import AnchoredExperiment


@torch.inference_mode()
def main(args):
    root = Path(args.experiment)
    protocol = json.loads((root / 'protocol.json').read_text())
    record = protocol['datasets'][args.dataset]
    p = dict(protocol, **record, dataset=args.dataset)
    # The reference is deliberately the compatibility FP64, non-streaming
    # path. It exercises exactly the Anchored observation and writer equations.
    reference = AnchoredExperiment(p, AnchoredInfraConfig(
        solver_dtype='float64', stream_logits=False, fast_math=False))
    m = reference.model
    loader_args = argparse.Namespace(dataset=args.dataset, data_root=record['data_root'],
        sample_seed=20260923, vdd_ontology='official', mode='full', num_shards=1,
        shard_index=0)
    entry = dict(m.config, banks={'original': next(iter(m.vocabulary['banks'].values()))})
    validation, _, images, masks, names_map = load_samples(loader_args, entry)
    lookup = {s.key: s for s in validation}
    optimized_model = FrozenGeometry(record['config_path'],
        checkpoint_dir=protocol['checkpoint_dir'], dinov3_repo=protocol['dinov3_repo'],
        upstream_root=protocol['upstream_root'], text_cache=record['text_cache'],
        efficient=True, wide_branch='query_slot', query_slot_capacity=24)
    optimized_model.profile = replace(optimized_model.profile, coupling=record['reference_g0'])
    optimized = OptimizedWideReadout(optimized_model, wide_readout_config('anchored_infra'))
    rows = []
    for key in record['cost_keys'][:args.limit]:
        image = images(lookup[key])
        ref = reference.predict_cell(image, {
            'method': 'Anchored', 'g0': record['reference_g0'], 'alpha': None})
        fast = optimized.predict(image)
        target = masks(lookup[key], args.dataset, tuple(image.shape[-2:]))
        rows.append(dict(key=key, changed_pixels=int(np.count_nonzero(ref != fast)),
                        total_pixels=int(ref.size),
                        reference_miou=float((ref == target).mean()),
                        optimized_miou=float((fast == target).mean())))
    output = dict(dataset=args.dataset, images=rows,
                  changed_pixels=sum(r['changed_pixels'] for r in rows),
                  total_pixels=sum(r['total_pixels'] for r in rows),
                  model='Anchored FP64 compatibility vs anchored_infra',
                  masks_used_only_for_scoring=True)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--limit', type=int, default=7)
    main(parser.parse_args())
