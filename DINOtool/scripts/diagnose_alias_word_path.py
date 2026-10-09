"""Observe two fixed existing inputs/domain without loading masks or changing predictions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dinotool.alias_word_path_audit import IMPLEMENTATION, WordPathAudit
from dinotool.bounded_fine_execution import FineCoverageExecution
from dinotool.model import checkpoint_manifest
from dinotool.reciprocal_alias_admission import ONE_SIDED_SOFT, predict_image
from eval_bounded_crop_head_alias import panel_inputs
from eval_rival_fine_full import check_frozen, frozen_state, save


@torch.inference_mode()
def main(args):
    output, samples, load_image, _, geometry, banks, vip, queries, checkpoints, old = panel_inputs(args)
    selected = (samples[0], samples[-1])
    if selected[0].key == selected[1].key:
        raise ValueError("Two distinct fixed panel inputs required.")
    state = frozen_state(geometry, vip)
    execution = FineCoverageExecution(cached=True, burst=True)
    audit = WordPathAudit()
    output.mkdir(parents=True)
    result = dict(status="running", implementation=IMPLEMENTATION, dataset=args.dataset,
        sample_keys=[s.key for s in selected], total_images=2, processed_images=0,
        target_masks_loaded=False, model_rule_changed=False, vocabulary=old["signature"]["vocabulary"],
        checkpoints=checkpoint_manifest(checkpoints), observations=[], rows=[])
    save(output / "results.json", result)
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        audit.begin(sample.key, banks)
        current, diagnostics = audit.predict(image, geometry, banks, vip, queries, execution=execution)
        previous, _ = predict_image(image, geometry, banks, vip, queries,
                                    methods=(ONE_SIDED_SOFT,), execution=execution)
        if any(not np.array_equal(current[p][ONE_SIDED_SOFT], previous[p][ONE_SIDED_SOFT]) for p in banks):
            raise RuntimeError("Audited complete-image prediction differs from unchanged soft source.")
        if any(not 0 < row["fine_forwards"] <= 16 or row["geometry_encodings"] > 4
               or row["wide_encodings"] > 4 for row in diagnostics.values()):
            raise RuntimeError("Existing bounded observation cap exceeded.")
        result.update(processed_images=number, rows=audit.rows,
            **check_frozen(state, geometry, vip))
        result["observations"].append(dict(sample_key=sample.key, diagnostics=diagnostics,
            complete_prediction_bitwise_equal=True))
        save(output / "results.json", result)
        print(f"{args.dataset}: audited {number}/2; exact unchanged prediction", flush=True)
    result.update(status="complete", exact_original_scores_and_predictions=True, execution=execution.report())
    save(output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "upstream-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--mode", choices=("smoke",), default="smoke")
    main(parser.parse_args())
