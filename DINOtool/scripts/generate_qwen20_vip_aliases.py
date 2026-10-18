"""Generate pure Qwen20 candidate banks for the VIP paper-rule experiment.

Each class is represented by its canonical taxonomy name followed by the first
19 structurally valid Qwen proposals.  No image, mask, mIoU, or VIP score is
read during generation.  The raw prompts and responses are retained so the
resulting bank can be audited and reproduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from dinotool.semantic_role_admission import FrozenSemanticRoles, SOURCE


DATASETS = {
    "vdd": "grounded_vdd_official20.json",
    "potsdam": "grounded_potsdam20.json",
    "udd5": "hero_udd5_vip20.json",
    "oem": "hero_oem_vip20.json",
    "loveda": "gar_llm_raw20_loveda_v1.json",
    "vaihingen": "grounded_vaihingen20.json",
    "landcoverai": "gear_landcoverai_v1_20.json",
    "flair1": "gear_flair1_main12_20.json",
}


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def prompt(name: str, competitors: list[str], accepted: list[str], needed: int) -> str:
    return (
        "You are generating a fixed open-vocabulary semantic-segmentation query bank. "
        "Return only a JSON array of short English noun phrases. Target class: "
        + json.dumps(name) + ". Competing classes in the same taxonomy: "
        + json.dumps(competitors) + ". Already accepted phrases (do not repeat): "
        + json.dumps(accepted) + ". Generate " + str(needed) + " new phrases. "
        "A phrase must directly denote the target class, a valid subtype, or a "
        "recognizable visual form of it. Do not return parts, nearby objects, "
        "scene context, materials shared with a rival, exact rival labels, "
        "plural-only variants, or explanations. Use at most six words per phrase."
    )


def parse(reply: str) -> list[str]:
    text = reply.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[text.find("\n") + 1 : -3].strip()
    if not text.startswith("["):
        raise ValueError("response is not a JSON array")
    values = json.loads(text)
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("expected an array of strings")
    return [" ".join(v.split()) for v in values]


def main(args: argparse.Namespace) -> None:
    configs = Path(args.configs_dir)
    datasets = tuple(args.datasets) if args.datasets else tuple(DATASETS)
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError("unknown datasets: " + ", ".join(sorted(unknown)))
    out = Path(args.output_dir)
    manifest_path = out / "generation.json"
    if manifest_path.exists():
        raise RuntimeError("refusing an existing Qwen generation: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    specs = {}
    records = []
    for dataset in datasets:
        path = configs / DATASETS[dataset]
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = payload.get("classes", payload)
        if not isinstance(classes, list) or not classes:
            raise ValueError("invalid class config: " + str(path))
        names = []
        rows = []
        for entry in classes:
            name = str(entry["name"] if isinstance(entry, dict) else entry)
            if name in names:
                raise ValueError("duplicate class: " + name)
            names.append(name)
        for index, name in enumerate(names):
            rows.append({"name": name, "competitors": [n for n in names if n != name],
                         "aliases": [name], "responses": [], "dataset": dataset,
                         "class_index": index})
        specs[dataset] = {"config_source": str(path), "class_names": names,
                          "classes": rows}
        records.extend(rows)
    model = FrozenSemanticRoles(SOURCE)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    result = {
        "status": "running", "implementation": "qwen20-vip-candidates-v1-20261018",
        "source_manifest": model.manifest, "target_images_loaded": False,
        "target_masks_loaded": False, "semantic_or_visual_filtering": False,
        "generation": {"seed": args.seed, "do_sample_after_first": True,
                        "temperature": 0.7, "top_p": 0.9, "num_beams": 1,
                        "max_new_tokens": 1024, "batch_size": args.batch_size,
                        "requested_total": 20, "canonical_plus_qwen_proposals": 19,
                        "maximum_attempts": args.attempts},
        "rule": "canonical taxonomy name plus first 19 valid Qwen proposals; "
                "duplicates and exact rival labels are rejected mechanically; "
                "no score, image, mask, or manual semantic filter is applied.",
        "datasets": specs, "records": records,
    }
    pending = list(range(len(records)))
    for attempt in range(args.attempts):
        if not pending:
            break
        next_pending = []
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start : start + args.batch_size]
            questions = [prompt(records[i]["name"], records[i]["competitors"],
                                records[i]["aliases"], 32) for i in indices]
            rendered = [model.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": q}]}],
                tokenize=False, add_generation_prompt=True) for q in questions]
            inputs = model.processor.tokenizer(rendered, padding=True, return_tensors="pt").to(model.device)
            generation = dict(do_sample=attempt > 0, num_beams=1,
                               max_new_tokens=1024, use_cache=True,
                               pad_token_id=model.processor.tokenizer.pad_token_id)
            if attempt > 0:
                generation.update(temperature=0.7, top_p=0.9)
            with torch.inference_mode():
                generated = model.model.generate(**inputs, **generation)
            replies = model.processor.tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            for index, question, reply in zip(indices, questions, replies):
                row = records[index]
                entry = {"attempt": attempt, "prompt": question,
                         "raw_response": reply, "rejected": []}
                try:
                    seen = {x.casefold() for x in row["aliases"]}
                    rivals = {x.casefold() for x in row["competitors"]}
                    for value in parse(reply):
                        reason = None
                        if not value:
                            reason = "empty"
                        elif len(value.split()) > 6:
                            reason = "length"
                        elif value.casefold() in seen:
                            reason = "duplicate"
                        elif value.casefold() in rivals:
                            reason = "exact_competitor"
                        elif len(row["aliases"]) >= 20:
                            reason = "over_twenty_budget"
                        if reason:
                            entry["rejected"].append({"alias": value, "reason": reason})
                            continue
                        row["aliases"].append(value)
                        seen.add(value.casefold())
                    row["responses"].append(entry)
                except (ValueError, TypeError) as error:
                    entry["format_error"] = str(error)
                    row["responses"].append(entry)
                if len(row["aliases"]) != 20:
                    next_pending.append(index)
            result.update(complete_classes=sum(len(r["aliases"]) == 20 for r in records),
                          total_classes=len(records), wall_seconds=time.perf_counter() - started,
                          peak_allocated_mib=float(torch.cuda.max_memory_allocated() / 1048576
                                                   if torch.cuda.is_available() else 0.0))
            save(manifest_path, result)
            print(json.dumps({"attempt": attempt, "complete_classes": result["complete_classes"],
                              "total_classes": len(records)}), flush=True)
        pending = next_pending
    if pending:
        result.update(status="failed", failed_records=pending)
        save(manifest_path, result)
        raise RuntimeError("Qwen did not fill every 20-slot bank; raw responses preserved")
    for dataset, group in specs.items():
        classes = [{"name": row["name"], "synonyms": row["aliases"]}
                   for row in records if row["dataset"] == dataset]
        payload = {"protocol": "Qwen20 candidate bank for VIP paper-rule distillation",
                   "source_generation": str(manifest_path), "classes": classes}
        save(out / (dataset + ".json"), payload)
        group["vocabulary_path"] = str(out / (dataset + ".json"))
        group["vocabulary_sha256"] = hashlib.sha256(
            (out / (dataset + ".json")).read_bytes()).hexdigest()
    result.update(status="complete", failed_records=[], weights_frozen=all(
        not p.requires_grad for p in model.model.parameters()),
        wall_seconds=time.perf_counter() - started)
    save(manifest_path, result)
    print(json.dumps({"status": "complete", "datasets": list(datasets),
                      "classes": len(records), "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--configs-dir", required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    main(parser.parse_args())
