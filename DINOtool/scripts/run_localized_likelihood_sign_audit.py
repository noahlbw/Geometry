"""Run only the fixed sign counterfactuals; no model selection or rollout."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from audit_localized_likelihood_signs import IMPLEMENTATION
from run_region_semantic_suite_a800 import TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import read_json, alive, exact_miou


GPUS = (4, 5, 6, 7)
SOURCE = TOOL/"results/geometry_localized_likelihood_cached_r2_20261002"


def launch(root, setting, gpu):
    if gpu not in GPUS or not idle(gpu):
        return None
    dataset, data, vocab, _, _ = setting
    output, log, session = root/dataset, root/f"{dataset}.log", "gls02_"+dataset
    if output.exists() or alive(session):
        raise RuntimeError("Refusing existing sign audit output/session.")
    args = [PYTHON, "-u", "scripts/audit_localized_likelihood_signs.py", "--dataset", dataset,
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
        "--source", str(SOURCE/dataset), "--output-dir", str(output)]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    return {"dataset": dataset, "gpu": gpu, "output": str(output), "log": str(log), "session": session}


def main(root):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing audit root.")
    root.mkdir(parents=True)
    (root/"protocol.json").write_text(json.dumps({"implementation": IMPLEMENTATION, "physical_gpus": GPUS,
        "source": str(SOURCE), "parameter_search": False, "new_model_selected": False,
        "source_model_rerun": False, "scope": "Fixed sign-removal counterfactuals on96 images' cached windows"}, indent=2)+"\n")
    queue, active, outcomes, failures = list(SETTINGS), {}, {}, {}
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            row = read_json(Path(job["output"])/"results.json")
            if (row and row["status"] == "complete" and row["implementation"] == IMPLEMENTATION
                    and row.get("source_predictions_exact") and row["coverage_verified"]):
                outcomes[job["dataset"]] = {key: {method: exact_miou(metric) for method, metric in group.items()}
                                             for key, group in row["metrics"].items()}
            else:
                failures[job["dataset"]] = Path(job["log"]).read_text()[-6000:]
            del active[gpu]
        if failures:
            queue.clear()
        for gpu in GPUS:
            if queue and gpu not in active and idle(gpu):
                job = launch(root, queue[0], gpu)
                if job:
                    queue.pop(0)
                    active[gpu] = job
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
            "implementation": IMPLEMENTATION, "physical_gpus": GPUS, "active": list(active.values()),
            "queued": [row[0] for row in queue], "completed": list(outcomes), "outcomes": outcomes,
            "failures": failures, "promotion_or_model_selection": False}
        temporary = root/"suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    (root/"suite_results.json").write_text(json.dumps(state, indent=2)+"\n")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
