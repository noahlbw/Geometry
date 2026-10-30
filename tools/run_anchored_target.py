"""Deploy and benchmark opt-in Anchored execution candidates on the A800 host.

This runner is deliberately separate from the old shared-canvas screen.  It
uploads only the frozen Anchored evaluator and launches fresh processes for a
fixed cost set and a small mIoU screen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import time

from a800_region_experiment import connect


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926"
REMOTE_EXPERIMENT = REMOTE_ROOT + "/results/geov2_anchored_target_20261031"
PYTHON = "/data/miniconda3/envs/pfu/bin/python"
THIRD = "/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
DATASETS = ("vdd", "potsdam", "voc20", "voc21")
GPUS = (1, 2, 3, 4)

FILES = (
    "dinotool/anchored_infra.py",
    "dinotool/anchored_slot_memory.py",
    "dinotool/batched_attention.py",
    "dinotool/compact_deployment.py",
    "dinotool/context_memory_observer.py",
    "dinotool/efficient_deployment.py",
    "dinotool/frozen_deployment.py",
    "dinotool/geov2_wide_optimized.py",
    "dinotool/query_slot_memory.py",
    "scripts/eval_geov2_wide_optimization.py",
)


def protocol() -> dict:
    source = json.loads(
        (LOCAL_ROOT / "research/geov2_wide_optimized_20261015/optimization_protocol.json")
        .read_text(encoding="utf-8")
    )
    source["implementation"] = "geov2-anchored-target-screen-20261031"
    source["screen_images"] = 16
    return source


def remote(command: str) -> str:
    client = connect()
    try:
        _, stdout, stderr = client.exec_command(command, timeout=180)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(err or out)
        return out
    finally:
        client.close()


def prepare() -> None:
    root = LOCAL_ROOT / "research/geov2_anchored_target_20261031"
    root.mkdir(parents=True, exist_ok=True)
    (root / "optimization_protocol.json").write_text(
        json.dumps(protocol(), indent=2) + "\n", encoding="utf-8"
    )
    remote("mkdir -p " + shlex.quote(REMOTE_EXPERIMENT))
    client = connect()
    try:
        with client.open_sftp() as sftp:
            sftp.put(
                str(root / "optimization_protocol.json"),
                REMOTE_EXPERIMENT + "/optimization_protocol.json",
            )
            for relative in FILES:
                sftp.put(str(LOCAL_ROOT / "DINOtool" / relative), REMOTE_ROOT + "/" + relative)
    finally:
        client.close()
    print(json.dumps(dict(experiment=REMOTE_EXPERIMENT, files=len(FILES))))


def launch(variants: list[str], mode: str) -> None:
    if len(variants) not in (1, len(DATASETS)):
        raise ValueError("Pass one variant for all datasets, or exactly four variants.")
    if len(variants) == 1:
        variants = variants * len(DATASETS)
    if len(variants) > len(DATASETS):
        raise ValueError("At most four variants can run concurrently in this launcher.")
    jobs = []
    client = connect()
    try:
        for gpu, dataset, variant in zip(GPUS, DATASETS, variants):
            out = f"{REMOTE_EXPERIMENT}/{mode}/{variant}/{dataset}"
            output_parent = out.rsplit("/", 1)[0]
            _, cleanup_stdout, cleanup_stderr = client.exec_command(
                "rm -rf " + shlex.quote(out) + " " + shlex.quote(out + "_s0"), timeout=60
            )
            if cleanup_stdout.channel.recv_exit_status():
                raise RuntimeError(cleanup_stderr.read().decode("utf-8", errors="replace"))
            command = (
                "mkdir -p "
                + shlex.quote(output_parent)
                + " && cd "
                + shlex.quote(REMOTE_ROOT)
                + " && export CUDA_VISIBLE_DEVICES="
                + str(gpu)
                + " && export PYTHONPATH="
                + shlex.quote(REMOTE_ROOT + ":" + REMOTE_ROOT + "/scripts:" + THIRD)
                + " && "
                + shlex.join(
                    [
                        PYTHON,
                        "-u",
                        REMOTE_ROOT + "/scripts/eval_geov2_wide_optimization.py",
                        "--experiment",
                        REMOTE_EXPERIMENT,
                        "--dataset",
                        dataset,
                        "--variant",
                        variant,
                        "--mode",
                        mode,
                        "--num-shards",
                        "1",
                        "--shard-index",
                        "0",
                    ]
                )
                + " > "
                + shlex.quote(out + ".log")
                + " 2>&1"
            )
            session = f"geov2_target_{mode}_{variant}_{dataset}"
            _, stdout, stderr = client.exec_command(
                "tmux new-session -d -s " + shlex.quote(session) + " " + shlex.quote(command),
                timeout=60,
            )
            status = stdout.channel.recv_exit_status()
            if status:
                raise RuntimeError(stderr.read().decode("utf-8", errors="replace"))
            jobs.append(dict(dataset=dataset, variant=variant, gpu=gpu, session=session, output=out))
    finally:
        client.close()
    root = LOCAL_ROOT / "research/geov2_anchored_target_20261031"
    (root / f"{mode}_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jobs, indent=2))


def status(mode: str) -> None:
    root = LOCAL_ROOT / "research/geov2_anchored_target_20261031"
    jobs = json.loads((root / f"{mode}_jobs.json").read_text(encoding="utf-8"))
    rows = []
    client = connect()
    try:
        for job in jobs:
            path = job["output"] + "_s0/results.json"
            command = (
                "python3 -c "
                + shlex.quote(
                    "import json,pathlib; p=pathlib.Path(" + repr(path) + "); "
                    "print(json.dumps(json.loads(p.read_text()) if p.exists() else {}))"
                )
            )
            _, stdout, _ = client.exec_command(command, timeout=60)
            text = stdout.read().decode("utf-8", errors="replace").strip()
            _, log_stdout, _ = client.exec_command(
                "tail -n 20 " + shlex.quote(job["output"] + ".log") + " 2>/dev/null || true",
                timeout=60,
            )
            log_text = log_stdout.read().decode("utf-8", errors="replace")
            rows.append(dict(job=job, result=json.loads(text) if text else {}, log_tail=log_text))
    finally:
        client.close()
    print(json.dumps(rows, indent=2))


def collect(mode: str) -> None:
    root = LOCAL_ROOT / "research/geov2_anchored_target_20261031"
    jobs = json.loads((root / f"{mode}_jobs.json").read_text(encoding="utf-8"))
    client = connect()
    try:
        with client.open_sftp() as sftp:
            for job in jobs:
                target = root / mode / job["variant"] / job["dataset"]
                target.mkdir(parents=True, exist_ok=True)
                for name in ("results.json", "statistics.npz", "progress.json"):
                    try:
                        sftp.get(job["output"] + "_s0/" + name, str(target / name))
                    except FileNotFoundError:
                        pass
    finally:
        client.close()
    print("collected", mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "status", "collect"))
    parser.add_argument("--variants", nargs="+", default=["anchored_infra"])
    parser.add_argument("--mode", choices=("smoke", "screen", "cost", "full"), default="cost")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "launch":
        launch(args.variants, args.mode)
    elif args.action == "status":
        status(args.mode)
    else:
        collect(args.mode)


if __name__ == "__main__":
    main()
