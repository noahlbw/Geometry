"""Read-only progress snapshot for the current fixed eight-domain suite."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                   "--format=csv,noheader"], text=True))
    for manifest in sorted(args.root.glob("*_samples.json")):
        dataset = manifest.name.removesuffix("_samples.json")
        path = args.root/dataset/"s0/results.json"
        merged = args.root/dataset/"merged.json"
        session = "grsr02_screen_"+dataset
        alive = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
        if path.exists():
            row = json.loads(path.read_text())
            print(json.dumps({"dataset": dataset, "status": row["status"], "images": row["processed_images"],
                "total": row["total_images"], "alive": alive, "merged": merged.exists(),
                "pool_replay_error": row["trained_pool_replay_max_error"],
                "miou": {key: {method: metric["mean_iou_percent"] for method, metric in methods.items()}
                         for key, methods in row["metrics"].items()}}))
        elif alive:
            print(dataset, "initializing/running first image")
        else:
            log = args.root/f"{dataset}.log"
            print(dataset, "NO ACTIVE SESSION", log.read_text()[-4000:] if log.exists() else "no log")
    result = args.root/"suite_results.json"
    if result.exists():
        print(result.read_text())
    controller = args.root.with_name(args.root.name+"_controller.log")
    if controller.exists():
        print("Controller tail", controller.read_text()[-4500:])


if __name__ == "__main__":
    main()
