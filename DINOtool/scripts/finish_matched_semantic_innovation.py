#!/usr/bin/env python3
"""Wait on exact matched-readout tmux handles, then coverage-verify the merge."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from merge_matched_semantic_innovation import main as merge


def main(args):
    root = Path(args.root)
    while True:
        complete = []
        for index in range(args.shards):
            path = root/f"s{index}"/"results.json"
            row = None
            if path.exists():
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            done = row is not None and row.get("status") == "complete" and row["processed_images"] == row["total_images"]
            complete.append(done)
            if not done:
                session = f"rcf04_{args.mode}_{args.dataset}_s{index}"
                if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
                    log = root/f"s{index}.log"
                    raise RuntimeError(f"Exited incomplete: {session}; log: {log}\n" + (log.read_text()[-6000:] if log.exists() else "Missing log"))
        if all(complete):
            break
        time.sleep(30)
    rows = [json.loads((root/f"s{i}"/"results.json").read_text()) for i in range(args.shards)]
    if sum(row["processed_images"] for row in rows) != args.expected_images:
        raise ValueError("Unexpected fixed sample count.")
    merge(argparse.Namespace(inputs=[str(root/f"s{i}") for i in range(args.shards)], output=str(root/"merged.json")))
    print(f"Verified {args.mode} {args.dataset}: {args.expected_images}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True, choices=("full", "diagnostic"))
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--expected-images", required=True, type=int)
    main(parser.parse_args())
