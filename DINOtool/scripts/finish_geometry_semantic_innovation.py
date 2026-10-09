#!/usr/bin/env python3
"""Wait for a batch's own shards, fail on exited jobs, then verify and merge."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from merge_geometry_semantic_innovation import main as merge


EXPECTED = {"vdd": 80, "potsdam": 504, "udd5": 40, "oem": 384,
            "vaihingen": 113, "landcoverai": 1602, "loveda": 1669, "flair1": 15700}


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
            done = (row is not None and row.get("status") == "complete"
                    and row["processed_images"] == row["total_images"])
            complete.append(done)
            if not done:
                session = f"gsi03_full_{args.dataset}_s{index}"
                running = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
                if not running:
                    log = root/f"s{index}.log"
                    raise RuntimeError(f"{session} exited without a complete result. Preserved log: {log}\n"
                                       + (log.read_text()[-6000:] if log.exists() else "Log missing."))
        if all(complete):
            break
        time.sleep(30)
    rows = [json.loads((root/f"s{i}"/"results.json").read_text()) for i in range(args.shards)]
    if sum(row["processed_images"] for row in rows) != EXPECTED[args.dataset]:
        raise ValueError("Unexpected full dataset size.")
    merge(argparse.Namespace(inputs=[str(root/f"s{i}") for i in range(args.shards)], output=str(root/"merged.json")))
    print(f"Verified full {args.dataset}: {EXPECTED[args.dataset]} images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True, choices=tuple(EXPECTED))
    parser.add_argument("--shards", type=int, required=True)
    main(parser.parse_args())
