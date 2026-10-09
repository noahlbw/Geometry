#!/usr/bin/env bash
# Download the exact COCO-Stuff assets required by the CAFe source-only runs.
# Archives are retained so interrupted downloads remain resumable and auditable.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/COCOStuff2017" >&2
  exit 2
fi

root=$1
archives="$root/archives"
logs="$root/logs"
mkdir -p "$archives" "$logs"

download() {
  local url=$1
  local name=$2
  aria2c \
    --continue=true \
    --file-allocation=none \
    --max-connection-per-server=8 \
    --split=8 \
    --min-split-size=8M \
    --max-tries=20 \
    --retry-wait=5 \
    --summary-interval=60 \
    --console-log-level=notice \
    --dir="$archives" \
    --out="$name" \
    "$url" 2>&1 | tee -a "$logs/download.log"
}

# Small assets first: they unblock label/mapping validation before the 18 GB
# training-image archive completes.
download "http://images.cocodataset.org/zips/val2017.zip" "val2017.zip"
download "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip" "stuffthingmaps_trainval2017.zip"
download "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip"

for archive in "$archives"/*.zip; do
  unzip -t "$archive" >> "$logs/zip_test.log"
done

date --iso-8601=seconds > "$logs/download_complete_at.txt"
