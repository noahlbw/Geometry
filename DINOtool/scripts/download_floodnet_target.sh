#!/usr/bin/env bash
# Download the official FloodNet Track-1 folder as target-only raw data.
# It intentionally neither extracts archives nor launches any evaluation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/FloodNet_target" >&2
  exit 2
fi

target_root=$1
if [[ "$target_root" != /* ]]; then
  echo "Target root must be an absolute path: $target_root" >&2
  exit 2
fi

folder_url='https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH?usp=sharing'
download_root="$target_root/downloads"
complete_marker="$target_root/DOWNLOAD_COMPLETE"

if [[ -e "$complete_marker" ]]; then
  echo "FloodNet target download already completed at $target_root"
  exit 0
fi
if [[ -e "$target_root" && ! -d "$target_root" ]]; then
  echo "Refusing non-directory target: $target_root" >&2
  exit 1
fi
command -v gdown >/dev/null || { echo "gdown is required" >&2; exit 1; }

mkdir -p "$download_root"
gdown --folder --continue --output "$download_root" "$folder_url"

file_count=$(find "$download_root" -type f | wc -l)
if [[ "$file_count" -eq 0 ]]; then
  echo "The official FloodNet Track-1 folder contained no files." >&2
  exit 1
fi
find "$download_root" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "$target_root/download_manifest.tsv"
(cd "$download_root" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$target_root/download_sha256.tsv"
printf '%s\n' "$folder_url" > "$target_root/official_source_url.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$complete_marker"
echo "FloodNet Track-1 target-only download verified at $target_root"
