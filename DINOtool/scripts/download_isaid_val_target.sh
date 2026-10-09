#!/usr/bin/env bash
# Download only the official iSAID validation folder for target-only evaluation.
# This script deliberately does not extract or consume the data for training.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/iSAID_val_target" >&2
  exit 2
fi

target_root=$1
if [[ "$target_root" != /* ]]; then
  echo "Target root must be an absolute path: $target_root" >&2
  exit 2
fi

folder_url='https://drive.google.com/drive/folders/17MErPhWQrwr92Ca1Maf4mwiarPS5rcWM?usp=sharing'
download_root="$target_root/downloads"
manifest="$target_root/download_manifest.tsv"
checksums="$target_root/download_sha256.tsv"
complete_marker="$target_root/DOWNLOAD_COMPLETE"

if [[ -e "$complete_marker" ]]; then
  echo "iSAID validation target download already completed at $target_root"
  exit 0
fi
if [[ -e "$target_root" && ! -d "$target_root" ]]; then
  echo "Refusing non-directory target: $target_root" >&2
  exit 1
fi
command -v gdown >/dev/null || { echo "gdown is required" >&2; exit 1; }

mkdir -p "$download_root"
gdown --folder --continue --output "$download_root" "$folder_url"

mapfile -t archives < <(find "$download_root" -type f \( -iname '*.zip' -o -iname '*.tar' -o -iname '*.tgz' \) -print | sort)
if [[ ${#archives[@]} -eq 0 ]]; then
  echo "The official validation folder did not yield an archive; inspect its manifest before extraction." >&2
  exit 1
fi
for archive in "${archives[@]}"; do
  case "$archive" in
    *.zip|*.ZIP) unzip -tq "$archive" >/dev/null ;;
    *.tar) tar -tf "$archive" >/dev/null ;;
    *.tgz) tar -tzf "$archive" >/dev/null ;;
  esac
done

find "$download_root" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "$manifest"
(cd "$download_root" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$checksums"
printf '%s\n' "$folder_url" > "$target_root/official_source_url.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$complete_marker"
echo "iSAID validation target download verified at $target_root"
