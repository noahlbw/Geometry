#!/usr/bin/env bash
# Extract a verified iSAID target-only download into an immutable evaluation root.
# This script never trains, launches inference, or accepts a target checkpoint.
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
download_root="$target_root/downloads"
extract_root="$target_root/extracted"

for required in "$target_root/DOWNLOAD_COMPLETE" "$target_root/download_manifest.tsv" "$target_root/download_sha256.tsv" "$target_root/official_source_url.txt"; do
  [[ -f "$required" ]] || { echo "Missing verified download provenance: $required" >&2; exit 1; }
done
[[ -d "$download_root" ]] || { echo "Missing download directory: $download_root" >&2; exit 1; }
[[ ! -e "$extract_root" ]] || { echo "Refusing to overwrite an existing extraction: $extract_root" >&2; exit 1; }

mapfile -t archives < <(find "$download_root" -type f -iname '*.zip' -print | LC_ALL=C sort)
if [[ ${#archives[@]} -ne 1 ]]; then
  printf 'Expected exactly one official validation ZIP, found %s:\n' "${#archives[@]}" >&2
  printf '  %s\n' "${archives[@]:-<none>}" >&2
  exit 1
fi
archive=${archives[0]}
unzip -tq "$archive" >/dev/null

# Preserve any failed extraction for forensic inspection instead of deleting it.
staging=$(mktemp -d "$target_root/.isaid-extract-staging.XXXXXX")
unzip -q "$archive" -d "$staging"
mapfile -t validation_dirs < <(find "$staging" -type d -path '*/val/images' -print | LC_ALL=C sort)
if [[ ${#validation_dirs[@]} -ne 1 ]]; then
  printf 'Expected exactly one extracted official val/images directory, found %s:\n' "${#validation_dirs[@]}" >&2
  printf '  %s\n' "${validation_dirs[@]:-<none>}" >&2
  echo "Preserved staging directory for inspection: $staging" >&2
  exit 1
fi

image_dir=${validation_dirs[0]}
rgb_count=$(find "$image_dir" -maxdepth 1 -type f -name '*.png' ! -name '*_instance_color_RGB.png' ! -name '*_instance_id_RGB.png' | wc -l)
semantic_count=$(find "$image_dir" -maxdepth 1 -type f -name '*_instance_color_RGB.png' | wc -l)
instance_count=$(find "$image_dir" -maxdepth 1 -type f -name '*_instance_id_RGB.png' | wc -l)
[[ "$rgb_count" -gt 0 && "$rgb_count" -eq "$semantic_count" && "$rgb_count" -eq "$instance_count" ]] || {
  echo "iSAID validation triples are incomplete: rgb=$rgb_count semantic=$semantic_count instance=$instance_count" >&2
  echo "Preserved staging directory for inspection: $staging" >&2
  exit 1
}

mv "$staging" "$extract_root"
find "$extract_root" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "$target_root/extraction_manifest.tsv"
(cd "$extract_root" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$target_root/extraction_sha256.tsv"
printf 'rgb=%s\nsemantic=%s\ninstance=%s\n' "$rgb_count" "$semantic_count" "$instance_count" > "$target_root/extraction_counts.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$target_root/EXTRACT_COMPLETE"
echo "iSAID validation target extracted and verified at $extract_root"
