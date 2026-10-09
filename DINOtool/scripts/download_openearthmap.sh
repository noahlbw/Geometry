#!/usr/bin/env bash
set -euo pipefail

# Download the official OpenEarthMap archive with resume and checksum support.
# The source labels have mixed upstream licenses; consult the project attribution
# before using resulting checkpoints outside research-compatible terms.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/datasets/OpenEarthMap" >&2
  exit 2
fi

dataset_root=$1
archive_dir=$(dirname "$dataset_root")
archive_path="$archive_dir/OpenEarthMap.zip"
archive_dataset_root="$archive_dir/OpenEarthMap_wo_xBD"
url="https://zenodo.org/api/records/7223446/files/OpenEarthMap.zip/content"
expected_md5="64155d1dc9d3b68536063f79878e1a67"
archive_bytes=9099481727
# Keep the archive layout stable when reducing concurrency after a CDN rate
# limit. Existing installations used eight ranges, so retain that partition
# count while only opening two connections at a time by default.
download_jobs=${OEM_DOWNLOAD_JOBS:-2}
part_count=${OEM_PART_COUNT:-8}
curl_max_attempts=${OEM_CURL_MAX_ATTEMPTS:-360}

write_provenance() {
  local target_root=$1
  cat > "$target_root/provenance.json" <<EOF
{
  "dataset": "OpenEarthMap",
  "doi": "10.5281/zenodo.7223446",
  "record": "https://zenodo.org/records/7223446",
  "archive": "OpenEarthMap.zip",
  "archive_md5": "$expected_md5",
  "archive_directory": "$(basename "$archive_dataset_root")",
  "license_note": "OpenEarthMap label licensing follows the underlying RGB imagery and may vary by source. The project states CC BY-NC-SA 4.0 for labels where an image license is absent or public domain.",
  "downloaded_utc": "$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

if [[ -f "$dataset_root/train.txt" && -f "$dataset_root/val.txt" ]]; then
  echo "OpenEarthMap is already unpacked at $dataset_root"
  exit 0
fi

# The official archive is intentionally named OpenEarthMap_wo_xBD because it
# excludes xBD RGB images. Preserve a stable caller-provided path with a
# symlink while retaining the archive's factual directory name for provenance.
if [[ -f "$archive_dataset_root/train.txt" && -f "$archive_dataset_root/val.txt" ]]; then
  if [[ -e "$dataset_root" ]]; then
    echo "Requested dataset path exists without OpenEarthMap split files: $dataset_root" >&2
    exit 1
  fi
  ln -s "$archive_dataset_root" "$dataset_root"
  write_provenance "$archive_dataset_root"
  echo "OpenEarthMap is already unpacked at $archive_dataset_root"
  exit 0
fi

if [[ -e "$dataset_root" ]]; then
  echo "Refusing to unpack into a non-empty path without OpenEarthMap split files: $dataset_root" >&2
  exit 1
fi

mkdir -p "$archive_dir"

if ! [[ "$download_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "OEM_DOWNLOAD_JOBS must be a positive integer, got '$download_jobs'" >&2
  exit 2
fi

if ! [[ "$part_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "OEM_PART_COUNT must be a positive integer, got '$part_count'" >&2
  exit 2
fi

if ! [[ "$curl_max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "OEM_CURL_MAX_ATTEMPTS must be a positive integer, got '$curl_max_attempts'" >&2
  exit 2
fi

if [[ -f "$archive_path" && $(stat -c %s "$archive_path") -ne "$archive_bytes" ]]; then
  backup_path="$archive_path.single.partial.$(date --utc +%Y%m%dT%H%M%SZ)"
  mv "$archive_path" "$backup_path"
  echo "Preserved prior single-stream partial download at $backup_path"
fi
parts_dir="$archive_path.parts"
mkdir -p "$parts_dir"
chunk_bytes=$(((archive_bytes + part_count - 1) / part_count))
active_pids=()
status=0
for ((index = 0; index < part_count; index += 1)); do
  if [[ "${#active_pids[@]}" -ge "$download_jobs" ]]; then
    if ! wait "${active_pids[0]}"; then
      status=1
    fi
    active_pids=("${active_pids[@]:1}")
  fi
    start=$((index * chunk_bytes))
    if [[ "$start" -ge "$archive_bytes" ]]; then
      break
    fi
    end=$((start + chunk_bytes - 1))
    if [[ "$end" -ge "$archive_bytes" ]]; then
      end=$((archive_bytes - 1))
    fi
    part_path="$parts_dir/part-$(printf '%03d' "$index")"
    (
      expected_part_bytes=$((end - start + 1))
      existing_bytes=0
      if [[ -f "$part_path" ]]; then
        existing_bytes=$(stat -c %s "$part_path")
      fi

      # A connection can be closed by the CDN after it has written a valid
      # prefix. Fold that prefix into the durable part before issuing the next
      # range request, so rerunning this script never loses downloaded bytes.
      shopt -s nullglob
      temporary_parts=("$part_path".tmp.*)
      if [[ "${#temporary_parts[@]}" -gt 1 ]]; then
        echo "Found multiple incomplete temporary parts for $part_path; refusing to guess their order." >&2
        exit 1
      fi
      if [[ "${#temporary_parts[@]}" -eq 1 ]]; then
        temporary_bytes=$(stat -c %s "${temporary_parts[0]}")
        remaining_bytes=$((expected_part_bytes - existing_bytes))
        if [[ "$temporary_bytes" -gt "$remaining_bytes" ]]; then
          echo "Temporary part is larger than its remaining range: ${temporary_parts[0]}" >&2
          exit 1
        fi
        if [[ "$temporary_bytes" -gt 0 ]]; then
          cat "${temporary_parts[0]}" >> "$part_path"
        fi
        rm -f "${temporary_parts[0]}"
        existing_bytes=$((existing_bytes + temporary_bytes))
      fi

      if [[ "$existing_bytes" -gt "$expected_part_bytes" ]]; then
        echo "Part is larger than its expected range: $part_path" >&2
        exit 1
      fi
      if [[ "$existing_bytes" -eq "$expected_part_bytes" ]]; then
        exit 0
      fi
      stalled_attempts=0
      while [[ "$existing_bytes" -lt "$expected_part_bytes" ]]; do
        resume_start=$((start + existing_bytes))
        remaining_bytes=$((expected_part_bytes - existing_bytes))
        temporary_part="$part_path.tmp.$BASHPID"
        rm -f "$temporary_part"

        if curl --silent --show-error --fail --location --connect-timeout 60 --speed-time 120 --speed-limit 1024 \
          --range "$resume_start-$end" --output "$temporary_part" "$url"; then
          curl_status=0
        else
          curl_status=$?
        fi

        downloaded_bytes=0
        if [[ -f "$temporary_part" ]]; then
          downloaded_bytes=$(stat -c %s "$temporary_part")
        fi
        if [[ "$downloaded_bytes" -gt "$remaining_bytes" ]]; then
          echo "Range download size mismatch for $part_path" >&2
          exit 1
        fi
        if [[ "$downloaded_bytes" -gt 0 ]]; then
          cat "$temporary_part" >> "$part_path"
          rm -f "$temporary_part"
          existing_bytes=$((existing_bytes + downloaded_bytes))
          stalled_attempts=0
          if [[ "$curl_status" -ne 0 ]]; then
            echo "Resuming $part_path after curl exit $curl_status; ${existing_bytes}/${expected_part_bytes} bytes complete." >&2
          fi
          continue
        fi

        rm -f "$temporary_part"
        stalled_attempts=$((stalled_attempts + 1))
        if [[ "$stalled_attempts" -ge "$curl_max_attempts" ]]; then
          echo "OpenEarthMap range download made no progress for $stalled_attempts attempts: $part_path" >&2
          exit 1
        fi
        retry_delay=$((stalled_attempts * 5))
        if [[ "$retry_delay" -gt 60 ]]; then
          retry_delay=60
        fi
        echo "Retrying $part_path after curl exit $curl_status ($stalled_attempts/$curl_max_attempts without progress; waiting ${retry_delay}s)." >&2
        sleep "$retry_delay"
      done
    ) &
    active_pids+=("$!")
done
for pid in "${active_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one OpenEarthMap range download failed. Re-run this command to resume completed parts." >&2
  exit 1
fi
assembled_path="$archive_path.assembled.tmp"
: > "$assembled_path"
for ((index = 0; index < part_count; index += 1)); do
  part_path="$parts_dir/part-$(printf '%03d' "$index")"
  cat "$part_path" >> "$assembled_path"
done
assembled_bytes=$(stat -c %s "$assembled_path")
if [[ "$assembled_bytes" -ne "$archive_bytes" ]]; then
  echo "Assembled archive has unexpected size: $assembled_bytes" >&2
  exit 1
fi
mv "$assembled_path" "$archive_path"

actual_md5=$(md5sum "$archive_path" | awk '{print $1}')
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "Checksum mismatch for $archive_path: expected $expected_md5, got $actual_md5" >&2
  exit 1
fi

unzip -oq "$archive_path" -d "$archive_dir"
actual_dataset_root="$dataset_root"
if [[ ! -f "$actual_dataset_root/train.txt" || ! -f "$actual_dataset_root/val.txt" ]]; then
  if [[ -f "$archive_dataset_root/train.txt" && -f "$archive_dataset_root/val.txt" ]]; then
    actual_dataset_root="$archive_dataset_root"
    if [[ ! -e "$dataset_root" ]]; then
      ln -s "$archive_dataset_root" "$dataset_root"
    fi
  else
    echo "Archive unpacked but expected split files were not found under $dataset_root or $archive_dataset_root" >&2
    exit 1
  fi
fi
if [[ ! -f "$actual_dataset_root/train.txt" || ! -f "$actual_dataset_root/val.txt" ]]; then
  echo "OpenEarthMap split files are missing under $actual_dataset_root" >&2
  exit 1
fi

write_provenance "$actual_dataset_root"

echo "OpenEarthMap is ready at $actual_dataset_root"
