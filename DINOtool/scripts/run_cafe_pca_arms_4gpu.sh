#!/usr/bin/env bash
# Sequential matched source-only PCA-DINO screen on one four-GPU allocation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_ROOT" >&2
    exit 2
fi

root=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
launcher=$script_dir/run_cafe_pca_4gpu.sh
arms=(serial parallel pca_epl pca_epl_fod)

[[ ! -e $root && -x $launcher ]] || {
    echo "output root must be fresh and run_cafe_pca_4gpu.sh must be executable" >&2
    exit 2
}
mkdir -p "$root"

for arm in "${arms[@]}"; do
    output=$root/$arm
    "$launcher" "$output" screen "$arm" 2>&1 | tee "$root/${arm}.log"
done
