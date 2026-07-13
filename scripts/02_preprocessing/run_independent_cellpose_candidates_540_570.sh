#!/usr/bin/env bash
set -euo pipefail

# Generate independent CPSAM/Cellpose candidate masks for original frames 540-570.
#
# The script runs Cellpose with the cpsam pretrained model on raw and
# gamma-preprocessed input folders and stores one output mask folder per
# candidate setting. Existing complete 31-frame outputs are skipped.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE="${PROJECT_ROOT}/../Tribolium_Daten"
LOG_DIR="${PROJECT_ROOT}/logs/cellpose"

mkdir -p "${LOG_DIR}"

run_cellpose_candidate () {
    local name="$1"
    local input_dir="$2"
    local diameter="$3"

    local output_dir="${BASE}/${name}"
    local log_file="${LOG_DIR}/${name}.log"

    mkdir -p "${output_dir}"

    local existing_count=0
    if compgen -G "${output_dir}/*_cp_masks.png" > /dev/null; then
        existing_count=$(ls "${output_dir}"/*_cp_masks.png | wc -l)
    fi

    if [ "${existing_count}" -eq 31 ]; then
        echo "Skipping ${name}: already has 31 masks."
        return
    fi

    echo
    echo "Running ${name}"
    echo "Input: ${input_dir}"
    echo "Output: ${output_dir}"
    echo "Diameter: ${diameter}"
    echo "Log: ${log_file}"

    python -m cellpose \
      --dir "${input_dir}" \
      --savedir "${output_dir}" \
      --verbose \
      --use_gpu \
      --save_png \
      --pretrained_model cpsam \
      --diameter "${diameter}" \
      --no_npy \
      > "${log_file}" 2>&1

    echo "Finished ${name}"
}

# Raw-image cpsam candidates.
run_cellpose_candidate \
  "independent_cellpose_cpsam_raw_d45_540_570" \
  "${BASE}/independent_inputs/raw_540_570" \
  45

run_cellpose_candidate \
  "independent_cellpose_cpsam_raw_d65_540_570" \
  "${BASE}/independent_inputs/raw_540_570" \
  65

# Gamma-preprocessed cpsam candidates.
run_cellpose_candidate \
  "independent_cellpose_cpsam_gamma05_d55_540_570" \
  "${BASE}/independent_inputs/gamma_0.5_540_570" \
  55

run_cellpose_candidate \
  "independent_cellpose_cpsam_gamma07_d55_540_570" \
  "${BASE}/independent_inputs/gamma_0.7_540_570" \
  55

run_cellpose_candidate \
  "independent_cellpose_cpsam_gamma09_d55_540_570" \
  "${BASE}/independent_inputs/gamma_0.9_540_570" \
  55

echo
echo "All independent Cellpose candidates finished."
