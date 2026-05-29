#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/result/run_all_mindts_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"
echo "Logs: ${LOG_DIR}"

scripts=(
  "scripts/univariate_detection/detect_label/KR_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/MDT_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/EWJ_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/Environment_script/MindTS.sh"
  "scripts/multivariate_detection/detect_label/Energy_script/MindTS.sh"
  "scripts/multivariate_detection/detect_label/Weather_script/MindTS.sh"
)

for script in "${scripts[@]}"; do
  dataset_name="$(basename "$(dirname "$script")" "_script")"
  log_file="${LOG_DIR}/${dataset_name}.txt"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${script}"
  if bash "$script" 2>&1 | tee "$log_file"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${script}"
    echo "Log saved to ${log_file}"
  else
    status=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed ${script} (exit ${status})" | tee -a "$log_file"
    echo "Log saved to ${log_file}"
    exit "$status"
  fi
done
