#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/result/run_all_mindts_logs/$(date '+%Y%m%d_%H%M%S')"

# Experiment switches. Edit these defaults, or override them from the command line.
MINDTS_SAVE_ROOT="${1:-${MINDTS_SAVE_ROOT:-label_text_gaussian_nll}}"
MINDTS_USE_INFORMATION_CONDENSER="${MINDTS_USE_INFORMATION_CONDENSER:-false}"
MINDTS_ALIGN_LOSS_TYPE="${MINDTS_ALIGN_LOSS_TYPE:-text_gaussian_nll}"

MINDTS_USE_INFORMATION_CONDENSER="$(printf '%s' "$MINDTS_USE_INFORMATION_CONDENSER" | tr '[:upper:]' '[:lower:]')"
case "$MINDTS_USE_INFORMATION_CONDENSER" in
  true|false) ;;
  *)
    echo "Invalid MINDTS_USE_INFORMATION_CONDENSER=${MINDTS_USE_INFORMATION_CONDENSER}. Use true or false." >&2
    exit 1
    ;;
esac

case "$MINDTS_ALIGN_LOSS_TYPE" in
  contrastive|text_gaussian_nll|symmetric_gaussian_kl|none) ;;
  *)
    echo "Invalid MINDTS_ALIGN_LOSS_TYPE=${MINDTS_ALIGN_LOSS_TYPE}." >&2
    echo "Use contrastive, text_gaussian_nll, symmetric_gaussian_kl, or none." >&2
    exit 1
    ;;
esac

MINDTS_MODEL_HYPER_PARAM_OVERRIDES="{\"use_information_condenser\": ${MINDTS_USE_INFORMATION_CONDENSER}, \"align_loss_type\": \"${MINDTS_ALIGN_LOSS_TYPE}\"}"
export MINDTS_SAVE_ROOT
export MINDTS_MODEL_HYPER_PARAM_OVERRIDES
mkdir -p "$LOG_DIR"
echo "Logs: ${LOG_DIR}"
echo "Save root: result/${MINDTS_SAVE_ROOT}"
echo "Model overrides: ${MINDTS_MODEL_HYPER_PARAM_OVERRIDES}"

scripts=(
  "scripts/univariate_detection/detect_label/KR_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/MDT_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/EWJ_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/Environment_script/MindTS.sh"
  # "scripts/multivariate_detection/detect_label/Energy_script/MindTS.sh"
  # "scripts/multivariate_detection/detect_label/Weather_script/MindTS.sh"
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
