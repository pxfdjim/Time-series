#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/result/run_all_mindts_logs/$(date '+%Y%m%d_%H%M%S')"

# Experiment settings. Edit these defaults, or override them from the command line.
# Usage:
#   bash scripts/run_all_mindts.sh [save_root] [recon_loss_type] [recon_logvar_min] [recon_logvar_max] [max_parallel]
MINDTS_SAVE_ROOT="${1:-${MINDTS_SAVE_ROOT:-}}"
MINDTS_RECON_LOSS_TYPE="${2:-${MINDTS_RECON_LOSS_TYPE:-mse}}"
MINDTS_RECON_LOGVAR_MIN="${3:-${MINDTS_RECON_LOGVAR_MIN:--6.0}}"
MINDTS_RECON_LOGVAR_MAX="${4:-${MINDTS_RECON_LOGVAR_MAX:-2.0}}"
MINDTS_MAX_PARALLEL="${5:-${MINDTS_MAX_PARALLEL:-6}}"

MINDTS_ALIGN_LOSS_TYPE="${MINDTS_ALIGN_LOSS_TYPE:-text_gaussian_nll}"
CROSS_VIEW_DIRECTION="prompt_feature_query_semantic_features"
RECONSTRUCTION_DIRECTION="reconstruction_patch_features_then_llm_features"


MINDTS_GPUS="${MINDTS_GPUS:-0 1}"

MINDTS_RECON_LOSS_TYPE="$(printf '%s' "$MINDTS_RECON_LOSS_TYPE" | tr '[:upper:]' '[:lower:]')"

case "$MINDTS_RECON_LOSS_TYPE" in
  mse|gaussian_nll) ;;
  *)
    echo "Invalid MINDTS_RECON_LOSS_TYPE=${MINDTS_RECON_LOSS_TYPE}. Use mse or gaussian_nll." >&2
    exit 1
    ;;
esac

case "$MINDTS_ALIGN_LOSS_TYPE" in
  text_gaussian_nll|none) ;;
  *)
    echo "Invalid MINDTS_ALIGN_LOSS_TYPE=${MINDTS_ALIGN_LOSS_TYPE}." >&2
    echo "Use text_gaussian_nll or none." >&2
    exit 1
    ;;
esac

case "$MINDTS_MAX_PARALLEL" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_MAX_PARALLEL=${MINDTS_MAX_PARALLEL}. Use a positive integer." >&2
    exit 1
    ;;
esac

if [[ -z "$MINDTS_GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    gpu_count="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
  fi

  if (( gpu_count >= 2 )); then
    MINDTS_GPUS="0 1"
  elif (( gpu_count == 1 )); then
    MINDTS_GPUS="0"
  fi
fi

if [[ -n "$MINDTS_GPUS" ]]; then
  MINDTS_GPU_CLI_ARGS="--gpus ${MINDTS_GPUS}"
else
  MINDTS_GPU_CLI_ARGS=""
fi

if [[ -z "$MINDTS_SAVE_ROOT" ]]; then
  feature_tags=("${MINDTS_ALIGN_LOSS_TYPE}" "exchange_recon_text")
  if [[ "$MINDTS_RECON_LOSS_TYPE" == "gaussian_nll" ]]; then
    feature_tags+=("recon_gaussian_nll")
  fi
  MINDTS_SAVE_ROOT="label_$(IFS=_; printf '%s' "${feature_tags[*]}")"
fi

MINDTS_MODEL_HYPER_PARAM_OVERRIDES="{\"align_loss_type\": \"${MINDTS_ALIGN_LOSS_TYPE}\", \"recon_loss_type\": \"${MINDTS_RECON_LOSS_TYPE}\", \"recon_logvar_min\": ${MINDTS_RECON_LOGVAR_MIN}, \"recon_logvar_max\": ${MINDTS_RECON_LOGVAR_MAX}}"
export MINDTS_SAVE_ROOT
export MINDTS_MODEL_HYPER_PARAM_OVERRIDES
export MINDTS_GPU_CLI_ARGS
mkdir -p "$LOG_DIR"
echo "Logs: ${LOG_DIR}"
echo "Save root: result/${MINDTS_SAVE_ROOT}"
echo "GPU args: ${MINDTS_GPU_CLI_ARGS:-<none>}"
echo "Reconstruction loss: ${MINDTS_RECON_LOSS_TYPE} [logvar_min=${MINDTS_RECON_LOGVAR_MIN}, logvar_max=${MINDTS_RECON_LOGVAR_MAX}]"
echo "Cross-view direction: ${CROSS_VIEW_DIRECTION}"
echo "Reconstruction direction: ${RECONSTRUCTION_DIRECTION}"
echo "Max parallel datasets: ${MINDTS_MAX_PARALLEL}"
echo "Model overrides: ${MINDTS_MODEL_HYPER_PARAM_OVERRIDES}"

scripts=(
  "scripts/univariate_detection/detect_label/KR_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/Environment_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/MDT_script/MindTS.sh"
  "scripts/multivariate_detection/detect_label/Energy_script/MindTS.sh"
  "scripts/univariate_detection/detect_label/EWJ_script/MindTS.sh"
  "scripts/multivariate_detection/detect_label/Weather_script/MindTS.sh"
)

run_one_dataset() {
  local script="$1"
  local dataset_name
  local log_file
  local status

  dataset_name="$(basename "$(dirname "$script")" "_script")"
  log_file="${LOG_DIR}/${dataset_name}.txt"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${script}"
  if bash "$script" >"$log_file" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${script}"
    echo "Log saved to ${log_file}"
  else
    status=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed ${script} (exit ${status})" | tee -a "$log_file"
    echo "Log saved to ${log_file}"
    return "$status"
  fi
}

failed=0
for script in "${scripts[@]}"; do
  while (( $(jobs -pr | wc -l) >= MINDTS_MAX_PARALLEL )); do
    if ! wait -n; then
      failed=1
      break 2
    fi
  done

  run_one_dataset "$script" &
done

if (( failed )); then
  echo "A dataset run failed. Stopping remaining jobs..."
  jobs -pr | xargs -r kill
  wait || true
  exit 1
fi

while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    failed=1
  fi
done

if (( failed )); then
  echo "One or more dataset runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All dataset runs finished. Logs saved to ${LOG_DIR}"
