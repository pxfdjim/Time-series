#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/result/run_all_mindts_logs/$(date '+%Y%m%d_%H%M%S')"

# Experiment switches. Edit these defaults, or override them from the command line.
# Usage:
#   bash scripts/run_all_mindts.sh [save_root] [recon_loss_type] [recon_logvar_min] [recon_logvar_max] [max_parallel] [use_frequency_branch] [exchange_text_features] [reconstruction_exchange_text_features] [use_de_stationary]
# Frequency branch knobs can also be set with:
#   MINDTS_USE_FREQUENCY_BRANCH=true|false
#   MINDTS_FREQUENCY_KEEP_MODES=4
#   MINDTS_TIME_FREQ_ALIGN_WEIGHT=0.2
MINDTS_SAVE_ROOT="${1:-${MINDTS_SAVE_ROOT:-}}"
MINDTS_RECON_LOSS_TYPE="${2:-${MINDTS_RECON_LOSS_TYPE:-mse}}"
MINDTS_RECON_LOGVAR_MIN="${3:-${MINDTS_RECON_LOGVAR_MIN:--6.0}}"
MINDTS_RECON_LOGVAR_MAX="${4:-${MINDTS_RECON_LOGVAR_MAX:-2.0}}"
MINDTS_MAX_PARALLEL="${5:-${MINDTS_MAX_PARALLEL:-6}}"
MINDTS_USE_FREQUENCY_BRANCH="${6:-${MINDTS_USE_FREQUENCY_BRANCH:-false}}"
MINDTS_EXCHANGE_TEXT_FEATURES="${7:-${MINDTS_EXCHANGE_TEXT_FEATURES:-true}}"
MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES="${8:-${MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES:-${MINDTS_EXCHANGE_TEXT_FEATURES}}}"
MINDTS_USE_DE_STATIONARY="${9:-${MINDTS_USE_DE_STATIONARY:-true}}"
MINDTS_USE_DE_STATIONARY_CROSS_VIEW="${MINDTS_USE_DE_STATIONARY_CROSS_VIEW:-false}"
MINDTS_FREQUENCY_KEEP_MODES="${MINDTS_FREQUENCY_KEEP_MODES:-4}"
MINDTS_TIME_FREQ_ALIGN_WEIGHT="${MINDTS_TIME_FREQ_ALIGN_WEIGHT:-0.2}"
MINDTS_USE_INFORMATION_CONDENSER="${MINDTS_USE_INFORMATION_CONDENSER:-false}"


MINDTS_ALIGN_LOSS_TYPE="${MINDTS_ALIGN_LOSS_TYPE:-text_gaussian_nll}"


MINDTS_GPUS="${MINDTS_GPUS:-0 1}"

MINDTS_RECON_LOSS_TYPE="$(printf '%s' "$MINDTS_RECON_LOSS_TYPE" | tr '[:upper:]' '[:lower:]')"
MINDTS_USE_FREQUENCY_BRANCH="$(printf '%s' "$MINDTS_USE_FREQUENCY_BRANCH" | tr '[:upper:]' '[:lower:]')"
MINDTS_USE_INFORMATION_CONDENSER="$(printf '%s' "$MINDTS_USE_INFORMATION_CONDENSER" | tr '[:upper:]' '[:lower:]')"
MINDTS_EXCHANGE_TEXT_FEATURES="$(printf '%s' "$MINDTS_EXCHANGE_TEXT_FEATURES" | tr '[:upper:]' '[:lower:]')"
MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES="$(printf '%s' "$MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES" | tr '[:upper:]' '[:lower:]')"
MINDTS_USE_DE_STATIONARY="$(printf '%s' "$MINDTS_USE_DE_STATIONARY" | tr '[:upper:]' '[:lower:]')"
MINDTS_USE_DE_STATIONARY_CROSS_VIEW="$(printf '%s' "$MINDTS_USE_DE_STATIONARY_CROSS_VIEW" | tr '[:upper:]' '[:lower:]')"

validate_bool() {
  local name="$1"
  local value="$2"
  case "$value" in
    true|false) ;;
    *)
      echo "Invalid ${name}=${value}. Use true or false." >&2
      exit 1
      ;;
  esac
}

case "$MINDTS_RECON_LOSS_TYPE" in
  mse|gaussian_nll) ;;
  *)
    echo "Invalid MINDTS_RECON_LOSS_TYPE=${MINDTS_RECON_LOSS_TYPE}. Use mse or gaussian_nll." >&2
    exit 1
    ;;
esac

validate_bool "MINDTS_USE_INFORMATION_CONDENSER" "$MINDTS_USE_INFORMATION_CONDENSER"
validate_bool "MINDTS_USE_FREQUENCY_BRANCH" "$MINDTS_USE_FREQUENCY_BRANCH"
validate_bool "MINDTS_EXCHANGE_TEXT_FEATURES" "$MINDTS_EXCHANGE_TEXT_FEATURES"
validate_bool "MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES" "$MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES"
validate_bool "MINDTS_USE_DE_STATIONARY" "$MINDTS_USE_DE_STATIONARY"
validate_bool "MINDTS_USE_DE_STATIONARY_CROSS_VIEW" "$MINDTS_USE_DE_STATIONARY_CROSS_VIEW"

case "$MINDTS_ALIGN_LOSS_TYPE" in
  contrastive|text_gaussian_nll|symmetric_gaussian_kl|none) ;;
  *)
    echo "Invalid MINDTS_ALIGN_LOSS_TYPE=${MINDTS_ALIGN_LOSS_TYPE}." >&2
    echo "Use contrastive, text_gaussian_nll, symmetric_gaussian_kl, or none." >&2
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
  feature_tags=("${MINDTS_ALIGN_LOSS_TYPE}")
  if [[ "$MINDTS_RECON_LOSS_TYPE" == "gaussian_nll" ]]; then
    feature_tags+=("recon_gaussian_nll")
  fi
  if [[ "$MINDTS_USE_FREQUENCY_BRANCH" == "true" ]]; then
    feature_tags+=("frequency")
  fi
  if [[ "$MINDTS_USE_INFORMATION_CONDENSER" == "true" ]]; then
    feature_tags+=("condenser")
  fi
  if [[ "$MINDTS_EXCHANGE_TEXT_FEATURES" == "true" ]]; then
    feature_tags+=("exchange_text")
  fi
  if [[ "$MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES" == "true" ]]; then
    feature_tags+=("exchange_recon_text")
  fi
  if [[ "$MINDTS_USE_DE_STATIONARY" == "true" ]]; then
    feature_tags+=("destationary")
  fi
  if [[ "$MINDTS_USE_DE_STATIONARY_CROSS_VIEW" == "true" ]]; then
    feature_tags+=("cross_destationary")
  fi
  MINDTS_SAVE_ROOT="label_$(IFS=_; printf '%s' "${feature_tags[*]}")"
fi

MINDTS_MODEL_HYPER_PARAM_OVERRIDES="{\"use_information_condenser\": ${MINDTS_USE_INFORMATION_CONDENSER}, \"align_loss_type\": \"${MINDTS_ALIGN_LOSS_TYPE}\", \"recon_loss_type\": \"${MINDTS_RECON_LOSS_TYPE}\", \"recon_logvar_min\": ${MINDTS_RECON_LOGVAR_MIN}, \"recon_logvar_max\": ${MINDTS_RECON_LOGVAR_MAX}, \"use_frequency_branch\": ${MINDTS_USE_FREQUENCY_BRANCH}, \"frequency_keep_modes\": ${MINDTS_FREQUENCY_KEEP_MODES}, \"time_freq_align_weight\": ${MINDTS_TIME_FREQ_ALIGN_WEIGHT}, \"exchange_text_features\": ${MINDTS_EXCHANGE_TEXT_FEATURES}, \"reconstruction_exchange_text_features\": ${MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES}, \"use_de_stationary\": ${MINDTS_USE_DE_STATIONARY}, \"use_de_stationary_cross_view\": ${MINDTS_USE_DE_STATIONARY_CROSS_VIEW}}"
export MINDTS_SAVE_ROOT
export MINDTS_MODEL_HYPER_PARAM_OVERRIDES
export MINDTS_GPU_CLI_ARGS
mkdir -p "$LOG_DIR"
echo "Logs: ${LOG_DIR}"
echo "Save root: result/${MINDTS_SAVE_ROOT}"
echo "GPU args: ${MINDTS_GPU_CLI_ARGS:-<none>}"
echo "Reconstruction loss: ${MINDTS_RECON_LOSS_TYPE} [logvar_min=${MINDTS_RECON_LOGVAR_MIN}, logvar_max=${MINDTS_RECON_LOGVAR_MAX}]"
echo "Frequency branch: ${MINDTS_USE_FREQUENCY_BRANCH} [keep_modes=${MINDTS_FREQUENCY_KEEP_MODES}, time_freq_align_weight=${MINDTS_TIME_FREQ_ALIGN_WEIGHT}]"
echo "Exchange text features: ${MINDTS_EXCHANGE_TEXT_FEATURES}"
echo "Reconstruction exchange text features: ${MINDTS_RECONSTRUCTION_EXCHANGE_TEXT_FEATURES}"
echo "De-stationary time encoder: ${MINDTS_USE_DE_STATIONARY}"
echo "De-stationary cross-view: ${MINDTS_USE_DE_STATIONARY_CROSS_VIEW}"
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
