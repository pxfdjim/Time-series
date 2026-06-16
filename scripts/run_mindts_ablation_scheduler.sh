#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MINDTS_CONDA_ENV="${MINDTS_CONDA_ENV:-mind}"
if [[ -n "$MINDTS_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate ${MINDTS_CONDA_ENV}, but conda was not found." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$MINDTS_CONDA_ENV"
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "The active python cannot import torch. Activate the MindTS environment first or set MINDTS_CONDA_ENV." >&2
  echo "python: $(command -v python)" >&2
  exit 1
fi

RUN_ID="${MINDTS_SCHEDULER_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${ROOT_DIR}/result/mindts_ablation_scheduler_logs/${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

ALIGN_LOSS_TYPE="${MINDTS_ALIGN_LOSS_TYPE:-text_gaussian_nll}"
RECON_LOSS_TYPE="${MINDTS_RECON_LOSS_TYPE:-mse}"
RECON_LOGVAR_MIN="${MINDTS_RECON_LOGVAR_MIN:--6.0}"
RECON_LOGVAR_MAX="${MINDTS_RECON_LOGVAR_MAX:-2.0}"
USE_FREQUENCY_BRANCH="${MINDTS_USE_FREQUENCY_BRANCH:-false}"
USE_INFORMATION_CONDENSER="${MINDTS_USE_INFORMATION_CONDENSER:-false}"
USE_DE_STATIONARY_CROSS_VIEW="false"
RECONSTRUCTION_EXCHANGE_TEXT_FEATURES="true"
FREQUENCY_KEEP_MODES="${MINDTS_FREQUENCY_KEEP_MODES:-4}"
TIME_FREQ_ALIGN_WEIGHT="${MINDTS_TIME_FREQ_ALIGN_WEIGHT:-0.2}"

GPU_MEMORY_READY_PCT="${MINDTS_SCHEDULER_GPU_MEMORY_READY_PCT:-80}"
GPU_UTIL_READY_PCT="${MINDTS_SCHEDULER_GPU_UTIL_READY_PCT:-85}"
MAX_TASKS_PER_GPU="${MINDTS_SCHEDULER_MAX_TASKS_PER_GPU:-1}"
POLL_SECONDS="${MINDTS_SCHEDULER_POLL_SECONDS:-20}"
DRY_RUN="${MINDTS_SCHEDULER_DRY_RUN:-false}"
SKIP_DONE="${MINDTS_SCHEDULER_SKIP_DONE:-true}"
STOP_ON_FAILURE="${MINDTS_SCHEDULER_STOP_ON_FAILURE:-false}"

normalize_bool() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

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

USE_FREQUENCY_BRANCH="$(normalize_bool "$USE_FREQUENCY_BRANCH")"
USE_INFORMATION_CONDENSER="$(normalize_bool "$USE_INFORMATION_CONDENSER")"
DRY_RUN="$(normalize_bool "$DRY_RUN")"
SKIP_DONE="$(normalize_bool "$SKIP_DONE")"
STOP_ON_FAILURE="$(normalize_bool "$STOP_ON_FAILURE")"
RECON_LOSS_TYPE="$(normalize_bool "$RECON_LOSS_TYPE")"

validate_bool "MINDTS_USE_FREQUENCY_BRANCH" "$USE_FREQUENCY_BRANCH"
validate_bool "MINDTS_USE_INFORMATION_CONDENSER" "$USE_INFORMATION_CONDENSER"
validate_bool "MINDTS_SCHEDULER_DRY_RUN" "$DRY_RUN"
validate_bool "MINDTS_SCHEDULER_SKIP_DONE" "$SKIP_DONE"
validate_bool "MINDTS_SCHEDULER_STOP_ON_FAILURE" "$STOP_ON_FAILURE"

case "$RECON_LOSS_TYPE" in
  mse|gaussian_nll) ;;
  *)
    echo "Invalid MINDTS_RECON_LOSS_TYPE=${RECON_LOSS_TYPE}. Use mse or gaussian_nll." >&2
    exit 1
    ;;
esac

case "$ALIGN_LOSS_TYPE" in
  contrastive|text_gaussian_nll|symmetric_gaussian_kl|none) ;;
  *)
    echo "Invalid MINDTS_ALIGN_LOSS_TYPE=${ALIGN_LOSS_TYPE}." >&2
    exit 1
    ;;
esac

case "$MAX_TASKS_PER_GPU" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_SCHEDULER_MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$POLL_SECONDS" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_SCHEDULER_POLL_SECONDS=${POLL_SECONDS}. Use a positive integer." >&2
    exit 1
    ;;
esac

if [[ -n "${MINDTS_SCHEDULER_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${MINDTS_SCHEDULER_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  GPUS=("")
fi

if (( ${#GPUS[@]} == 0 )); then
  echo "No GPU candidates found. Set MINDTS_SCHEDULER_GPUS, e.g. '0 1'." >&2
  exit 1
fi

declare -A DATASET_SCRIPT=(
  [Energy]="scripts/multivariate_detection/detect_label/Energy_script/MindTS.sh"
  [Weather]="scripts/multivariate_detection/detect_label/Weather_script/MindTS.sh"
  [EWJ]="scripts/univariate_detection/detect_label/EWJ_script/MindTS.sh"
  [Environment]="scripts/univariate_detection/detect_label/Environment_script/MindTS.sh"
  [MDT]="scripts/univariate_detection/detect_label/MDT_script/MindTS.sh"
  [KR]="scripts/univariate_detection/detect_label/KR_script/MindTS.sh"
)

default_datasets=(Energy Weather EWJ Environment MDT KR)
if [[ -n "${MINDTS_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${MINDTS_DATASETS}"
else
  DATASETS=("${default_datasets[@]}")
fi

for dataset in "${DATASETS[@]}"; do
  if [[ -z "${DATASET_SCRIPT[$dataset]:-}" ]]; then
    echo "Unknown dataset '${dataset}'. Supported: ${default_datasets[*]}" >&2
    exit 1
  fi
done

experiment_rows=(
  "label_text_gaussian_nll_exchange_recon_text_destationary|true|false"
  "label_text_gaussian_nll_exchange_text_exchange_recon_text_destationary|true|true"
  "label_text_gaussian_nll_exchange_recon_text|false|false"
  "label_text_gaussian_nll_exchange_text_exchange_recon_text|false|true"
)

TASKS=()
for row in "${experiment_rows[@]}"; do
  IFS='|' read -r save_root use_de_stationary exchange_text_features <<<"$row"
  for dataset in "${DATASETS[@]}"; do
    TASKS+=("${save_root}|${use_de_stationary}|${exchange_text_features}|${dataset}|${DATASET_SCRIPT[$dataset]}")
  done
done

task_id_for() {
  local save_root="$1"
  local dataset="$2"
  printf '%s__%s' "$save_root" "$dataset"
}

write_config_snapshot() {
  local config_file="$1"
  local save_root="$2"
  local dataset="$3"
  local gpu="$4"
  local script="$5"
  local use_de_stationary="$6"
  local exchange_text_features="$7"
  local overrides="$8"

  mkdir -p "$(dirname "$config_file")"
  {
    printf '{\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "save_root": "%s",\n' "$save_root"
    printf '  "dataset": "%s",\n' "$dataset"
    printf '  "script": "%s",\n' "$script"
    printf '  "gpu": "%s",\n' "$gpu"
    printf '  "align_loss_type": "%s",\n' "$ALIGN_LOSS_TYPE"
    printf '  "recon_loss_type": "%s",\n' "$RECON_LOSS_TYPE"
    printf '  "use_de_stationary": %s,\n' "$use_de_stationary"
    printf '  "use_de_stationary_cross_view": %s,\n' "$USE_DE_STATIONARY_CROSS_VIEW"
    printf '  "exchange_text_features": %s,\n' "$exchange_text_features"
    printf '  "reconstruction_exchange_text_features": %s,\n' "$RECONSTRUCTION_EXCHANGE_TEXT_FEATURES"
    printf '  "use_frequency_branch": %s,\n' "$USE_FREQUENCY_BRANCH"
    printf '  "use_information_condenser": %s,\n' "$USE_INFORMATION_CONDENSER"
    printf '  "model_hyper_param_overrides": %s\n' "$overrides"
    printf '}\n'
  } >"$config_file"
}

build_overrides() {
  local use_de_stationary="$1"
  local exchange_text_features="$2"
  printf '{"use_information_condenser": %s, "align_loss_type": "%s", "recon_loss_type": "%s", "recon_logvar_min": %s, "recon_logvar_max": %s, "use_frequency_branch": %s, "frequency_keep_modes": %s, "time_freq_align_weight": %s, "exchange_text_features": %s, "reconstruction_exchange_text_features": %s, "use_de_stationary": %s, "use_de_stationary_cross_view": %s}' \
    "$USE_INFORMATION_CONDENSER" \
    "$ALIGN_LOSS_TYPE" \
    "$RECON_LOSS_TYPE" \
    "$RECON_LOGVAR_MIN" \
    "$RECON_LOGVAR_MAX" \
    "$USE_FREQUENCY_BRANCH" \
    "$FREQUENCY_KEEP_MODES" \
    "$TIME_FREQ_ALIGN_WEIGHT" \
    "$exchange_text_features" \
    "$RECONSTRUCTION_EXCHANGE_TEXT_FEATURES" \
    "$use_de_stationary" \
    "$USE_DE_STATIONARY_CROSS_VIEW"
}

gpu_ready() {
  local gpu="$1"
  if [[ -z "$gpu" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local line used total util mem_pct
  line="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$line" ]]; then
    return 1
  fi
  IFS=',' read -r used total util <<<"$line"
  used="${used//[[:space:]]/}"
  total="${total//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  if [[ -z "$used" || -z "$total" || "$total" == "0" ]]; then
    return 1
  fi
  mem_pct=$(( used * 100 / total ))
  (( mem_pct <= GPU_MEMORY_READY_PCT && util <= GPU_UTIL_READY_PCT ))
}

declare -A GPU_ACTIVE=()
for gpu in "${GPUS[@]}"; do
  GPU_ACTIVE["$gpu"]=0
done

declare -A PID_STATUS_FILE=()
declare -A PID_GPU=()
declare -A PID_TASK=()
failed=0
completed=0
started=0
total=${#TASKS[@]}

log_scheduler() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" | tee -a "${LOG_DIR}/scheduler.log"
}

poll_finished() {
  local pid status_file gpu task_name status wait_status active_count
  for pid in "${!PID_STATUS_FILE[@]}"; do
    status_file="${PID_STATUS_FILE[$pid]}"
    if [[ ! -f "$status_file" ]]; then
      continue
    fi
    gpu="${PID_GPU[$pid]}"
    task_name="${PID_TASK[$pid]}"
    status="$(cat "$status_file")"
    set +e
    wait "$pid"
    wait_status=$?
    set -e
    active_count="${GPU_ACTIVE[$gpu]:-0}"
    if (( active_count > 0 )); then
      GPU_ACTIVE["$gpu"]=$(( active_count - 1 ))
    else
      GPU_ACTIVE["$gpu"]=0
    fi
    unset 'PID_STATUS_FILE[$pid]' 'PID_GPU[$pid]' 'PID_TASK[$pid]'
    completed=$(( completed + 1 ))
    if [[ "$status" == "0" ]]; then
      log_scheduler "DONE ${task_name} on GPU ${gpu} (${completed}/${total})"
    else
      failed=1
      log_scheduler "FAILED ${task_name} on GPU ${gpu} with exit ${status}; wait_status=${wait_status} (${completed}/${total})"
    fi
  done
}

launch_task() {
  local task="$1"
  local gpu="$2"
  local save_root use_de_stationary exchange_text_features dataset script
  IFS='|' read -r save_root use_de_stationary exchange_text_features dataset script <<<"$task"

  local task_id task_dir config_file overrides log_file status_file done_file failed_file gpu_args
  task_id="$(task_id_for "$save_root" "$dataset")"
  task_dir="${ROOT_DIR}/result/${save_root}/${dataset}"
  config_file="${task_dir}/experiment_config.json"
  overrides="$(build_overrides "$use_de_stationary" "$exchange_text_features")"
  log_file="${LOG_DIR}/${task_id}.log"
  status_file="${STATE_DIR}/${task_id}.exit"
  done_file="${STATE_DIR}/${task_id}.done"
  failed_file="${STATE_DIR}/${task_id}.failed"
  rm -f "$status_file" "$failed_file"
  mkdir -p "$task_dir"
  write_config_snapshot "$config_file" "$save_root" "$dataset" "$gpu" "$script" "$use_de_stationary" "$exchange_text_features" "$overrides"

  if [[ -n "$gpu" ]]; then
    gpu_args="--gpus ${gpu}"
  else
    gpu_args=""
  fi

  log_scheduler "START ${task_id} on GPU ${gpu:-none}; result/${save_root}/${dataset}"
  (
    set +e
    {
      printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$task_id"
      printf 'GPU: %s\n' "${gpu:-none}"
      printf 'Save root: %s\n' "$save_root"
      printf 'Dataset: %s\n' "$dataset"
      printf 'Overrides: %s\n' "$overrides"
      printf 'Script: %s\n\n' "$script"
    } >>"$log_file"

    export MINDTS_SAVE_ROOT="$save_root"
    export MINDTS_MODEL_HYPER_PARAM_OVERRIDES="$overrides"
    export MINDTS_GPU_CLI_ARGS="$gpu_args"
    bash "$script" >>"$log_file" 2>&1
    status=$?
    printf '[%s] EXIT %s status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$task_id" "$status" >>"$log_file"
    printf '%s\n' "$status" >"$status_file"
    if [[ "$status" == "0" ]]; then
      date '+%Y-%m-%d %H:%M:%S' >"$done_file"
    else
      date '+%Y-%m-%d %H:%M:%S' >"$failed_file"
    fi
    exit "$status"
  ) &

  local pid=$!
  PID_STATUS_FILE["$pid"]="$status_file"
  PID_GPU["$pid"]="$gpu"
  PID_TASK["$pid"]="$task_id"
  GPU_ACTIVE["$gpu"]=$(( GPU_ACTIVE["$gpu"] + 1 ))
  started=$(( started + 1 ))
}

log_scheduler "Run id: ${RUN_ID}"
log_scheduler "Log dir: ${LOG_DIR}"
log_scheduler "Conda env: ${MINDTS_CONDA_ENV:-<none>}; python: $(command -v python)"
log_scheduler "GPUs: ${GPUS[*]}"
log_scheduler "Tasks: ${total}; datasets: ${DATASETS[*]}"
log_scheduler "GPU ready thresholds: memory<=${GPU_MEMORY_READY_PCT}%, util<=${GPU_UTIL_READY_PCT}%; max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"

if [[ "$DRY_RUN" == "true" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r save_root use_de_stationary exchange_text_features dataset script <<<"$task"
    printf '%s | dataset=%s | use_de_stationary=%s | exchange_text_features=%s | reconstruction_exchange_text_features=%s | cross_view_destationary=%s\n' \
      "$save_root" "$dataset" "$use_de_stationary" "$exchange_text_features" "$RECONSTRUCTION_EXCHANGE_TEXT_FEATURES" "$USE_DE_STATIONARY_CROSS_VIEW"
  done | tee -a "${LOG_DIR}/scheduler.log"
  exit 0
fi

next_task=0
while (( completed < total )); do
  poll_finished
  if (( failed )) && [[ "$STOP_ON_FAILURE" == "true" ]]; then
    log_scheduler "Stop-on-failure is enabled; no new tasks will be launched."
    break
  fi

  launched_this_round=0
  while (( next_task < total )); do
    task="${TASKS[$next_task]}"
    IFS='|' read -r save_root use_de_stationary exchange_text_features dataset script <<<"$task"
    task_id="$(task_id_for "$save_root" "$dataset")"
    done_file="${STATE_DIR}/${task_id}.done"

    if [[ "$SKIP_DONE" == "true" && -f "$done_file" ]]; then
      completed=$(( completed + 1 ))
      next_task=$(( next_task + 1 ))
      log_scheduler "SKIP already done ${task_id} (${completed}/${total})"
      continue
    fi

    selected_gpu=""
    for gpu in "${GPUS[@]}"; do
      if (( GPU_ACTIVE["$gpu"] >= MAX_TASKS_PER_GPU )); then
        continue
      fi
      if gpu_ready "$gpu"; then
        selected_gpu="$gpu"
        break
      fi
    done

    if [[ -z "$selected_gpu" && "${GPUS[0]}" != "" ]]; then
      break
    fi

    launch_task "$task" "$selected_gpu"
    next_task=$(( next_task + 1 ))
    launched_this_round=1
  done

  if (( completed >= total )); then
    break
  fi

  if (( launched_this_round == 0 )); then
    sleep "$POLL_SECONDS"
  fi
done

poll_finished

if (( failed )); then
  log_scheduler "One or more tasks failed. Check ${LOG_DIR}."
  exit 1
fi

log_scheduler "All tasks finished successfully."
