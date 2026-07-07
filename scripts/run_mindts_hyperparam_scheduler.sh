#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MINDTS_HPARAM_CONDA_ENV="${MINDTS_HPARAM_CONDA_ENV:-mind_qwen3}"
if [[ -n "$MINDTS_HPARAM_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate ${MINDTS_HPARAM_CONDA_ENV}, but conda was not found." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$MINDTS_HPARAM_CONDA_ENV"
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "The active python cannot import torch. Activate the MindTS environment first or set MINDTS_HPARAM_CONDA_ENV." >&2
  echo "python: $(command -v python)" >&2
  exit 1
fi

RUN_ID="${MINDTS_HPARAM_SCHEDULER_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${ROOT_DIR}/result/mindts_hyperparam_scheduler_logs/${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

ALIGN_LOSS_TYPE="text_gaussian_nll"
RECON_LOSS_TYPE="mse"
RECON_LOGVAR_MIN="-6.0"
RECON_LOGVAR_MAX="2.0"
LLM_MODEL_TAG="qwen3_1p7b"
LLM_MODEL_PATH="models/Qwen3-1.7B"
LLM_MODEL_NAME="Qwen3-1.7B"

GPU_MEMORY_READY_PCT="${MINDTS_HPARAM_SCHEDULER_GPU_MEMORY_READY_PCT:-95}"
GPU_UTIL_READY_PCT="${MINDTS_HPARAM_SCHEDULER_GPU_UTIL_READY_PCT:-100}"
MAX_TASKS_PER_GPU="${MINDTS_HPARAM_SCHEDULER_MAX_TASKS_PER_GPU:-3}"
POLL_SECONDS="${MINDTS_HPARAM_SCHEDULER_POLL_SECONDS:-20}"
ONE_LAUNCH_PER_GPU_PER_POLL="${MINDTS_HPARAM_SCHEDULER_ONE_LAUNCH_PER_GPU_PER_POLL:-false}"
DRY_RUN="${MINDTS_HPARAM_SCHEDULER_DRY_RUN:-false}"
SKIP_DONE="${MINDTS_HPARAM_SCHEDULER_SKIP_DONE:-true}"
STOP_ON_FAILURE="${MINDTS_HPARAM_SCHEDULER_STOP_ON_FAILURE:-false}"
NUM_EPOCHS="${MINDTS_HPARAM_NUM_EPOCHS:-5}"

MASK_RATIO_VALUES="${MINDTS_HPARAM_MASK_RATIO_VALUES:-0.1 0.2 0.3 0.4 0.5 0.6}"
LAMDA1_VALUES="${MINDTS_HPARAM_LAMDA1_VALUES:-0 0.1 0.5 1.0 2.0}"

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

slug_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//- /m}"
  value="${value//-/m}"
  printf '%s' "$value"
}

DRY_RUN="$(normalize_bool "$DRY_RUN")"
SKIP_DONE="$(normalize_bool "$SKIP_DONE")"
STOP_ON_FAILURE="$(normalize_bool "$STOP_ON_FAILURE")"
ONE_LAUNCH_PER_GPU_PER_POLL="$(normalize_bool "$ONE_LAUNCH_PER_GPU_PER_POLL")"
validate_bool "MINDTS_HPARAM_SCHEDULER_DRY_RUN" "$DRY_RUN"
validate_bool "MINDTS_HPARAM_SCHEDULER_SKIP_DONE" "$SKIP_DONE"
validate_bool "MINDTS_HPARAM_SCHEDULER_STOP_ON_FAILURE" "$STOP_ON_FAILURE"
validate_bool "MINDTS_HPARAM_SCHEDULER_ONE_LAUNCH_PER_GPU_PER_POLL" "$ONE_LAUNCH_PER_GPU_PER_POLL"

case "$MAX_TASKS_PER_GPU" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_HPARAM_SCHEDULER_MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$POLL_SECONDS" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_HPARAM_SCHEDULER_POLL_SECONDS=${POLL_SECONDS}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$NUM_EPOCHS" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_HPARAM_NUM_EPOCHS=${NUM_EPOCHS}. Use a positive integer." >&2
    exit 1
    ;;
esac

if [[ -n "${MINDTS_HPARAM_SCHEDULER_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${MINDTS_HPARAM_SCHEDULER_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  GPUS=("")
fi

if (( ${#GPUS[@]} == 0 )); then
  echo "No GPU candidates found. Set MINDTS_HPARAM_SCHEDULER_GPUS, e.g. '0 1'." >&2
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
if [[ -n "${MINDTS_HPARAM_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${MINDTS_HPARAM_DATASETS}"
else
  DATASETS=("${default_datasets[@]}")
fi

for dataset in "${DATASETS[@]}"; do
  if [[ -z "${DATASET_SCRIPT[$dataset]:-}" ]]; then
    echo "Unknown dataset '${dataset}'. Supported: ${default_datasets[*]}" >&2
    exit 1
  fi
done

if [[ ! -d "$LLM_MODEL_PATH" ]]; then
  echo "Missing LLM model path: ${LLM_MODEL_PATH}" >&2
  exit 1
fi

log_scheduler() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" | tee -a "${LOG_DIR}/scheduler.log"
}

TASKS=()
for value in $MASK_RATIO_VALUES; do
  save_root="label_hparam_${LLM_MODEL_TAG}_mask_ratio_m$(slug_float "$value")"
  for dataset in "${DATASETS[@]}"; do
    TASKS+=("mask_ratio|${value}|${save_root}|${dataset}|${DATASET_SCRIPT[$dataset]}")
  done
done
for value in $LAMDA1_VALUES; do
  save_root="label_hparam_${LLM_MODEL_TAG}_lamda1_l$(slug_float "$value")"
  for dataset in "${DATASETS[@]}"; do
    TASKS+=("lamda1|${value}|${save_root}|${dataset}|${DATASET_SCRIPT[$dataset]}")
  done
done

task_id_for() {
  local hparam="$1"
  local value="$2"
  local dataset="$3"
  printf '%s_%s__%s' "$hparam" "$(slug_float "$value")" "$dataset"
}

build_overrides() {
  local hparam="$1"
  local value="$2"
  python - "$hparam" "$value" "$ALIGN_LOSS_TYPE" "$RECON_LOSS_TYPE" "$RECON_LOGVAR_MIN" "$RECON_LOGVAR_MAX" "$LLM_MODEL_PATH" "$LLM_MODEL_NAME" "$NUM_EPOCHS" <<'PY'
import json
import sys

hparam, value = sys.argv[1], float(sys.argv[2])
overrides = {
    "use_information_condenser": False,
    "align_loss_type": sys.argv[3],
    "recon_loss_type": sys.argv[4],
    "recon_logvar_min": float(sys.argv[5]),
    "recon_logvar_max": float(sys.argv[6]),
    "use_frequency_branch": False,
    "frequency_keep_modes": 4,
    "time_freq_align_weight": 0.2,
    "exchange_text_features": False,
    "reconstruction_exchange_text_features": True,
    "use_de_stationary": False,
    "use_de_stationary_cross_view": False,
    "llm_model_path": sys.argv[7],
    "llm_model_name": sys.argv[8],
    "num_epochs": int(sys.argv[9]),
}
if hparam == "mask_ratio":
    overrides["mask_ratio"] = value
    overrides["lamda1"] = 1.0
elif hparam == "lamda1":
    overrides["lamda1"] = value
else:
    raise SystemExit(f"unknown hparam: {hparam}")
print(json.dumps(overrides, separators=(",", ":")))
PY
}

write_config_snapshot() {
  local config_file="$1"
  local save_root="$2"
  local hparam="$3"
  local value="$4"
  local dataset="$5"
  local gpu="$6"
  local script="$7"
  local overrides="$8"

  mkdir -p "$(dirname "$config_file")"
  {
    printf '{\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "save_root": "%s",\n' "$save_root"
    printf '  "dataset": "%s",\n' "$dataset"
    printf '  "script": "%s",\n' "$script"
    printf '  "gpu": "%s",\n' "$gpu"
    printf '  "hparam": "%s",\n' "$hparam"
    printf '  "hparam_value": %s,\n' "$value"
    printf '  "llm_model_tag": "%s",\n' "$LLM_MODEL_TAG"
    printf '  "llm_model_name": "%s",\n' "$LLM_MODEL_NAME"
    printf '  "llm_model_path": "%s",\n' "$LLM_MODEL_PATH"
    printf '  "align_loss_type": "%s",\n' "$ALIGN_LOSS_TYPE"
    printf '  "recon_loss_type": "%s",\n' "$RECON_LOSS_TYPE"
    printf '  "model_hyper_param_overrides": %s\n' "$overrides"
    printf '}\n'
  } >"$config_file"
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
  local hparam value save_root dataset script
  IFS='|' read -r hparam value save_root dataset script <<<"$task"

  local task_id task_dir config_file overrides log_file status_file done_file failed_file gpu_args
  task_id="$(task_id_for "$hparam" "$value" "$dataset")"
  task_dir="${ROOT_DIR}/result/${save_root}/${dataset}"
  config_file="${task_dir}/experiment_config.json"
  overrides="$(build_overrides "$hparam" "$value")"
  log_file="${LOG_DIR}/${task_id}.log"
  status_file="${STATE_DIR}/${task_id}.exit"
  done_file="${STATE_DIR}/${task_id}.done"
  failed_file="${STATE_DIR}/${task_id}.failed"
  rm -f "$status_file" "$failed_file"
  mkdir -p "$task_dir"
  write_config_snapshot "$config_file" "$save_root" "$hparam" "$value" "$dataset" "$gpu" "$script" "$overrides"

  if [[ -n "$gpu" ]]; then
    gpu_args="--gpus ${gpu}"
  else
    gpu_args=""
  fi

  log_scheduler "START ${task_id} on GPU ${gpu:-none}; result/${save_root}/${dataset}; ${hparam}=${value}"
  (
    set +e
    {
      printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$task_id"
      printf 'GPU: %s\n' "${gpu:-none}"
      printf 'Save root: %s\n' "$save_root"
      printf 'Dataset: %s\n' "$dataset"
      printf 'Hyperparameter: %s=%s\n' "$hparam" "$value"
      printf 'LLM model name: %s\n' "$LLM_MODEL_NAME"
      printf 'LLM model path: %s\n' "$LLM_MODEL_PATH"
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
log_scheduler "Conda env: ${MINDTS_HPARAM_CONDA_ENV:-<none>}; python: $(command -v python)"
log_scheduler "GPUs: ${GPUS[*]}"
log_scheduler "Datasets: ${DATASETS[*]}"
log_scheduler "Mask ratio values: ${MASK_RATIO_VALUES}"
log_scheduler "lamda1 values: ${LAMDA1_VALUES}"
log_scheduler "Tasks: ${total}; fixed LLM=${LLM_MODEL_NAME}; align=${ALIGN_LOSS_TYPE}; recon=${RECON_LOSS_TYPE}; num_epochs=${NUM_EPOCHS}"
log_scheduler "GPU ready thresholds: memory<=${GPU_MEMORY_READY_PCT}%, util<=${GPU_UTIL_READY_PCT}%; max_tasks_per_gpu=${MAX_TASKS_PER_GPU}; one_launch_per_gpu_per_poll=${ONE_LAUNCH_PER_GPU_PER_POLL}"

if [[ "$DRY_RUN" == "true" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r hparam value save_root dataset script <<<"$task"
    printf '%s=%s | dataset=%s | save_root=%s | script=%s\n' "$hparam" "$value" "$dataset" "$save_root" "$script"
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
  declare -A GPU_LAUNCHED_THIS_ROUND=()
  while (( next_task < total )); do
    task="${TASKS[$next_task]}"
    IFS='|' read -r hparam value save_root dataset script <<<"$task"
    task_id="$(task_id_for "$hparam" "$value" "$dataset")"
    done_file="${STATE_DIR}/${task_id}.done"

    if [[ "$SKIP_DONE" == "true" && -f "$done_file" ]]; then
      completed=$(( completed + 1 ))
      next_task=$(( next_task + 1 ))
      log_scheduler "SKIP already done ${task_id} (${completed}/${total})"
      continue
    fi

    selected_gpu=""
    for gpu in "${GPUS[@]}"; do
      if [[ "$ONE_LAUNCH_PER_GPU_PER_POLL" == "true" && -n "${GPU_LAUNCHED_THIS_ROUND[$gpu]:-}" ]]; then
        continue
      fi
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
    GPU_LAUNCHED_THIS_ROUND["$selected_gpu"]=1
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
