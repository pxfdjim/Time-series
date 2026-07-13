#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MINDTS_ABLATION_CONDA_ENV="${MINDTS_ABLATION_CONDA_ENV:-mind_qwen3}"
if [[ -n "$MINDTS_ABLATION_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate ${MINDTS_ABLATION_CONDA_ENV}, but conda was not found." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$MINDTS_ABLATION_CONDA_ENV"
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "The active python cannot import torch. Activate the MindTS environment first or set MINDTS_ABLATION_CONDA_ENV." >&2
  exit 1
fi

RUN_ID="${MINDTS_ABLATION_SCHEDULER_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${ROOT_DIR}/result/mindts_ablation_scheduler_logs/${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

LLM_MODEL_TAG="qwen3_1p7b"
LLM_MODEL_PATH="models/Qwen3-1.7B"
LLM_MODEL_NAME="Qwen3-1.7B"
DEFAULT_ALIGN_LOSS_TYPE="text_gaussian_nll"
RECON_LOSS_TYPE="mse"
RECON_LOGVAR_MIN="-6.0"
RECON_LOGVAR_MAX="2.0"
NUM_EPOCHS="${MINDTS_ABLATION_NUM_EPOCHS:-5}"
MAX_TASKS_PER_GPU="${MINDTS_ABLATION_MAX_TASKS_PER_GPU:-3}"
POLL_SECONDS="${MINDTS_ABLATION_POLL_SECONDS:-20}"
GPU_MEMORY_READY_PCT="${MINDTS_ABLATION_GPU_MEMORY_READY_PCT:-100}"
GPU_UTIL_READY_PCT="${MINDTS_ABLATION_GPU_UTIL_READY_PCT:-100}"
DRY_RUN="${MINDTS_ABLATION_DRY_RUN:-false}"
SKIP_DONE="${MINDTS_ABLATION_SKIP_DONE:-true}"
STOP_ON_FAILURE="${MINDTS_ABLATION_STOP_ON_FAILURE:-false}"
VIS_EXPORT_ROOT="${MINDTS_ABLATION_VIS_EXPORT_ROOT:-anomaly_segment_visualization/exports/qwen3_1p7b_ablation}"

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

DRY_RUN="$(normalize_bool "$DRY_RUN")"
SKIP_DONE="$(normalize_bool "$SKIP_DONE")"
STOP_ON_FAILURE="$(normalize_bool "$STOP_ON_FAILURE")"
validate_bool "MINDTS_ABLATION_DRY_RUN" "$DRY_RUN"
validate_bool "MINDTS_ABLATION_SKIP_DONE" "$SKIP_DONE"
validate_bool "MINDTS_ABLATION_STOP_ON_FAILURE" "$STOP_ON_FAILURE"

case "$MAX_TASKS_PER_GPU" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_ABLATION_MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}. Use a positive integer." >&2
    exit 1
    ;;
esac

if [[ -n "${MINDTS_ABLATION_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${MINDTS_ABLATION_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  GPUS=("")
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
if [[ -n "${MINDTS_ABLATION_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${MINDTS_ABLATION_DATASETS}"
else
  DATASETS=("${default_datasets[@]}")
fi

default_variants=(full wo_text_branch wo_semantic_reconstruction wo_component_semantics)
if [[ -n "${MINDTS_ABLATION_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<<"${MINDTS_ABLATION_VARIANTS}"
else
  VARIANTS=("${default_variants[@]}")
fi

# Optional entries use group|model_variant|alignment_loss. When omitted, keep
# the original one-variant-per-group behavior for backward compatibility.
EXPERIMENT_SPECS=()
if [[ -n "${MINDTS_ABLATION_EXPERIMENT_SPECS:-}" ]]; then
  read -r -a EXPERIMENT_SPECS <<<"${MINDTS_ABLATION_EXPERIMENT_SPECS}"
else
  for variant in "${VARIANTS[@]}"; do
    EXPERIMENT_SPECS+=("${variant}|${variant}|${DEFAULT_ALIGN_LOSS_TYPE}")
  done
fi

for spec in "${EXPERIMENT_SPECS[@]}"; do
  IFS='|' read -r group variant align_loss_type <<<"$spec"
  if [[ -z "$group" || -z "$variant" || -z "$align_loss_type" ]]; then
    echo "Invalid experiment spec '${spec}'. Use group|variant|alignment_loss." >&2
    exit 1
  fi
done

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

build_overrides() {
  local variant="$1"
  local align_loss_type="$2"
  python - "$variant" "$align_loss_type" "$RECON_LOSS_TYPE" "$RECON_LOGVAR_MIN" "$RECON_LOGVAR_MAX" "$LLM_MODEL_PATH" "$LLM_MODEL_NAME" "$NUM_EPOCHS" <<'PY'
import json
import sys

variant = sys.argv[1]
overrides = {
    "model_impl": "ablation",
    "ablation_variant": variant,
    "export_intermediate": True,
    "use_information_condenser": False,
    "align_loss_type": sys.argv[2],
    "recon_loss_type": sys.argv[3],
    "recon_logvar_min": float(sys.argv[4]),
    "recon_logvar_max": float(sys.argv[5]),
    "use_frequency_branch": False,
    "frequency_keep_modes": 4,
    "time_freq_align_weight": 0.2,
    "exchange_text_features": False,
    "reconstruction_exchange_text_features": True,
    "use_de_stationary": False,
    "use_de_stationary_cross_view": False,
    "llm_model_path": sys.argv[6],
    "llm_model_name": sys.argv[7],
    "num_epochs": int(sys.argv[8]),
    "lamda1": 1.0,
}
print(json.dumps(overrides, separators=(",", ":")))
PY
}

save_root_for() {
  local group="$1"
  printf 'label_ablation_%s_%s' "$LLM_MODEL_TAG" "$group"
}

task_id_for() {
  local group="$1"
  local dataset="$2"
  printf '%s__%s' "$group" "$dataset"
}

write_config_snapshot() {
  local config_file="$1"
  local save_root="$2"
  local group="$3"
  local variant="$4"
  local align_loss_type="$5"
  local dataset="$6"
  local gpu="$7"
  local script="$8"
  local overrides="$9"

  mkdir -p "$(dirname "$config_file")"
  {
    printf '{\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "save_root": "%s",\n' "$save_root"
    printf '  "dataset": "%s",\n' "$dataset"
    printf '  "group": "%s",\n' "$group"
    printf '  "variant": "%s",\n' "$variant"
    printf '  "align_loss_type": "%s",\n' "$align_loss_type"
    printf '  "script": "%s",\n' "$script"
    printf '  "gpu": "%s",\n' "$gpu"
    printf '  "llm_model_tag": "%s",\n' "$LLM_MODEL_TAG"
    printf '  "llm_model_name": "%s",\n' "$LLM_MODEL_NAME"
    printf '  "llm_model_path": "%s",\n' "$LLM_MODEL_PATH"
    printf '  "visual_export_dir": "%s/%s",\n' "$VIS_EXPORT_ROOT" "$save_root"
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

TASKS=()
for spec in "${EXPERIMENT_SPECS[@]}"; do
  IFS='|' read -r group variant align_loss_type <<<"$spec"
  save_root="$(save_root_for "$group")"
  for dataset in "${DATASETS[@]}"; do
    TASKS+=("${group}|${variant}|${align_loss_type}|${save_root}|${dataset}|${DATASET_SCRIPT[$dataset]}")
  done
done

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
  local group variant align_loss_type save_root dataset script
  IFS='|' read -r group variant align_loss_type save_root dataset script <<<"$task"

  local task_id task_dir config_file overrides log_file status_file done_file failed_file gpu_args export_dir
  task_id="$(task_id_for "$group" "$dataset")"
  task_dir="${ROOT_DIR}/result/${save_root}/${dataset}"
  config_file="${task_dir}/experiment_config.json"
  overrides="$(build_overrides "$variant" "$align_loss_type")"
  log_file="${LOG_DIR}/${task_id}.log"
  status_file="${STATE_DIR}/${task_id}.exit"
  done_file="${STATE_DIR}/${task_id}.done"
  failed_file="${STATE_DIR}/${task_id}.failed"
  export_dir="${VIS_EXPORT_ROOT}/${save_root}"
  rm -f "$status_file" "$failed_file"
  mkdir -p "$task_dir"
  write_config_snapshot "$config_file" "$save_root" "$group" "$variant" "$align_loss_type" "$dataset" "$gpu" "$script" "$overrides"

  if [[ -n "$gpu" ]]; then
    gpu_args="--gpus ${gpu}"
  else
    gpu_args=""
  fi

  log_scheduler "START ${task_id} on GPU ${gpu:-none}; result/${save_root}/${dataset}; variant=${variant}; align=${align_loss_type}"
  (
    set +e
    {
      printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$task_id"
      printf 'GPU: %s\n' "${gpu:-none}"
      printf 'Save root: %s\n' "$save_root"
      printf 'Dataset: %s\n' "$dataset"
      printf 'Group: %s\n' "$group"
      printf 'Variant: %s\n' "$variant"
      printf 'Alignment: %s\n' "$align_loss_type"
      printf 'Visual export: %s\n' "$export_dir"
      printf 'Overrides: %s\n' "$overrides"
      printf 'Script: %s\n\n' "$script"
    } >>"$log_file"

    export MINDTS_SAVE_ROOT="$save_root"
    export MINDTS_MODEL_HYPER_PARAM_OVERRIDES="$overrides"
    export MINDTS_GPU_CLI_ARGS="$gpu_args"
    export MINDTS_VIS_EXPORT_DIR="$export_dir"
    export MINDTS_SAVE_CHECKPOINT=true
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
log_scheduler "Conda env: ${MINDTS_ABLATION_CONDA_ENV:-<none>}; python: $(command -v python)"
log_scheduler "GPUs: ${GPUS[*]}"
log_scheduler "Datasets: ${DATASETS[*]}"
log_scheduler "Experiment specs: ${EXPERIMENT_SPECS[*]}"
log_scheduler "Tasks: ${total}; fixed LLM=${LLM_MODEL_NAME}; epochs=${NUM_EPOCHS}; max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
log_scheduler "Visual export root: ${VIS_EXPORT_ROOT}"

if [[ "$DRY_RUN" == "true" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r group variant align_loss_type save_root dataset script <<<"$task"
    printf 'group=%s | variant=%s | align=%s | dataset=%s | save_root=%s | script=%s\n' "$group" "$variant" "$align_loss_type" "$dataset" "$save_root" "$script"
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
    IFS='|' read -r group variant align_loss_type save_root dataset script <<<"$task"
    task_id="$(task_id_for "$group" "$dataset")"
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
