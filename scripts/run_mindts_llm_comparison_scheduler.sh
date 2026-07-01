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

RUN_ID="${MINDTS_LLM_SCHEDULER_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${ROOT_DIR}/result/mindts_llm_comparison_scheduler_logs/${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

# Best known setting from result/label_text_gaussian_nll_exchange_recon_text.
ALIGN_LOSS_TYPE="text_gaussian_nll"
RECON_LOSS_TYPE="mse"
RECON_LOGVAR_MIN="-6.0"
RECON_LOGVAR_MAX="2.0"
CROSS_VIEW_DIRECTION="prompt_feature_query_semantic_features"
RECONSTRUCTION_DIRECTION="reconstruction_patch_features_then_llm_features"

GPU_MEMORY_READY_PCT="${MINDTS_LLM_SCHEDULER_GPU_MEMORY_READY_PCT:-95}"
GPU_UTIL_READY_PCT="${MINDTS_LLM_SCHEDULER_GPU_UTIL_READY_PCT:-100}"
MAX_TASKS_PER_GPU="${MINDTS_LLM_SCHEDULER_MAX_TASKS_PER_GPU:-3}"
POLL_SECONDS="${MINDTS_LLM_SCHEDULER_POLL_SECONDS:-20}"
ONE_LAUNCH_PER_GPU_PER_POLL="${MINDTS_LLM_SCHEDULER_ONE_LAUNCH_PER_GPU_PER_POLL:-true}"
DRY_RUN="${MINDTS_LLM_SCHEDULER_DRY_RUN:-false}"
SKIP_DONE="${MINDTS_LLM_SCHEDULER_SKIP_DONE:-true}"
STOP_ON_FAILURE="${MINDTS_LLM_SCHEDULER_STOP_ON_FAILURE:-false}"
SAVE_SUFFIX="${MINDTS_LLM_SAVE_SUFFIX:-}"
NUM_EPOCHS="${MINDTS_LLM_NUM_EPOCHS:-5}"

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
ONE_LAUNCH_PER_GPU_PER_POLL="$(normalize_bool "$ONE_LAUNCH_PER_GPU_PER_POLL")"
validate_bool "MINDTS_LLM_SCHEDULER_DRY_RUN" "$DRY_RUN"
validate_bool "MINDTS_LLM_SCHEDULER_SKIP_DONE" "$SKIP_DONE"
validate_bool "MINDTS_LLM_SCHEDULER_STOP_ON_FAILURE" "$STOP_ON_FAILURE"
validate_bool "MINDTS_LLM_SCHEDULER_ONE_LAUNCH_PER_GPU_PER_POLL" "$ONE_LAUNCH_PER_GPU_PER_POLL"

case "$MAX_TASKS_PER_GPU" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_LLM_SCHEDULER_MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$POLL_SECONDS" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_LLM_SCHEDULER_POLL_SECONDS=${POLL_SECONDS}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$NUM_EPOCHS" in
  ''|*[!0-9]*|0)
    echo "Invalid MINDTS_LLM_NUM_EPOCHS=${NUM_EPOCHS}. Use a positive integer." >&2
    exit 1
    ;;
esac

if [[ -n "${MINDTS_LLM_SCHEDULER_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${MINDTS_LLM_SCHEDULER_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  GPUS=("")
fi

if (( ${#GPUS[@]} == 0 )); then
  echo "No GPU candidates found. Set MINDTS_LLM_SCHEDULER_GPUS, e.g. '0 1'." >&2
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

candidate_model_rows=(
  "qwen2p5_1p5b|models/Qwen2.5-1.5B-Instruct|Qwen2.5-1.5B-Instruct"
  "qwen3_1p7b|models/Qwen3-1.7B|Qwen3-1.7B"
  "smollm2_1p7b|models/SmolLM2-1.7B-Instruct|SmolLM2-1.7B-Instruct"
  "tinyllama_1p1b|models/TinyLlama-1.1B-Chat-v1.0|TinyLlama-1.1B-Chat-v1.0"
  "mobilellama_1p4b|models/MobileLLaMA-1.4B-Base|MobileLLaMA-1.4B-Base"
  "phi_1p5|models/phi-1_5|phi-1_5"
  "olmo1b_0724|models/OLMo-1B-0724-hf|OLMo-1B-0724-hf"
)

if [[ -n "${MINDTS_LLM_MODELS:-}" ]]; then
  read -r -a REQUESTED_MODELS <<<"${MINDTS_LLM_MODELS}"
else
  REQUESTED_MODELS=()
fi

model_requested() {
  local model_tag="$1"
  if (( ${#REQUESTED_MODELS[@]} == 0 )); then
    return 0
  fi
  local requested
  for requested in "${REQUESTED_MODELS[@]}"; do
    if [[ "$requested" == "$model_tag" ]]; then
      return 0
    fi
  done
  return 1
}

log_scheduler() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" | tee -a "${LOG_DIR}/scheduler.log"
}

model_supported() {
  local model_path="$1"
  python - "$model_path" <<'PY'
import sys
from transformers import AutoConfig

try:
    cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc).splitlines()[0]}", file=sys.stderr)
    raise SystemExit(1)

print(
    f"model_type={getattr(cfg, 'model_type', None)} "
    f"hidden_size={getattr(cfg, 'hidden_size', getattr(cfg, 'n_embd', None))} "
    f"num_hidden_layers={getattr(cfg, 'num_hidden_layers', None)}"
)
PY
}

model_rows=()
for row in "${candidate_model_rows[@]}"; do
  IFS='|' read -r model_tag model_path model_name <<<"$row"
  if ! model_requested "$model_tag"; then
    continue
  fi
  if [[ ! -d "$model_path" ]]; then
    log_scheduler "SKIP model ${model_name}: ${model_path} does not exist"
    printf '%s | missing path: %s\n' "$model_name" "$model_path" >>"${LOG_DIR}/skipped_models.log"
    continue
  fi
  if support_info="$(model_supported "$model_path" 2>"${LOG_DIR}/.${model_tag}.support.err")"; then
    log_scheduler "MODEL OK ${model_name}: ${support_info}"
    model_rows+=("$row")
  else
    reason="$(cat "${LOG_DIR}/.${model_tag}.support.err")"
    log_scheduler "SKIP model ${model_name}: ${reason}"
    printf '%s | %s | %s\n' "$model_name" "$model_path" "$reason" >>"${LOG_DIR}/skipped_models.log"
  fi
  rm -f "${LOG_DIR}/.${model_tag}.support.err"
done

if (( ${#model_rows[@]} == 0 )); then
  log_scheduler "No compatible LLM models found."
  exit 1
fi

TASKS=()
for row in "${model_rows[@]}"; do
  IFS='|' read -r model_tag model_path model_name <<<"$row"
  save_root="label_text_gaussian_nll_exchange_recon_text_llm_${model_tag}"
  if [[ -n "$SAVE_SUFFIX" ]]; then
    save_root="${save_root}_${SAVE_SUFFIX}"
  fi
  for dataset in "${DATASETS[@]}"; do
    TASKS+=("${save_root}|${model_tag}|${model_path}|${model_name}|${dataset}|${DATASET_SCRIPT[$dataset]}")
  done
done

task_id_for() {
  local save_root="$1"
  local dataset="$2"
  printf '%s__%s' "$save_root" "$dataset"
}

build_overrides() {
  local model_path="$1"
  local model_name="$2"
  printf '{"align_loss_type": "%s", "recon_loss_type": "%s", "recon_logvar_min": %s, "recon_logvar_max": %s, "llm_model_path": "%s", "llm_model_name": "%s", "num_epochs": %s}' \
    "$ALIGN_LOSS_TYPE" \
    "$RECON_LOSS_TYPE" \
    "$RECON_LOGVAR_MIN" \
    "$RECON_LOGVAR_MAX" \
    "$model_path" \
    "$model_name" \
    "$NUM_EPOCHS"
}

write_config_snapshot() {
  local config_file="$1"
  local save_root="$2"
  local model_tag="$3"
  local model_path="$4"
  local model_name="$5"
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
    printf '  "script": "%s",\n' "$script"
    printf '  "gpu": "%s",\n' "$gpu"
    printf '  "llm_model_tag": "%s",\n' "$model_tag"
    printf '  "llm_model_name": "%s",\n' "$model_name"
    printf '  "llm_model_path": "%s",\n' "$model_path"
    printf '  "align_loss_type": "%s",\n' "$ALIGN_LOSS_TYPE"
    printf '  "recon_loss_type": "%s",\n' "$RECON_LOSS_TYPE"
    printf '  "cross_view_direction": "%s",\n' "$CROSS_VIEW_DIRECTION"
    printf '  "reconstruction_direction": "%s",\n' "$RECONSTRUCTION_DIRECTION"
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
  local save_root model_tag model_path model_name dataset script
  IFS='|' read -r save_root model_tag model_path model_name dataset script <<<"$task"

  local task_id task_dir config_file overrides log_file status_file done_file failed_file gpu_args
  task_id="$(task_id_for "$save_root" "$dataset")"
  task_dir="${ROOT_DIR}/result/${save_root}/${dataset}"
  config_file="${task_dir}/experiment_config.json"
  overrides="$(build_overrides "$model_path" "$model_name")"
  log_file="${LOG_DIR}/${task_id}.log"
  status_file="${STATE_DIR}/${task_id}.exit"
  done_file="${STATE_DIR}/${task_id}.done"
  failed_file="${STATE_DIR}/${task_id}.failed"
  rm -f "$status_file" "$failed_file"
  mkdir -p "$task_dir"
  write_config_snapshot "$config_file" "$save_root" "$model_tag" "$model_path" "$model_name" "$dataset" "$gpu" "$script" "$overrides"

  if [[ -n "$gpu" ]]; then
    gpu_args="--gpus ${gpu}"
  else
    gpu_args=""
  fi

  log_scheduler "START ${task_id} on GPU ${gpu:-none}; result/${save_root}/${dataset}; LLM=${model_name}"
  (
    set +e
    {
      printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$task_id"
      printf 'GPU: %s\n' "${gpu:-none}"
      printf 'Save root: %s\n' "$save_root"
      printf 'Dataset: %s\n' "$dataset"
      printf 'LLM model tag: %s\n' "$model_tag"
      printf 'LLM model name: %s\n' "$model_name"
      printf 'LLM model path: %s\n' "$model_path"
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
log_scheduler "Fixed best config: align=${ALIGN_LOSS_TYPE}, recon=${RECON_LOSS_TYPE}, cross_view=${CROSS_VIEW_DIRECTION}, reconstruction=${RECONSTRUCTION_DIRECTION}, num_epochs=${NUM_EPOCHS}"
log_scheduler "GPU ready thresholds: memory<=${GPU_MEMORY_READY_PCT}%, util<=${GPU_UTIL_READY_PCT}%; max_tasks_per_gpu=${MAX_TASKS_PER_GPU}; one_launch_per_gpu_per_poll=${ONE_LAUNCH_PER_GPU_PER_POLL}"

if [[ "$DRY_RUN" == "true" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r save_root model_tag model_path model_name dataset script <<<"$task"
    printf '%s | dataset=%s | llm=%s | path=%s | cross_view=%s | reconstruction=%s\n' \
      "$save_root" "$dataset" "$model_name" "$model_path" "$CROSS_VIEW_DIRECTION" "$RECONSTRUCTION_DIRECTION"
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
    IFS='|' read -r save_root model_tag model_path model_name dataset script <<<"$task"
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
