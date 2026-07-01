#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-mind}"
if [[ -n "$BASELINE_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate ${BASELINE_CONDA_ENV}, but conda was not found." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$BASELINE_CONDA_ENV"
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "The active python cannot import torch. Activate the baseline environment first or set BASELINE_CONDA_ENV." >&2
  echo "python: $(command -v python)" >&2
  exit 1
fi

RUN_ID="${BASELINE_SCHEDULER_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${ROOT_DIR}/result/baseline_comparison_scheduler_logs/${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

GPU_MEMORY_READY_PCT="${BASELINE_SCHEDULER_GPU_MEMORY_READY_PCT:-90}"
GPU_UTIL_READY_PCT="${BASELINE_SCHEDULER_GPU_UTIL_READY_PCT:-100}"
MAX_TASKS_PER_GPU="${BASELINE_SCHEDULER_MAX_TASKS_PER_GPU:-3}"
POLL_SECONDS="${BASELINE_SCHEDULER_POLL_SECONDS:-20}"
ONE_LAUNCH_PER_GPU_PER_POLL="${BASELINE_SCHEDULER_ONE_LAUNCH_PER_GPU_PER_POLL:-true}"
DRY_RUN="${BASELINE_SCHEDULER_DRY_RUN:-false}"
SKIP_DONE="${BASELINE_SCHEDULER_SKIP_DONE:-true}"
STOP_ON_FAILURE="${BASELINE_SCHEDULER_STOP_ON_FAILURE:-false}"
SAVE_PREFIX="${BASELINE_SAVE_PREFIX:-label_baseline}"
SAVE_SUFFIX="${BASELINE_SAVE_SUFFIX:-}"
NUM_EPOCHS="${BASELINE_NUM_EPOCHS:-5}"

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
validate_bool "BASELINE_SCHEDULER_DRY_RUN" "$DRY_RUN"
validate_bool "BASELINE_SCHEDULER_SKIP_DONE" "$SKIP_DONE"
validate_bool "BASELINE_SCHEDULER_STOP_ON_FAILURE" "$STOP_ON_FAILURE"
validate_bool "BASELINE_SCHEDULER_ONE_LAUNCH_PER_GPU_PER_POLL" "$ONE_LAUNCH_PER_GPU_PER_POLL"

case "$MAX_TASKS_PER_GPU" in
  ''|*[!0-9]*|0)
    echo "Invalid BASELINE_SCHEDULER_MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$POLL_SECONDS" in
  ''|*[!0-9]*|0)
    echo "Invalid BASELINE_SCHEDULER_POLL_SECONDS=${POLL_SECONDS}. Use a positive integer." >&2
    exit 1
    ;;
esac

case "$NUM_EPOCHS" in
  ''|*[!0-9]*|0)
    echo "Invalid BASELINE_NUM_EPOCHS=${NUM_EPOCHS}. Use a positive integer." >&2
    exit 1
    ;;
esac

if [[ -n "${BASELINE_SCHEDULER_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${BASELINE_SCHEDULER_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  GPUS=("")
fi

if (( ${#GPUS[@]} == 0 )); then
  echo "No GPU candidates found. Set BASELINE_SCHEDULER_GPUS, e.g. '0 1'." >&2
  exit 1
fi

declare -A DATASET_CONFIG=(
  [Energy]="unfixed_detect_label_multi_config.json"
  [Weather]="unfixed_detect_label_multi_config.json"
  [EWJ]="unfixed_detect_label_config.json"
  [Environment]="unfixed_detect_label_config.json"
  [MDT]="unfixed_detect_label_config.json"
  [KR]="unfixed_detect_label_config.json"
)

declare -A DATASET_FILE=(
  [Energy]="Energy.csv"
  [Weather]="Weather.csv"
  [EWJ]="EWJ.csv"
  [Environment]="Environment.csv"
  [MDT]="MDT.csv"
  [KR]="KR.csv"
)

declare -A DATASET_HYPER=(
  [Energy]='{"batch_size": 32, "d_ff": 8, "d_model": 256, "e_layers": 1, "d_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 24, "patch_len": 6, "stride": 6, "seg_len": 6, "enc_in": 9, "dec_in": 9, "c_out": 9}'
  [Weather]='{"batch_size": 64, "d_ff": 8, "d_model": 64, "e_layers": 1, "d_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 24, "patch_len": 6, "stride": 6, "seg_len": 6, "enc_in": 4, "dec_in": 4, "c_out": 4}'
  [EWJ]='{"batch_size": 16, "d_ff": 512, "d_model": 256, "e_layers": 1, "d_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 48, "patch_len": 6, "stride": 6, "seg_len": 6, "enc_in": 1, "dec_in": 1, "c_out": 1}'
  [Environment]='{"batch_size": 64, "d_ff": 64, "d_model": 64, "e_layers": 1, "d_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 72, "patch_len": 6, "stride": 6, "seg_len": 6, "enc_in": 1, "dec_in": 1, "c_out": 1}'
  [MDT]='{"batch_size": 32, "d_ff": 32, "d_model": 256, "e_layers": 1, "d_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 24, "patch_len": 6, "stride": 6, "seg_len": 6, "enc_in": 1, "dec_in": 1, "c_out": 1}'
  [KR]='{"batch_size": 16, "d_ff": 16, "d_model": 256, "e_layers": 1, "d_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 24, "patch_len": 8, "stride": 8, "seg_len": 8, "enc_in": 1, "dec_in": 1, "c_out": 1}'
)

default_datasets=(Energy Weather EWJ Environment MDT KR)
if [[ -n "${BASELINE_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${BASELINE_DATASETS}"
else
  DATASETS=("${default_datasets[@]}")
fi

for dataset in "${DATASETS[@]}"; do
  if [[ -z "${DATASET_FILE[$dataset]:-}" ]]; then
    echo "Unknown dataset '${dataset}'. Supported: ${default_datasets[*]}" >&2
    exit 1
  fi
done

candidate_model_rows=(
  "timesnet|time_series_library.TimesNet|transformer_adapter"
  "patchtst|time_series_library.PatchTST|transformer_adapter"
  "dlinear|time_series_library.DLinear|transformer_adapter"
  "nlinear|time_series_library.NLinear|transformer_adapter"
  "transformer|time_series_library.Transformer|transformer_adapter"
  "informer|time_series_library.Informer|transformer_adapter"
  "autoformer|time_series_library.Autoformer|transformer_adapter"
  "fedformer|time_series_library.FEDformer|transformer_adapter"
  "nonstationary_transformer|time_series_library.Nonstationary_Transformer|transformer_adapter"
  "itransformer|time_series_library.iTransformer|transformer_adapter"
  "crossformer|time_series_library.Crossformer|transformer_adapter"
  "lightts|time_series_library.LightTS|transformer_adapter"
  "micn|time_series_library.MICN|transformer_adapter"
  "reformer|time_series_library.Reformer|transformer_adapter"
  "pyraformer|time_series_library.Pyraformer|transformer_adapter"
  "koopa|time_series_library.Koopa|transformer_adapter"
  "film|time_series_library.FiLM|transformer_adapter"
  "linear|time_series_library.Linear|transformer_adapter"
  "triformer|time_series_library.Triformer|transformer_adapter"
)

default_model_tags=(
  timesnet
  patchtst
  dlinear
  nlinear
  transformer
  informer
  autoformer
  fedformer
  nonstationary_transformer
  itransformer
  crossformer
  lightts
  micn
  reformer
  pyraformer
  film
  linear
)

if [[ -n "${BASELINE_MODELS:-}" ]]; then
  read -r -a REQUESTED_MODELS <<<"${BASELINE_MODELS}"
else
  REQUESTED_MODELS=("${default_model_tags[@]}")
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

with_num_epochs() {
  local hyper_params="$1"
  python - "$hyper_params" "$NUM_EPOCHS" <<'PY'
import json
import sys

params = json.loads(sys.argv[1])
params["num_epochs"] = int(sys.argv[2])
print(json.dumps(params, separators=(",", ":")))
PY
}

model_supported() {
  local model_name="$1"
  local adapter="$2"
  python - "$model_name" "$adapter" <<'PY'
import sys
from ts_benchmark.models.model_loader import get_model_info
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerConfig

model_name, adapter = sys.argv[1], sys.argv[2]
model_cls = get_model_info({"model_name": model_name})
params = dict(
    batch_size=16,
    d_ff=16,
    d_model=256,
    e_layers=1,
    d_layers=1,
    horizon=0,
    norm=True,
    num_epochs=5,
    seq_len=24,
    patch_len=8,
    stride=8,
    seg_len=8,
    enc_in=1,
    dec_in=1,
    c_out=1,
)
config = TransformerConfig(**params)
config.task_name = "anomaly_detection"
config.label_len = 48
model = model_cls(config)
param_count = sum(p.numel() for p in model.parameters())
cfg = {"model_name": model_name}
if adapter != "None":
    cfg["adapter"] = adapter
info = get_model_info(cfg)
required = ",".join(sorted(info.get("required_hyper_params", {}).keys())) if isinstance(info, dict) else ""
print(f"params={param_count} required={required}")
PY
}

model_rows=()
for row in "${candidate_model_rows[@]}"; do
  IFS='|' read -r model_tag model_name adapter <<<"$row"
  if ! model_requested "$model_tag"; then
    continue
  fi
  if support_info="$(model_supported "$model_name" "$adapter" 2>"${LOG_DIR}/.${model_tag}.support.err")"; then
    log_scheduler "MODEL OK ${model_tag}: ${model_name} adapter=${adapter} ${support_info}"
    model_rows+=("$row")
  else
    reason="$(cat "${LOG_DIR}/.${model_tag}.support.err")"
    log_scheduler "SKIP model ${model_tag}: ${reason}"
    printf '%s | %s | %s | %s\n' "$model_tag" "$model_name" "$adapter" "$reason" >>"${LOG_DIR}/skipped_models.log"
  fi
  rm -f "${LOG_DIR}/.${model_tag}.support.err"
done

if (( ${#model_rows[@]} == 0 )); then
  log_scheduler "No compatible baseline models found."
  exit 1
fi

TASKS=()
for row in "${model_rows[@]}"; do
  IFS='|' read -r model_tag model_name adapter <<<"$row"
  save_root="${SAVE_PREFIX}_${model_tag}"
  if [[ -n "$SAVE_SUFFIX" ]]; then
    save_root="${save_root}_${SAVE_SUFFIX}"
  fi
  for dataset in "${DATASETS[@]}"; do
    TASKS+=("${save_root}|${model_tag}|${model_name}|${adapter}|${dataset}")
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
  local model_tag="$3"
  local model_name="$4"
  local adapter="$5"
  local dataset="$6"
  local gpu="$7"
  local hyper_params="$8"

  mkdir -p "$(dirname "$config_file")"
  {
    printf '{\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "save_root": "%s",\n' "$save_root"
    printf '  "dataset": "%s",\n' "$dataset"
    printf '  "gpu": "%s",\n' "$gpu"
    printf '  "baseline_model_tag": "%s",\n' "$model_tag"
    printf '  "baseline_model_name": "%s",\n' "$model_name"
    printf '  "adapter": "%s",\n' "$adapter"
    printf '  "model_hyper_params": %s\n' "$hyper_params"
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
  [[ "$mem_pct" -le "$GPU_MEMORY_READY_PCT" && "$util" -le "$GPU_UTIL_READY_PCT" ]]
}

poll_finished() {
  local pid status_file gpu task_id status
  for pid in "${!PID_STATUS_FILE[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    wait "$pid" || status=$?
    status="${status:-0}"
    status_file="${PID_STATUS_FILE[$pid]}"
    gpu="${PID_GPU[$pid]}"
    task_id="${PID_TASK[$pid]}"
    GPU_ACTIVE["$gpu"]=$(( GPU_ACTIVE["$gpu"] - 1 ))
    if [[ "$status" == "0" ]]; then
      completed=$(( completed + 1 ))
      touch "${STATE_DIR}/${task_id}.done"
      log_scheduler "DONE ${task_id} on GPU ${gpu} (${completed}/${total})"
    else
      completed=$(( completed + 1 ))
      failed=$(( failed + 1 ))
      touch "${STATE_DIR}/${task_id}.failed"
      log_scheduler "FAIL ${task_id} on GPU ${gpu} status=${status} (${completed}/${total})"
    fi
    printf '%s\n' "$status" >"$status_file"
    unset "PID_STATUS_FILE[$pid]" "PID_GPU[$pid]" "PID_TASK[$pid]"
  done
}

launch_task() {
  local task="$1"
  local gpu="$2"
  local save_root model_tag model_name adapter dataset config_path data_file hyper_params task_id result_dir task_log config_file status_file
  IFS='|' read -r save_root model_tag model_name adapter dataset <<<"$task"
  config_path="${DATASET_CONFIG[$dataset]}"
  data_file="${DATASET_FILE[$dataset]}"
  hyper_params="$(with_num_epochs "${DATASET_HYPER[$dataset]}")"
  task_id="$(task_id_for "$save_root" "$dataset")"
  result_dir="result/${save_root}/${dataset}"
  task_log="${LOG_DIR}/${task_id}.log"
  config_file="${LOG_DIR}/${task_id}.config.json"
  status_file="${STATE_DIR}/${task_id}.exit"
  mkdir -p "$result_dir"
  write_config_snapshot "$config_file" "$save_root" "$model_tag" "$model_name" "$adapter" "$dataset" "$gpu" "$hyper_params"
  log_scheduler "START ${task_id} on GPU ${gpu}; ${result_dir}; model=${model_name}"
  (
    set +u
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BASELINE_TASK_ID="$task_id"
    export BASELINE_MODEL_TAG="$model_tag"
    unset MINDTS_MODEL_HYPER_PARAM_OVERRIDES
    python ./scripts/run_benchmark.py \
      --config-path "$config_path" \
      --data-name-list "$data_file" \
      --model-name "$model_name" \
      --adapter "$adapter" \
      --model-hyper-params "$hyper_params" \
      --gpus "$gpu" \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "${save_root}/${dataset}"
    status=$?
    printf '[%s] EXIT %s status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$task_id" "$status"
    exit "$status"
  ) >"$task_log" 2>&1 &
  local pid=$!
  PID_STATUS_FILE["$pid"]="$status_file"
  PID_GPU["$pid"]="$gpu"
  PID_TASK["$pid"]="$task_id"
  GPU_ACTIVE["$gpu"]=$(( GPU_ACTIVE["$gpu"] + 1 ))
  started=$(( started + 1 ))
}

declare -A GPU_ACTIVE
for gpu in "${GPUS[@]}"; do
  GPU_ACTIVE["$gpu"]=0
done
declare -A PID_STATUS_FILE
declare -A PID_GPU
declare -A PID_TASK

failed=0
completed=0
started=0
total=${#TASKS[@]}

log_scheduler "Run id: ${RUN_ID}"
log_scheduler "Log dir: ${LOG_DIR}"
log_scheduler "Conda env: ${BASELINE_CONDA_ENV:-<none>}; python: $(command -v python)"
log_scheduler "GPUs: ${GPUS[*]}"
log_scheduler "Models: ${#model_rows[@]}; datasets: ${DATASETS[*]}; tasks: ${total}"
log_scheduler "num_epochs=${NUM_EPOCHS}"
log_scheduler "GPU ready thresholds: memory<=${GPU_MEMORY_READY_PCT}%, util<=${GPU_UTIL_READY_PCT}%; max_tasks_per_gpu=${MAX_TASKS_PER_GPU}; one_launch_per_gpu_per_poll=${ONE_LAUNCH_PER_GPU_PER_POLL}"

if [[ "$DRY_RUN" == "true" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r save_root model_tag model_name adapter dataset <<<"$task"
    printf '%s | dataset=%s | model=%s | adapter=%s | config=%s | data=%s\n' \
      "$save_root" "$dataset" "$model_name" "$adapter" "${DATASET_CONFIG[$dataset]}" "${DATASET_FILE[$dataset]}"
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
    IFS='|' read -r save_root model_tag model_name adapter dataset <<<"$task"
    task_id="$(task_id_for "$save_root" "$dataset")"
    done_file="${STATE_DIR}/${task_id}.done"

    if [[ "$SKIP_DONE" == "true" && -f "$done_file" ]]; then
      completed=$(( completed + 1 ))
      log_scheduler "SKIP done ${task_id} (${completed}/${total})"
      next_task=$(( next_task + 1 ))
      continue
    fi

    launched=0
    for gpu in "${GPUS[@]}"; do
      if [[ "$ONE_LAUNCH_PER_GPU_PER_POLL" == "true" && -n "${GPU_LAUNCHED_THIS_ROUND[$gpu]:-}" ]]; then
        continue
      fi
      if (( GPU_ACTIVE["$gpu"] >= MAX_TASKS_PER_GPU )); then
        continue
      fi
      if gpu_ready "$gpu"; then
        launch_task "$task" "$gpu"
        GPU_LAUNCHED_THIS_ROUND["$gpu"]=1
        launched=1
        launched_this_round=1
        next_task=$(( next_task + 1 ))
        break
      fi
    done

    if (( launched == 0 )); then
      break
    fi
  done

  poll_finished
  if (( completed >= total )); then
    break
  fi
  if (( launched_this_round == 0 )); then
    sleep "$POLL_SECONDS"
  else
    sleep 2
  fi
done

poll_finished
if (( failed )); then
  log_scheduler "Finished with ${failed} failed task(s), ${completed}/${total} completed."
  exit 1
fi
log_scheduler "All tasks completed successfully (${completed}/${total})."
