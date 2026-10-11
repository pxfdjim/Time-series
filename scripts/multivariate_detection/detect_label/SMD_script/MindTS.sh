#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${MINDTS_GPU_CLI_ARGS+x}" != "x" ]]; then
  MINDTS_GPU_CLI_ARGS="--gpus 0 1"
fi
gpu_args=()
if [[ -n "$MINDTS_GPU_CLI_ARGS" ]]; then
  read -r -a gpu_args <<<"$MINDTS_GPU_CLI_ARGS"
fi

# Starting settings for each machine's 38 metric channels.
# patch_num is derived from seq_len, patch_size, and stride: (100 - 20) / 20 + 1 = 5.
# stl_period=60 is pending validation; num_epochs=5 is the initial upper limit.
model_hyper_params='{"batch_size": 4, "d_ff": 64, "d_model": 64, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 100, "patch_size": 20, "stride": 20, "top_k": 5, "prompt_stat_precision": 3, "mask_ratio": 0.4, "enc_in_time": 38, "parallel_strategy": null, "stl_period": 60, "dataset_description": "SMD contains 38 server metric channels for one machine."}'

if (( $# > 1 )); then
  echo "Usage: bash $0 [machine-1-1]" >&2
  exit 2
fi

if (( $# == 1 )); then
  files=("dataset/anomaly_detect/data/SMD_$1.csv")
else
  shopt -s nullglob
  files=(dataset/anomaly_detect/data/SMD_machine-*.csv)
  if (( ${#files[@]} != 28 )); then
    echo "Expected 28 separate SMD machine CSVs, found ${#files[@]}; run python scripts/prepare_psm_smd_long.py --dataset smd first." >&2
    exit 1
  fi
fi

for file in "${files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing SMD machine file: $file" >&2
    exit 1
  fi
  series_name="$(basename "$file")"
  echo "Running MindTS on $series_name"
  python ./scripts/run_benchmark.py \
    --config-path unfixed_detect_label_multi_config.json \
    --data-name-list "$series_name" \
    --model-name MindTS.MindTS \
    --model-hyper-params "$model_hyper_params" \
    "${gpu_args[@]}" \
    --num-workers 1 --timeout 60000 \
    --save-path "${MINDTS_SAVE_ROOT:-label}/SMD/${series_name%.csv}"
done
