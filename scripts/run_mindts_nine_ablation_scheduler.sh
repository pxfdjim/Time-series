#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Nine requested experiment groups. Each group covers all six datasets.
export MINDTS_ABLATION_EXPERIMENT_SPECS="${MINDTS_NINE_EXPERIMENT_SPECS:-component_only|wo_behavioral_prompt|text_gaussian_nll align_mse|full|text_mse align_cosine|full|text_cosine trend_only|component_trend_only|text_gaussian_nll seasonal_only|component_seasonal_only|text_gaussian_nll residual_only|component_residual_only|text_gaussian_nll trend_seasonal|component_trend_seasonal|text_gaussian_nll trend_residual|component_trend_residual|text_gaussian_nll seasonal_residual|component_seasonal_residual|text_gaussian_nll}"
export MINDTS_ABLATION_CONDA_ENV="${MINDTS_NINE_CONDA_ENV:-mind_qwen3}"
export MINDTS_ABLATION_MAX_TASKS_PER_GPU="${MINDTS_NINE_MAX_TASKS_PER_GPU:-3}"
export MINDTS_ABLATION_GPUS="${MINDTS_NINE_GPUS:-0 1}"
export MINDTS_ABLATION_NUM_EPOCHS="${MINDTS_NINE_NUM_EPOCHS:-5}"
export MINDTS_ABLATION_GPU_MEMORY_READY_PCT="${MINDTS_NINE_GPU_MEMORY_READY_PCT:-100}"
export MINDTS_ABLATION_GPU_UTIL_READY_PCT="${MINDTS_NINE_GPU_UTIL_READY_PCT:-100}"
export MINDTS_ABLATION_VIS_EXPORT_ROOT="${MINDTS_NINE_VIS_EXPORT_ROOT:-anomaly_segment_visualization/exports/qwen3_1p7b_nine_ablation}"

exec bash "${ROOT_DIR}/scripts/run_mindts_ablation_scheduler.sh"
