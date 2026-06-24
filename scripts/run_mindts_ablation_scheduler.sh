#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "MindTS ablation switches were removed from the model path."
echo "Running the fixed best configuration instead."

exec bash scripts/run_all_mindts.sh "$@"
