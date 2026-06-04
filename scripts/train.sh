#!/usr/bin/env bash
# Train the AFib classifier with reproducible defaults.
set -euo pipefail

cd "$(dirname "$0")/.."
uv run af-train \
    --data-dir data/raw/training2017 \
    --output-dir outputs \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --seed 42 \
    "$@"
