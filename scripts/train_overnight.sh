#!/usr/bin/env bash
# Overnight training run. Designed to survive macOS sleep and pipe issues.
#
# REQUIREMENTS BEFORE RUNNING:
#   1. Mac connected to AC power (this script refuses to run on battery).
#   2. Lid open OR external display attached (caffeinate -i alone doesn't
#      prevent display sleep; if you close the lid without an external
#      monitor the Mac sleeps anyway on most models).
#   3. ~10 GB free disk (data + checkpoints + logs).
#
# Usage:
#   ./scripts/train_overnight.sh                # default: 15 epochs
#   ./scripts/train_overnight.sh --epochs 25    # override any af-train flag
#
# Resume the latest checkpoint instead of starting fresh:
#   ./scripts/train_overnight.sh --resume

set -euo pipefail

cd "$(dirname "$0")/.."

# ─── safety: AC power ────────────────────────────────────────────────────────
if pmset -g batt | grep -q "Battery Power"; then
    echo "❌ Mac is on battery. Connect AC and re-run."
    pmset -g batt | head -2
    exit 1
fi

# ─── safety: disk space ──────────────────────────────────────────────────────
free_gb=$(df -g / | awk 'NR==2 {print $4}')
if [ "${free_gb:-0}" -lt 8 ]; then
    echo "❌ Less than 8 GB free on /. Free some space and re-run."
    df -h /
    exit 1
fi

# ─── safety: editable install (uv sync sometimes skips this) ─────────────────
if ! .venv/bin/python -c "import af_explain" 2>/dev/null; then
    echo "ℹ️  af_explain not importable — running 'uv pip install -e .'"
    uv pip install -e .
fi

mkdir -p outputs
LOG="outputs/train-$(date +%Y%m%dT%H%M%S).log"
echo "📋 Logging to $LOG"
echo "🔋 Power: $(pmset -g batt | sed -n 1p)"
echo "💾 Free disk: ${free_gb} GB"
echo "🧪 Args: $*"
echo "─────────────────────────────────────────────────────"

# ─── default flags ───────────────────────────────────────────────────────────
ARGS=(
    --epochs 15
    --batch-size 32
    --num-workers 4
    --learning-rate 1e-3
    --seed 42
)
# Pass-through user args (override defaults).
[ $# -gt 0 ] && ARGS=("$@")

# ─── run ─────────────────────────────────────────────────────────────────────
#   caffeinate -d   prevent display sleep
#   caffeinate -i   prevent idle sleep
#   caffeinate -s   keep system awake while AC power
caffeinate -dis .venv/bin/af-train "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "🚀 Training launched (PID $PID). To follow progress:"
echo "    tail -f $LOG"
echo "To stop:"
echo "    kill $PID"
wait "$PID"
echo "✅ Training process exited with status $?"
