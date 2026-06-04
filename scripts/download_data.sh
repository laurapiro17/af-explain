#!/usr/bin/env bash
# Download the PhysioNet/CinC Challenge 2017 training set into data/raw/.
set -euo pipefail

cd "$(dirname "$0")/.."
uv run python -m af_explain.data.download
