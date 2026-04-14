#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$PROJECT_ROOT/vendor/Wan2.2}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Wan2.2-TI2V-5B}"
STORY_JSON="${STORY_JSON:-$PROJECT_ROOT/config/story.example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_COUNT="${GPU_COUNT:-1}"

mkdir -p "$OUTPUT_DIR"

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/story_to_movie.py" \
  --story "$STORY_JSON" \
  --wan-repo "$WAN_REPO_DIR" \
  --ckpt-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gpus "$GPU_COUNT" \
  --concat \
  "$@"
