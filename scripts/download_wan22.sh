#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$PROJECT_ROOT/vendor/Wan2.2}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/$(basename "$MODEL_ID")}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

mkdir -p "$PROJECT_ROOT/vendor" "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs"

if [[ ! -d "$WAN_REPO_DIR/.git" ]]; then
  git clone https://github.com/Wan-Video/Wan2.2.git "$WAN_REPO_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "huggingface_hub[cli]"

if [[ ! -f "$WAN_REPO_DIR/requirements.txt" ]]; then
  echo "Missing $WAN_REPO_DIR/requirements.txt" >&2
  exit 1
fi

echo "Installing Wan2.2 dependencies..."
if ! python -m pip install -r "$WAN_REPO_DIR/requirements.txt"; then
  echo "Base install failed. The official Wan repo notes that flash_attn can fail on some systems." >&2
  echo "Try installing the non-flash-attn dependencies first, then install flash_attn separately for your CUDA stack." >&2
  exit 1
fi

echo "Downloading $MODEL_ID to $MODEL_DIR"
huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"

cat <<EOF
Finished.
Wan repo:   $WAN_REPO_DIR
Venv:       $VENV_DIR
Model dir:  $MODEL_DIR
EOF
