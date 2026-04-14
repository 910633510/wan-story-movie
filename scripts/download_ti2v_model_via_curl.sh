#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models/Wan2.2-TI2V-5B}"
BASE_URL="${BASE_URL:-https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main}"

mkdir -p "$MODEL_ROOT/google/umt5-xxl"

remote_size() {
  local relative_path="$1"
  curl -sIL "$BASE_URL/$relative_path" | awk 'tolower($1)=="content-length:" {gsub("\r","",$2); print $2}' | tail -1
}

download() {
  local relative_path="$1"
  local destination="$MODEL_ROOT/$relative_path"
  local remote_bytes
  local local_bytes
  local status=0

  mkdir -p "$(dirname "$destination")"

  remote_bytes="$(remote_size "$relative_path")"
  local_bytes=0
  if [[ -f "$destination" ]]; then
    local_bytes="$(wc -c < "$destination" | tr -d ' ')"
  fi

  if [[ -n "$remote_bytes" && "$remote_bytes" =~ ^[0-9]+$ && "$local_bytes" -ge "$remote_bytes" && "$remote_bytes" -gt 0 ]]; then
    echo "Skipping $relative_path (already downloaded)"
    return 0
  fi

  echo "Downloading $relative_path"
  curl -L --fail -C - "$BASE_URL/$relative_path" -o "$destination" || status=$?

  if [[ "$status" -ne 0 ]]; then
    local_bytes=0
    if [[ -f "$destination" ]]; then
      local_bytes="$(wc -c < "$destination" | tr -d ' ')"
    fi
    if [[ "$status" -eq 22 && -n "$remote_bytes" && "$remote_bytes" =~ ^[0-9]+$ && "$local_bytes" -ge "$remote_bytes" && "$remote_bytes" -gt 0 ]]; then
      echo "Skipping $relative_path (already downloaded)"
      return 0
    fi
    return "$status"
  fi
}

download "Wan2.2_VAE.pth"
download "config.json"
download "configuration.json"
download "diffusion_pytorch_model-00001-of-00003.safetensors"
download "diffusion_pytorch_model-00002-of-00003.safetensors"
download "diffusion_pytorch_model-00003-of-00003.safetensors"
download "diffusion_pytorch_model.safetensors.index.json"
download "models_t5_umt5-xxl-enc-bf16.pth"
download "google/umt5-xxl/special_tokens_map.json"
download "google/umt5-xxl/spiece.model"
download "google/umt5-xxl/tokenizer.json"
download "google/umt5-xxl/tokenizer_config.json"

echo "Finished downloading Wan2.2-TI2V-5B into $MODEL_ROOT"
