#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$PROJECT_ROOT/vendor/Wan2.2}"
SLURM_FILE="${SLURM_FILE:-$PROJECT_ROOT/slurm/run_story_movie_uconn.slurm}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-wanmovie}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
PI_ACCOUNT="${PI_ACCOUNT:-}"
SUBMIT_JOB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      SUBMIT_JOB=true
      shift
      ;;
    --account)
      PI_ACCOUNT="$2"
      shift 2
      ;;
    --env-name)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$PROJECT_ROOT/vendor" "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

if [[ ! -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
  INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  else
    echo "Need curl or wget to install Miniconda." >&2
    exit 1
  fi
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
fi

source "$MINICONDA_DIR/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

conda activate "$CONDA_ENV_NAME"

if [[ ! -d "$WAN_REPO_DIR/.git" ]]; then
  git clone https://github.com/Wan-Video/Wan2.2.git "$WAN_REPO_DIR"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install "huggingface_hub[cli]"

python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python -m pip install \
  --index-url "$PYTORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

REQUIREMENTS_NO_TORCH="$PROJECT_ROOT/.requirements-no-torch.txt"
grep -Ev '^(torch(|vision|audio)([<>=].*)?)$' "$WAN_REPO_DIR/requirements.txt" > "$REQUIREMENTS_NO_TORCH"

if ! python -m pip install -r "$REQUIREMENTS_NO_TORCH"; then
  echo "Full Wan requirements install failed. Retrying without flash_attn." >&2
  REQUIREMENTS_NO_FLASH="$PROJECT_ROOT/.requirements-no-flashattn.txt"
  grep -v '^flash_attn$' "$REQUIREMENTS_NO_TORCH" > "$REQUIREMENTS_NO_FLASH"
  python -m pip install -r "$REQUIREMENTS_NO_FLASH"
  rm -f "$REQUIREMENTS_NO_FLASH"
fi
rm -f "$REQUIREMENTS_NO_TORCH"

python -m pip install opencv-python
python -m pip install einops
python -m pip install decord
python -m pip install librosa
"$PROJECT_ROOT/scripts/download_ti2v_model_via_curl.sh"

if [[ -n "$PI_ACCOUNT" ]]; then
  sed -i.bak "s/^#SBATCH --account=PINetidHere/#SBATCH --account=$PI_ACCOUNT/" "$SLURM_FILE"
else
  sed -i.bak '/^#SBATCH --account=PINetidHere/d' "$SLURM_FILE"
fi
rm -f "$SLURM_FILE.bak"

if $SUBMIT_JOB; then
  sbatch "$SLURM_FILE"
fi

echo "Bootstrap complete."
