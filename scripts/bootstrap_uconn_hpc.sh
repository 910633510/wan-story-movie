#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$PROJECT_ROOT/vendor/Wan2.2}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
SLURM_FILE="${SLURM_FILE:-$PROJECT_ROOT/slurm/run_story_movie_uconn.slurm}"
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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required on the HPC login node." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/vendor" "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

if [[ ! -d "$WAN_REPO_DIR/.git" ]]; then
  git clone https://github.com/Wan-Video/Wan2.2.git "$WAN_REPO_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "huggingface_hub[cli]"

if ! python -m pip install -r "$WAN_REPO_DIR/requirements.txt"; then
  echo "Full Wan requirements install failed. Retrying without flash_attn." >&2
  REQUIREMENTS_NO_FLASH="$PROJECT_ROOT/.requirements-no-flashattn.txt"
  grep -v '^flash_attn$' "$WAN_REPO_DIR/requirements.txt" > "$REQUIREMENTS_NO_FLASH"
  python -m pip install -r "$REQUIREMENTS_NO_FLASH"
  rm -f "$REQUIREMENTS_NO_FLASH"
fi

python -m pip install opencv-python
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
