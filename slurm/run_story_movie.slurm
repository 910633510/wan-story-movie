#!/bin/bash
#SBATCH --job-name=wan-story
#SBATCH --output=/path/to/wan-story-movie/logs/%x-%j.out
#SBATCH --error=/path/to/wan-story-movie/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/path/to/wan-story-movie"
WAN_REPO_DIR="$PROJECT_ROOT/vendor/Wan2.2"
MODEL_DIR="$PROJECT_ROOT/models/Wan2.2-TI2V-5B"
STORY_JSON="$PROJECT_ROOT/config/story.example.json"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$SLURM_JOB_ID"
VENV_DIR="$PROJECT_ROOT/.venv"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"

# Replace these with the modules on your HPC.
# module purge
# module load cuda/12.4
# module load python/3.11

source "$VENV_DIR/bin/activate"

python "$PROJECT_ROOT/scripts/story_to_movie.py" \
  --story "$STORY_JSON" \
  --wan-repo "$WAN_REPO_DIR" \
  --ckpt-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gpus 1 \
  --concat
