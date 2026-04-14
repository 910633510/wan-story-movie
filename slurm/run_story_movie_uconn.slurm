#!/bin/bash
#SBATCH --job-name=wan-story
#SBATCH --output=/home/%u/wan-story-movie/logs/wan-story-%j.out
#SBATCH --error=/home/%u/wan-story-movie/logs/wan-story-%j.err
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=epyc64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=PINetidHere

set -euo pipefail

# Replace PINetidHere with your PI account if your group requires it.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/wan-story-movie}"
WAN_REPO_DIR="$PROJECT_ROOT/vendor/Wan2.2"
MODEL_DIR="$PROJECT_ROOT/models/Wan2.2-TI2V-5B"
STORY_JSON="${STORY_JSON:-$PROJECT_ROOT/config/story.example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/$SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wanmovie

cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/scripts/story_to_movie.py" \
  --story "$STORY_JSON" \
  --wan-repo "$WAN_REPO_DIR" \
  --ckpt-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gpus 1 \
  --concat
