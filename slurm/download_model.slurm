#!/bin/bash
#SBATCH --job-name=wan-download
#SBATCH --output=/path/to/wan-story-movie/logs/%x-%j.out
#SBATCH --error=/path/to/wan-story-movie/logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/path/to/wan-story-movie"
PYTHON_BIN="python3"

# Replace these with the modules on your HPC if needed.
# module purge
# module load python/3.11

export PROJECT_ROOT
export PYTHON_BIN
export MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"

"$PROJECT_ROOT/scripts/download_wan22.sh"
