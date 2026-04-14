# Wan Story Movie

This project sets up a practical Wan-based pipeline for making a short movie from a story on HPC.

It is built around `Wan-AI/Wan2.2-TI2V-5B`, which the official Wan2.2 docs describe as the 5B text-image-to-video model that supports both text-to-video and image-to-video at 720P/24fps and can run on a 24GB GPU. That makes it the best fit for chained story generation on shared GPU clusters.

The continuation behavior here works like this:

1. Scene 1 is generated from text or from a user-supplied image.
2. For each following scene, the runner extracts the last frame of the prior clip.
3. That frame becomes the image prompt for the next scene, so the movie can continue shot-to-shot with better visual continuity.

This is a practical open-source approximation of "extend this video" behavior, but it is not identical to Grok's proprietary continuation model. If you later want stronger boundary control between clips, the closest official Wan feature is `Wan2.1-FLF2V-14B`, which takes both a first frame and a last frame.

## Project Layout

- `config/story.example.json`: example multi-scene story manifest
- `scripts/download_wan22.sh`: clones Wan2.2, creates a venv, installs dependencies, downloads the model
- `scripts/download_ti2v_model_via_curl.sh`: downloads the TI2V model directly with resumable `curl`
- `scripts/bootstrap_uconn_hpc.sh`: bootstraps a Python env on HPC, installs Wan deps, downloads the model, and can submit the job
- `scripts/story_to_movie.py`: runs a story scene-by-scene and chains clips using the last frame
- `scripts/run_story_movie.sh`: thin wrapper around the story runner
- `slurm/run_story_movie.slurm`: sample SLURM job for generation
- `slurm/run_story_movie_uconn.slurm`: UConn Storrs HPC batch script for a GPU job
- `slurm/download_model.slurm`: sample SLURM job for cloning and downloading the model

## Quick Start

Clone the repo + install deps + download the model:

```bash
cd /Users/xue/Project/wan-story-movie
./scripts/download_wan22.sh
```

If your machine or cluster has trouble with `huggingface-cli`, use the direct-resume downloader instead:

```bash
cd /Users/xue/Project/wan-story-movie
./scripts/download_ti2v_model_via_curl.sh
```

Run the example story on one GPU:

```bash
cd /Users/xue/Project/wan-story-movie
source .venv/bin/activate
./scripts/run_story_movie.sh
```

Run with multiple GPUs:

```bash
cd /Users/xue/Project/wan-story-movie
source .venv/bin/activate
GPU_COUNT=4 ./scripts/run_story_movie.sh
```

## Story Format

The story file is JSON with top-level defaults and a `scenes` array.

Example scene fields:

- `id`: stable scene identifier
- `prompt`: scene prompt passed to Wan
- `continue_from_previous`: when `true`, the runner seeds this clip from the previous clip's final frame
- `image`: optional seed image path for the scene
- `size`: optional override, default `1280*704`
- `frame_num`: optional override, default `121`
- `seed`: optional deterministic seed
- `use_prompt_extend`: optional per-scene flag
- `prompt_extend_method`: usually `local_qwen` or `dashscope`
- `prompt_extend_model`: for example `Qwen/Qwen2.5-7B-Instruct`

## HPC Notes

- Replace the placeholder module lines in the SLURM scripts with your cluster's CUDA and Python modules.
- The SLURM templates use one GPU by default because the TI2V-5B model is the most practical starting point.
- If your cluster has larger GPU nodes, you can set `--gpus` above `1` and the runner will switch to `torchrun` with `--dit_fsdp --t5_fsdp --ulysses_size`.
- The official Wan2.2 docs note that `flash_attn` may fail depending on your environment. If that happens, install the other requirements first and then install a CUDA-compatible `flash_attn` build separately.
- The direct `curl` downloader supports resume with `-C -`, which is useful on HPC systems where long downloads may be interrupted.

## UConn Storrs HPC

The cleanest way to run this on UConn Storrs HPC is:

1. Transfer the whole project to your HPC home directory.
2. Install Miniconda in your account if you have not already.
3. Create a dedicated Wan environment.
4. Test interactively on `general-gpu`.
5. Submit the real job with `sbatch`.

### 1. Log in

```bash
ssh your_netid@hpc2.storrs.hpc.uconn.edu
```

If `hpc2` is flaky, UConn says you can try `login4.storrs.hpc.uconn.edu`, `login5.storrs.hpc.uconn.edu`, or `login6.storrs.hpc.uconn.edu`.

### 2. Copy the project to HPC

From your laptop or desktop:

```bash
scp -r /Users/xue/Project/wan-story-movie your_netid@hpc2.storrs.hpc.uconn.edu:~
```

### 3. Install Miniconda on HPC

UConn's current docs recommend Miniconda under your home directory:

```bash
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
export PATH=$HOME/miniconda3/bin:$PATH
conda init bash
```

Log out, then log back in.

### 4. Create the environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n wanmovie python=3.10 -y
conda activate wanmovie
cd ~/wan-story-movie
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r vendor/Wan2.2/requirements.txt
python -m pip install opencv-python
```

If `flash_attn` fails, install the other packages first and then come back to it. That is consistent with the official Wan guidance.

### 5. Download the model on HPC

```bash
cd ~/wan-story-movie
./scripts/download_ti2v_model_via_curl.sh
```

That script is resumable, so if your connection drops, just run it again.

### 6. Test interactively on a GPU node

UConn's current examples use the `general-gpu` partition for GPU interactive work:

```bash
srun -n 1 -N 1 -p general-gpu --gres=gpu:1 --time=02:00:00 --pty bash
```

Then inside that interactive job:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wanmovie
cd ~/wan-story-movie
python scripts/story_to_movie.py \
  --story config/story.example.json \
  --wan-repo vendor/Wan2.2 \
  --ckpt-dir models/Wan2.2-TI2V-5B \
  --output-dir outputs/interactive-test \
  --gpus 1 \
  --concat
```

### 7. Submit the real batch job

Edit `slurm/run_story_movie_uconn.slurm` so `PROJECT_ROOT` points at your home directory copy, then:

```bash
cd ~/wan-story-movie
sbatch slurm/run_story_movie_uconn.slurm
```

If UConn returns `Invalid account or account/partition combination`, set `#SBATCH --account=PINetidHere` to your lab's actual PI account.

Useful status commands:

```bash
squeue -u your_netid
tail -f ~/wan-story-movie/logs/wan-story-<jobid>.out
```

### One-line bootstrap

If you just want one command to paste after logging into HPC:

If your group requires a PI account:

```bash
bash -lc 'if [ -d "$HOME/wan-story-movie/.git" ]; then git -C "$HOME/wan-story-movie" pull; else git clone https://github.com/910633510/wan-story-movie.git "$HOME/wan-story-movie"; fi; cd "$HOME/wan-story-movie"; PI_ACCOUNT=YOUR_PI_ACCOUNT ./scripts/bootstrap_uconn_hpc.sh --submit'
```

If your group does not require an account line:

```bash
bash -lc 'if [ -d "$HOME/wan-story-movie/.git" ]; then git -C "$HOME/wan-story-movie" pull; else git clone https://github.com/910633510/wan-story-movie.git "$HOME/wan-story-movie"; fi; cd "$HOME/wan-story-movie"; ./scripts/bootstrap_uconn_hpc.sh --submit'
```

## Sources

- Official Wan2.2 repo: https://github.com/Wan-Video/Wan2.2
- Wan2.2 README: https://github.com/Wan-Video/Wan2.2/blob/main/README.md
- TI2V-5B model: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Wan2.1 VACE / FLF2V reference: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- UConn Storrs HPC overview: https://uconn.atlassian.net/wiki/spaces/SH/overview
- UConn Miniconda setup: https://uconn.atlassian.net/wiki/spaces/SH/pages/26079723879/
