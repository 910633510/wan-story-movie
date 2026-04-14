#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a sequence of Wan2.2 clips and chain them into a short movie."
    )
    parser.add_argument(
        "--story",
        type=Path,
        default=project_root / "config" / "story.example.json",
        help="Path to a story JSON file.",
    )
    parser.add_argument(
        "--wan-repo",
        type=Path,
        default=project_root / "vendor" / "Wan2.2",
        help="Path to the cloned Wan2.2 repository.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=project_root / "models" / "Wan2.2-TI2V-5B",
        help="Path to the Wan2.2 TI2V checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "latest",
        help="Directory for generated clips and metadata.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable inside the Wan environment.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use. If >1, torchrun + FSDP/Ulysses is used.",
    )
    parser.add_argument(
        "--task",
        default="ti2v-5B",
        choices=["ti2v-5B"],
        help="Wan task to run. This scaffold targets the TI2V-5B model.",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Concatenate scene clips into a single movie if ffmpeg is available.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenes whose output files already exist.",
    )
    return parser.parse_args()


def load_story(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "scenes" not in data or not isinstance(data["scenes"], list) or not data["scenes"]:
        raise ValueError(f"{path} must contain a non-empty 'scenes' list")
    return data


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.story.exists():
        raise FileNotFoundError(f"Story file not found: {args.story}")
    if not args.wan_repo.exists():
        raise FileNotFoundError(f"Wan repo not found: {args.wan_repo}")
    if not (args.wan_repo / "generate.py").exists():
        raise FileNotFoundError(f"generate.py not found inside {args.wan_repo}")
    if not args.ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)


def extract_last_frame(video_path: Path, image_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"No frames found in video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read last frame from {video_path}")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Unable to write last frame to {image_path}")


def build_command(
    args: argparse.Namespace,
    story: dict[str, Any],
    scene: dict[str, Any],
    index: int,
    output_video: Path,
    image_path: Path | None,
) -> list[str]:
    scene_size = scene.get("size", story.get("default_size", "1280*704"))
    scene_seed = int(scene.get("seed", story.get("default_seed", 0) + index))
    frame_num = int(scene.get("frame_num", story.get("default_frame_num", 121)))

    if args.gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.gpus}",
            "generate.py",
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size",
            str(args.gpus),
        ]
    else:
        cmd = [
            args.python_bin,
            "generate.py",
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
        ]

    cmd.extend(
        [
            "--task",
            args.task,
            "--size",
            scene_size,
            "--ckpt_dir",
            str(args.ckpt_dir),
            "--prompt",
            scene["prompt"],
            "--frame_num",
            str(frame_num),
            "--base_seed",
            str(scene_seed),
            "--save_file",
            str(output_video),
        ]
    )

    if image_path is not None:
        cmd.extend(["--image", str(image_path)])

    use_prompt_extend = bool(scene.get("use_prompt_extend", story.get("use_prompt_extend", False)))
    if use_prompt_extend:
        cmd.append("--use_prompt_extend")
        cmd.extend(
            [
                "--prompt_extend_method",
                scene.get("prompt_extend_method", story.get("prompt_extend_method", "local_qwen")),
                "--prompt_extend_model",
                scene.get("prompt_extend_model", story.get("prompt_extend_model", "Qwen/Qwen2.5-7B-Instruct")),
            ]
        )

    if "sample_steps" in scene:
        cmd.extend(["--sample_steps", str(int(scene["sample_steps"]))])
    if "sample_shift" in scene:
        cmd.extend(["--sample_shift", str(float(scene["sample_shift"]))])
    if "sample_guide_scale" in scene:
        cmd.extend(["--sample_guide_scale", str(float(scene["sample_guide_scale"]))])

    return cmd


def concat_videos(clips: list[Path], output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; skipping concatenation.", file=sys.stderr)
        return False

    list_file = output_path.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{clip.resolve()}'\n" for clip in clips),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def main() -> int:
    args = parse_args()
    story = load_story(args.story)
    ensure_paths(args)

    clips: list[Path] = []
    manifest: dict[str, Any] = {
        "movie_title": story.get("movie_title", "wan-story-movie"),
        "story_file": str(args.story.resolve()),
        "wan_repo": str(args.wan_repo.resolve()),
        "checkpoint_dir": str(args.ckpt_dir.resolve()),
        "gpus": args.gpus,
        "clips": [],
    }

    previous_clip: Path | None = None
    for index, scene in enumerate(story["scenes"], start=1):
        scene_id = scene.get("id", f"scene-{index:02d}")
        output_video = args.output_dir / f"{index:02d}_{scene_id}.mp4"
        input_image: Path | None = None

        if previous_clip is not None and bool(scene.get("continue_from_previous", False)):
            input_image = args.output_dir / "frames" / f"{index:02d}_{scene_id}_seed.png"
            extract_last_frame(previous_clip, input_image)
        elif "image" in scene:
            input_image = Path(scene["image"]).expanduser().resolve()

        if args.resume and output_video.exists():
            print(f"Skipping existing clip: {output_video}")
        else:
            cmd = build_command(args, story, scene, index, output_video, input_image)
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=args.wan_repo, check=True)

        clips.append(output_video)
        previous_clip = output_video
        manifest["clips"].append(
            {
                "scene_id": scene_id,
                "prompt": scene["prompt"],
                "output_video": str(output_video.resolve()),
                "input_image": str(input_image.resolve()) if input_image else None,
                "continued_from_previous": bool(scene.get("continue_from_previous", False)),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if args.concat and clips:
        movie_path = args.output_dir / "movie.mp4"
        if concat_videos(clips, movie_path):
            print(f"Wrote concatenated movie: {movie_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
