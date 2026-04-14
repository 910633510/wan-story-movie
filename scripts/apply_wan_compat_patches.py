#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def patch_model_py(repo_dir: Path) -> bool:
    path = repo_dir / "wan" / "modules" / "model.py"
    return replace_once(
        path,
        "from .attention import flash_attention",
        "from .attention import attention as flash_attention",
    )


def patch_t5_py(repo_dir: Path) -> bool:
    path = repo_dir / "wan" / "modules" / "t5.py"
    text = path.read_text(encoding="utf-8")
    if "device=None," in text and "torch.cuda.is_available()" in text:
        return False

    old_signature = "        device=torch.cuda.current_device(),"
    if old_signature not in text:
        raise RuntimeError(f"Expected signature not found in {path}")
    text = text.replace(old_signature, "        device=None,")

    old_body = "        self.text_len = text_len\n        self.dtype = dtype\n        self.device = device\n"
    new_body = (
        "        self.text_len = text_len\n"
        "        self.dtype = dtype\n"
        "        if device is None:\n"
        "            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'\n"
        "        self.device = device\n"
    )
    if old_body not in text:
        raise RuntimeError(f"Expected init body not found in {path}")
    text = text.replace(old_body, new_body)
    path.write_text(text, encoding="utf-8")
    return True


def patch_wan_init(repo_dir: Path) -> bool:
    path = repo_dir / "wan" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    target = (
        "# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.\n"
        "from . import configs, distributed, modules\n"
        "from .image2video import WanI2V\n"
        "from .text2video import WanT2V\n"
        "from .textimage2video import WanTI2V\n"
        "\n"
        "try:\n"
        "    from .speech2video import WanS2V\n"
        "except ModuleNotFoundError:\n"
        "    WanS2V = None\n"
        "\n"
        "try:\n"
        "    from .animate import WanAnimate\n"
        "except ModuleNotFoundError:\n"
        "    WanAnimate = None\n"
    )
    if text == target:
        return False
    path.write_text(target, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply compatibility patches to a local Wan2.2 checkout.")
    parser.add_argument("--wan-repo", type=Path, required=True, help="Path to the Wan2.2 repo root.")
    args = parser.parse_args()

    repo_dir = args.wan_repo.resolve()
    changed = []
    for name, fn in [
        ("wan/__init__.py", patch_wan_init),
        ("wan/modules/model.py", patch_model_py),
        ("wan/modules/t5.py", patch_t5_py),
    ]:
        if fn(repo_dir):
            changed.append(name)

    if changed:
        print("Patched:")
        for name in changed:
            print(f"  - {name}")
    else:
        print("Already patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
