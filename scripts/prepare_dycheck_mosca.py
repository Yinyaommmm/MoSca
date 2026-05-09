#!/usr/bin/env python3
"""Prepare PAGE/DyCheck iPhone folders for MoSca.

The original dataset is kept read-only. RGB, cameras, test images, and
test masks are symlinked. Depth is converted from npy to MoSca's npz format.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_SCENES = [
    "apple",
    "block",
    "paper-windmill",
    "space-out",
    "spin",
    "teddy",
    "wheel",
]


def reset_link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    dst.symlink_to(src)


def convert_depth(src: Path, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        return
    arr = np.load(src)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, dep=arr.astype(np.float32, copy=False))


def prepare_scene(src_root: Path, dst_root: Path, scene: str, force_depth: bool = False) -> None:
    src = src_root / scene
    dst = dst_root / scene
    if not src.exists():
        raise FileNotFoundError(src)

    dirs = {
        "images": dst / "images",
        "cameras": dst / "cameras",
        "sensor_depth": dst / "sensor_depth",
        "test_images": dst / "test_images",
        "test_cameras": dst / "test_cameras",
        "test_covisible": dst / "test_covisible",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rgb_dir = src / "rgb" / "2x"
    depth_dir = src / "depth" / "2x"
    camera_dir = src / "camera"
    covisible_dir = src / "covisible" / "2x" / "val"

    train_rgbs = sorted(rgb_dir.glob("0_*.png"))
    if not train_rgbs:
        raise RuntimeError(f"No training RGB frames found for {scene}")

    for p in train_rgbs:
        reset_link(dirs["images"] / p.name, p)
        cam = camera_dir / f"{p.stem}.json"
        dep = depth_dir / f"{p.stem}.npy"
        if not cam.exists():
            raise FileNotFoundError(cam)
        if not dep.exists():
            raise FileNotFoundError(dep)
        reset_link(dirs["cameras"] / cam.name, cam)
        convert_depth(dep, dirs["sensor_depth"] / f"{p.stem}.npz", force=force_depth)

    test_rgbs = sorted(p for p in rgb_dir.glob("*.png") if not p.name.startswith("0_"))
    for p in test_rgbs:
        reset_link(dirs["test_images"] / p.name, p)
        cam = camera_dir / f"{p.stem}.json"
        if not cam.exists():
            raise FileNotFoundError(cam)
        reset_link(dirs["test_cameras"] / cam.name, cam)

    if covisible_dir.exists():
        for p in sorted(covisible_dir.glob("*.png")):
            reset_link(dirs["test_covisible"] / p.name, p)

    print(
        f"{scene}: train={len(train_rgbs)} test={len(test_rgbs)} "
        f"depth={len(list(dirs['sensor_depth'].glob('*.npz')))} "
        f"covisible={len(list(dirs['test_covisible'].glob('*.png')))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, default=Path("/root/autodl-tmp/datasets/dyncheck"))
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/dyncheck-mosca"),
    )
    parser.add_argument("--force-depth", action="store_true")
    parser.add_argument("scenes", nargs="*", default=DEFAULT_SCENES)
    args = parser.parse_args()

    for scene in args.scenes:
        prepare_scene(args.src_root, args.dst_root, scene, force_depth=args.force_depth)


if __name__ == "__main__":
    main()
