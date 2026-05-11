#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "scene",
    "run",
    "num_eval_frames",
    "psnr",
    "ssim",
    "lpips",
    "mpsnr",
    "mssim",
    "mlpips",
    "pck@0.05",
    "fps",
    "metrics_file",
]


def parse_scalar_file(path: Path, pattern: str):
    if not path.exists():
        return ""
    match = re.search(pattern, path.read_text())
    return float(match.group(1)) if match else ""


def find_latest_complete_run(scene_dir: Path):
    complete_runs = []
    for run_dir in scene_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "tto_dycheck_metrics.xlsx"
        pck_file = run_dir / "pck5.txt"
        fps_file = run_dir / "fps_eval.txt"
        if metrics_file.exists() and pck_file.exists() and fps_file.exists():
            complete_runs.append(run_dir)
    if not complete_runs:
        return None
    return max(complete_runs, key=lambda p: p.stat().st_mtime)


def collect_scene_metrics(scene_dir: Path):
    run_dir = find_latest_complete_run(scene_dir)
    if run_dir is None:
        return None

    metrics_file = run_dir / "tto_dycheck_metrics.xlsx"
    df = pd.read_excel(metrics_file)
    ave = df[df["fn"].astype(str).str.upper() == "AVE"]
    if ave.empty:
        ave_row = {
            "psnr": df["psnr"].mean(),
            "ssim": df["ssim"].mean(),
            "lpips": df["lpips"].mean(),
            "mpsnr": df["mpsnr"].mean(),
            "mssim": df["mssim"].mean(),
            "mlpips": df["mlpips"].mean(),
        }
    else:
        ave_row = ave.iloc[0].to_dict()

    num_eval_frames = len(df[df["fn"].astype(str).str.upper() != "AVE"])
    pck = parse_scalar_file(run_dir / "pck5.txt", r"PCK@0\.05:\s*([0-9.eE+-]+)")
    fps = parse_scalar_file(run_dir / "fps_eval.txt", r"FPS:\s*([0-9.eE+-]+)")

    return {
        "scene": scene_dir.name,
        "run": run_dir.name,
        "num_eval_frames": num_eval_frames,
        "psnr": float(ave_row["psnr"]),
        "ssim": float(ave_row["ssim"]),
        "lpips": float(ave_row["lpips"]),
        "mpsnr": float(ave_row["mpsnr"]),
        "mssim": float(ave_row["mssim"]),
        "mlpips": float(ave_row["mlpips"]),
        "pck@0.05": pck,
        "fps": fps,
        "metrics_file": str(metrics_file),
    }


def append_average(rows):
    if not rows:
        return rows
    avg = {"scene": "AVERAGE", "run": "", "metrics_file": ""}
    for key in METRIC_COLUMNS:
        if key in {"scene", "run", "metrics_file"}:
            continue
        vals = [row[key] for row in rows if row[key] != ""]
        avg[key] = sum(vals) / len(vals) if vals else ""
    return rows + [avg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/code/MoSca/output/sam3_mask"),
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=[
            "apple",
            "block",
            "paper-windmill",
            "space-out",
            "spin",
            "teddy",
            "wheel",
        ],
    )
    args = parser.parse_args()

    rows = []
    status_rows = []
    for scene in args.scenes:
        scene_dir = args.output_root / scene
        if not scene_dir.is_dir():
            status_rows.append({"scene": scene, "status": "missing"})
            continue
        row = collect_scene_metrics(scene_dir)
        if row is None:
            status_rows.append({"scene": scene, "status": "incomplete"})
            continue
        rows.append(row)
        status_rows.append({"scene": scene, "status": "complete"})

    rows_with_avg = append_average(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "all.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_with_avg)

    with (args.output_root / "run_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "status"])
        writer.writeheader()
        writer.writerows(status_rows)

    print(args.output_root / "all.csv")
    if rows_with_avg:
        print(pd.DataFrame(rows_with_avg).to_string(index=False))


if __name__ == "__main__":
    main()
