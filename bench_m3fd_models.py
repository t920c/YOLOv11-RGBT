#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from rgbt_dataset_runtime import dataset_root_from_name, write_runtime_yaml_from_source


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    files = [path for path in folder.rglob("*") if path.suffix.lower() in exts]
    files.sort()
    return files


def load_rgbt_tensor(vis_path: Path, ir_path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    vis = Image.open(vis_path).convert("RGB").resize((imgsz, imgsz))
    vis_np = np.array(vis, dtype=np.float32) / 255.0

    ir = Image.open(ir_path).convert("L").resize((imgsz, imgsz))
    ir_np = np.array(ir, dtype=np.float32) / 255.0

    rgbt = np.concatenate([vis_np, ir_np[..., None]], axis=2)
    return torch.from_numpy(rgbt).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def measure_fps_rgbt(
    model: YOLO,
    vis_dir: Path,
    ir_dir: Path,
    imgsz: int,
    device: torch.device,
    n_warmup: int = 10,
    n_iters: int = 100,
    stride: int = 1,
) -> tuple[float, int]:
    vis_files = list_images(vis_dir)
    if not vis_files:
        raise FileNotFoundError(f"No visible images found in: {vis_dir}")

    pairs: list[tuple[Path, Path]] = []
    for vis_path in vis_files[::stride]:
        ir_path = ir_dir / vis_path.relative_to(vis_dir)
        if ir_path.exists():
            pairs.append((vis_path, ir_path))
    if not pairs:
        for vis_path in vis_files[::stride]:
            ir_path = ir_dir / vis_path.name
            if ir_path.exists():
                pairs.append((vis_path, ir_path))
    if not pairs:
        raise FileNotFoundError(f"Cannot find matching infrared images under: {ir_dir}")

    torch_model = model.model
    torch_model.eval()
    torch_model.to(device)

    for i in range(min(n_warmup, len(pairs))):
        x = load_rgbt_tensor(pairs[i][0], pairs[i][1], imgsz, device)
        _ = torch_model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    n = 0
    for i in range(n_iters):
        vis_path, ir_path = pairs[i % len(pairs)]
        x = load_rgbt_tensor(vis_path, ir_path, imgsz, device)
        _ = torch_model(x)
        n += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    secs = max(end - start, 1e-9)
    return n / secs, n


def get_gflops_from_model(model: YOLO) -> float | None:
    try:
        value = getattr(model.model, "gflops", None)
        return float(value) if value is not None else None
    except Exception:
        return None


def find_weights(weights_dir: str, pattern: str) -> list[str]:
    files = [str(path) for path in Path(weights_dir).glob(pattern)]
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset YAML such as M3FD-rgbt.yaml.")
    parser.add_argument("--weights_dir", required=True, help="Directory containing .pt weights.")
    parser.add_argument("--pattern", default="*.pt", help="Glob pattern for weight files.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--fixed_root",
        default=str(dataset_root_from_name("M3FD")),
        help="Actual extracted dataset root on this machine.",
    )
    parser.add_argument("--csv", default="bench_m3fd_results.csv")
    parser.add_argument("--fps_warmup", type=int, default=10)
    parser.add_argument("--fps_iters", type=int, default=100)
    parser.add_argument("--fps_stride", type=int, default=1, help="Use every Nth image for FPS pairing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")

    temp_yaml = Path("/tmp") / f"m3fd_fixed_{os.getpid()}.yaml"
    fixed_yaml = write_runtime_yaml_from_source(args.data, dataset_root=args.fixed_root, output_path=temp_yaml)

    root = Path(args.fixed_root)
    vis_dir = root / "visible" / args.split
    ir_dir = root / "infrared" / args.split
    if not vis_dir.exists() and args.split == "val" and (root / "visible" / "test").exists():
        vis_dir = root / "visible" / "test"
        ir_dir = root / "infrared" / "test"
    if not vis_dir.exists():
        raise FileNotFoundError(f"Visible split not found: {vis_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"Infrared split not found: {ir_dir}")

    weights = find_weights(args.weights_dir, args.pattern)
    if not weights:
        raise FileNotFoundError(f"No weights matched in {args.weights_dir} with pattern {args.pattern}")

    print(f"Found {len(weights)} weights:")
    for weight in weights:
        print(f" - {weight}")

    rows: list[dict[str, object]] = []
    print("\n" + "=" * 90)
    print("RESULTS (GFLOPs, mAP50-95, mAP50, FPS)")
    print("=" * 90 + "\n")

    for weight in weights:
        print("-" * 90)
        print(f"WEIGHT: {weight}")

        model = YOLO(weight)
        result = model.val(
            data=str(fixed_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            split=args.split,
            channels=4,
            use_simotm="RGBT",
            pairs_rgb_ir=["visible", "infrared"],
            verbose=False,
        )

        map5095 = getattr(getattr(result, "box", None), "map", None)
        map50 = getattr(getattr(result, "box", None), "map50", None)
        gflops = get_gflops_from_model(model)
        fps, n = measure_fps_rgbt(
            model,
            vis_dir=vis_dir,
            ir_dir=ir_dir,
            imgsz=args.imgsz,
            device=device,
            n_warmup=args.fps_warmup,
            n_iters=args.fps_iters,
            stride=args.fps_stride,
        )

        print(f"mAP50-95: {map5095}")
        print(f"mAP50   : {map50}")
        print(f"GFLOPs  : {gflops} (None means not exposed; see model summary line)")
        print(f"FPS     : {fps:.2f}  (measured on {n} iters, imgsz={args.imgsz})")

        rows.append(
            {
                "weight": weight,
                "gflops": gflops if gflops is not None else "",
                "map50_95": float(map5095) if map5095 is not None else "",
                "map50": float(map50) if map50 is not None else "",
                "fps": float(fps),
            }
        )

    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["weight", "gflops", "map50_95", "map50", "fps"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved CSV:", args.csv)
    print("Done.")


if __name__ == "__main__":
    main()
