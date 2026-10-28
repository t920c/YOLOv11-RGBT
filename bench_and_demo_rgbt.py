from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from rgbt_dataset_runtime import write_runtime_yaml_from_source


def parse_gflops_from_info(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GFLOPs", text)
    return float(match.group(1)) if match else None


def find_pair_ir(vis_path: Path, vis_root: Path, ir_root: Path) -> Path:
    return ir_root / vis_path.relative_to(vis_root)


def load_rgbt_tensor(vis_path: Path, ir_path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
    if vis is None:
        raise FileNotFoundError(f"Visible image not found: {vis_path}")

    ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
    if ir is None:
        raise FileNotFoundError(f"Infrared image not found: {ir_path}")
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

    vis_resized = cv2.resize(vis, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    ir_resized = cv2.resize(ir, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(vis_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ir1 = (ir_resized.astype(np.float32) / 255.0)[..., None]
    rgbt = np.concatenate([rgb, ir1], axis=2)
    tensor = torch.from_numpy(rgbt).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def draw_comparison_results(vis_path: Path, ir_path: Path, boxes, out_path: Path) -> None:
    vis_img = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
    ir_gray = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
    if vis_img is None or ir_gray is None:
        return

    if ir_gray.ndim == 3:
        ir_img = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    else:
        ir_img = cv2.cvtColor(cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    if ir_img is None:
        return

    h, w = vis_img.shape[:2]
    ir_img = cv2.resize(ir_img, (w, h))

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else None
        cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
        x1, y1, x2, y2 = xyxy.tolist()

        for image in (vis_img, ir_img):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls}:{conf:.2f}" if conf is not None else f"{cls}"
            cv2.putText(image, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    comparison = np.hstack([vis_img, ir_img])
    cv2.putText(comparison, "Visible", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Infrared", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), comparison)


def measure_fps_model_only(model: YOLO, device: torch.device, imgsz: int, channels: int, batch: int, iters: int) -> float:
    model.model.to(device)
    model.model.eval()

    x = torch.randn(batch, channels, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model.model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model.model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    return (batch * iters) / (t1 - t0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Source dataset YAML. A local runtime YAML will be generated.")
    parser.add_argument("--dataset-root", default=None, help="Override extracted dataset root for this machine.")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--use_simotm", default="RGBT")
    parser.add_argument("--pairs_rgb_ir", nargs=2, default=["visible", "infrared"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--n_images", type=int, default=12)
    parser.add_argument("--out_dir", default="runs/bench_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if args.device != "cpu" else "cpu")
    model = YOLO(args.weights)
    runtime_yaml = write_runtime_yaml_from_source(
        args.data,
        dataset_root=args.dataset_root,
        visible_dir_name=args.pairs_rgb_ir[0],
        infrared_dir_name=args.pairs_rgb_ir[1],
    )

    try:
        info_str = str(model.info(verbose=False))
        gflops = parse_gflops_from_info(info_str)
    except Exception:
        gflops = None

    results = model.val(
        data=str(runtime_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        channels=args.channels,
        use_simotm=args.use_simotm,
        pairs_rgb_ir=args.pairs_rgb_ir,
        verbose=False,
    )
    map50 = float(results.box.map50)
    map5095 = float(results.box.map)
    fps = measure_fps_model_only(model, device, args.imgsz, args.channels, args.batch, args.iters)

    print("=" * 90)
    print(f"Weights  : {args.weights}")
    print(f"GFLOPs   : {gflops}")
    print(f"mAP50    : {map50:.6f}")
    print(f"mAP50-95 : {map5095:.6f}")
    print(f"FPS      : {fps:.2f} (batch={args.batch})")
    print(f"runtime_data_yaml: {runtime_yaml}")
    print("=" * 90)

    with open(runtime_yaml, "r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    root = Path(data_cfg.get("path", ""))
    split_key = "val" if args.split == "val" else "test"
    vis_split = data_cfg.get(split_key, "visible/test")

    vis_root = (root / vis_split).resolve()
    ir_root = Path(str(vis_root).replace(args.pairs_rgb_ir[0], args.pairs_rgb_ir[1]))
    vis_imgs = sorted(path for path in vis_root.rglob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png"))

    target_stem = "set11_V000_I01239"
    target_img = [path for path in vis_imgs if path.stem == target_stem]

    random.seed(0)
    sample = random.sample(vis_imgs, k=min(args.n_images, len(vis_imgs)))
    if target_img and target_img[0] not in sample:
        sample[0] = target_img[0]

    out_img_dir = Path(args.out_dir) / "demo_diagnostic"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for vis_path in sample:
        ir_path = find_pair_ir(vis_path, vis_root, ir_root)
        if not ir_path.exists():
            continue

        im = load_rgbt_tensor(vis_path, ir_path, imgsz=args.imgsz, device=device)
        with torch.no_grad():
            prediction = model.predict(im, verbose=False, channels=args.channels, use_simotm=args.use_simotm, conf=0.1)[0]

        out_path = out_img_dir / f"{vis_path.stem}_compare.jpg"
        draw_comparison_results(vis_path, ir_path, prediction.boxes, out_path)

    print(f"[OK] Diagnostic images saved to: {out_img_dir}")


if __name__ == "__main__":
    main()
