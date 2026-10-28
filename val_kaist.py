from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from kaist_runtime import DEFAULT_DATASET_ROOT, REPO_ROOT, find_latest_weights, write_runtime_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate KAIST8 RGBT weights on this machine.")
    parser.add_argument("--weights", default=None, help="Checkpoint path. Defaults to the latest run under runs/.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="KAIST dataset root directory.")
    parser.add_argument(
        "--data-yaml",
        default=str(REPO_ROOT / "runs" / "tmp" / "KAIST8-rgbt.runtime.yaml"),
        help="Path to the generated runtime dataset YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "kaist"))
    parser.add_argument("--name", default="val")
    parser.add_argument("--split", default="val")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--visible-dir-name", default="visible")
    parser.add_argument("--infrared-dir-name", default="infrared")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = write_runtime_yaml(
        dataset_root=Path(args.dataset_root),
        output_path=Path(args.data_yaml),
        visible_dir_name=args.visible_dir_name,
        infrared_dir_name=args.infrared_dir_name,
    )
    weights = find_latest_weights(args.weights)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        channels=4,
        use_simotm="RGBT",
        pairs_rgb_ir=[args.visible_dir_name, args.infrared_dir_name],
        project=args.project,
        name=args.name,
        split=args.split,
        fraction=args.fraction,
        exist_ok=args.exist_ok,
    )

    print(f"runtime_data_yaml={data_yaml}")
    print(f"weights={weights}")
    if hasattr(metrics, "box"):
        print(f"map50={metrics.box.map50:.6f}")
        print(f"map50_95={metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
