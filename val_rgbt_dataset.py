from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from rgbt_model_runtime import resolve_model_with_scale
from rgbt_dataset_runtime import REPO_ROOT, write_runtime_yaml_from_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an RGB-T YOLO checkpoint from a source YAML.")
    parser.add_argument("--source-yaml", required=True, help="Source dataset YAML.")
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument("--dataset-root", default=None, help="Extracted dataset root override.")
    parser.add_argument("--runtime-yaml", default=None, help="Where to write the generated runtime YAML.")
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml",
        help="Model YAML path used only when --model-scale is needed for consistency checks.",
    )
    parser.add_argument("--model-scale", default=None, help="Optional model scale such as n, s, m, l, x.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "rgbt"))
    parser.add_argument("--name", default="val")
    parser.add_argument("--split", default="val")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--visible-dir-name", default="visible")
    parser.add_argument("--infrared-dir-name", default="infrared")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_yaml = Path(args.runtime_yaml).expanduser().resolve() if args.runtime_yaml else None
    data_yaml = write_runtime_yaml_from_source(
        args.source_yaml,
        dataset_root=args.dataset_root,
        output_path=runtime_yaml,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        visible_dir_name=args.visible_dir_name,
        infrared_dir_name=args.infrared_dir_name,
    )

    _ = resolve_model_with_scale(args.model, args.model_scale, "val")
    model = YOLO(args.weights)
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
    print(f"model_scale={args.model_scale}")
    print(f"weights={Path(args.weights).expanduser().resolve()}")
    if hasattr(metrics, "box"):
        print(f"map50={metrics.box.map50:.6f}")
        print(f"map50_95={metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
