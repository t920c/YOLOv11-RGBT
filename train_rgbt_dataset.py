from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from rgbt_model_runtime import resolve_model_with_scale
from rgbt_dataset_runtime import REPO_ROOT, write_runtime_yaml_from_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RGB-T YOLO model from a source YAML on the current machine.")
    parser.add_argument("--source-yaml", required=True, help="Source dataset YAML.")
    parser.add_argument("--dataset-root", default=None, help="Extracted dataset root override.")
    parser.add_argument("--runtime-yaml", default=None, help="Where to write the generated runtime YAML.")
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml",
        help="Model YAML or checkpoint path.",
    )
    parser.add_argument("--model-scale", default=None, help="Optional model scale such as n, s, m, l, x.")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional pretrained checkpoint to load into the RGB-T architecture, e.g. yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "rgbt"))
    parser.add_argument("--name", default="train")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--optimizer", default=None, help="Optimizer override, e.g. SGD, AdamW, auto.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning-rate factor.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optimizer weight decay.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic in the last N epochs.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic probability.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp probability.")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation augmentation.")
    parser.add_argument("--translate", type=float, default=None, help="Translation augmentation.")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation.")
    parser.add_argument("--fliplr", type=float, default=None, help="Left-right flip probability.")
    parser.add_argument("--save-period", type=int, default=None, help="Save checkpoint every N epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR scheduler.")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training.")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--visible-dir-name", default="visible")
    parser.add_argument("--infrared-dir-name", default="infrared")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace, data_yaml: Path) -> dict:
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "channels": 4,
        "use_simotm": "RGBT",
        "pairs_rgb_ir": [args.visible_dir_name, args.infrared_dir_name],
        "project": args.project,
        "name": args.name,
        "fraction": args.fraction,
        "exist_ok": args.exist_ok,
    }

    optional_keys = {
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "save_period": args.save_period,
        "seed": args.seed,
    }
    for key, value in optional_keys.items():
        if value is not None:
            train_kwargs[key] = value

    if args.cos_lr:
        train_kwargs["cos_lr"] = True
    if args.multi_scale:
        train_kwargs["multi_scale"] = True

    return train_kwargs


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

    model_path = resolve_model_with_scale(args.model, args.model_scale, "train")
    model = YOLO(model_path)
    if args.pretrained:
        model.load(args.pretrained)

    train_kwargs = build_train_kwargs(args, data_yaml)
    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir).resolve()
    print(f"runtime_data_yaml={data_yaml}")
    print(f"model={model_path}")
    print(f"model_scale={args.model_scale}")
    print(f"pretrained={args.pretrained}")
    print(f"save_dir={save_dir}")
    print(f"best_weights={save_dir / 'weights' / 'best.pt'}")
    print(f"last_weights={save_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
