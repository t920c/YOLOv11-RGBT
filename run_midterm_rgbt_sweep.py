from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from ultralytics import YOLO

from rgbt_model_runtime import resolve_model_with_scale
from rgbt_dataset_runtime import REPO_ROOT, write_runtime_yaml_from_source


@dataclass
class SweepPreset:
    name: str
    purpose: str
    overrides: dict


PRESETS = [
    SweepPreset(
        name="01_scratch_minimal",
        purpose="Pure baseline without pretrained weights and with conservative augmentation.",
        overrides={
            "pretrained": None,
            "imgsz": 320,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "mosaic": 0.0,
            "mixup": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.2,
            "fliplr": 0.0,
        },
    ),
    SweepPreset(
        name="02_add_pretrained",
        purpose="Add YOLO11n pretrained weights while keeping the same simple training setup.",
        overrides={"pretrained": "yolo11n.pt"},
    ),
    SweepPreset(
        name="03_enable_default_aug",
        purpose="Enable stronger augmentation to improve generalization beyond the simple baseline.",
        overrides={
            "mosaic": 1.0,
            "mixup": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "close_mosaic": 10,
        },
    ),
    SweepPreset(
        name="04_raise_imgsz_640",
        purpose="Increase input resolution so small pedestrians are easier to preserve.",
        overrides={"imgsz": 640},
    ),
    SweepPreset(
        name="05_switch_adamw",
        purpose="Switch to AdamW, which is often a stable next optimization step after the baseline.",
        overrides={
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    ),
    SweepPreset(
        name="06_enable_cosine_lr",
        purpose="Use cosine LR scheduling for a smoother late-training optimization curve.",
        overrides={"cos_lr": True},
    ),
    SweepPreset(
        name="07_reduce_final_mosaic",
        purpose="Close mosaic earlier so the final stage fits real image statistics more closely.",
        overrides={"close_mosaic": 15, "mosaic": 0.7},
    ),
    SweepPreset(
        name="08_tune_regularization",
        purpose="Slightly increase weight decay and lower the initial LR for steadier convergence.",
        overrides={
            "lr0": 0.0007,
            "lrf": 0.05,
            "weight_decay": 0.001,
        },
    ),
    SweepPreset(
        name="09_enable_multiscale",
        purpose="Enable multi-scale training to improve robustness across object sizes.",
        overrides={"multi_scale": True},
    ),
    SweepPreset(
        name="10_final_candidate",
        purpose="Final candidate that keeps the stronger settings while reducing overly aggressive augmentation.",
        overrides={
            "mosaic": 0.5,
            "translate": 0.05,
            "scale": 0.4,
            "fliplr": 0.5,
            "lr0": 0.0005,
            "lrf": 0.05,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 10-stage RGB-T optimization sweep for midterm checkpoints.",
    )
    parser.add_argument("--source-yaml", required=True, help="Source dataset YAML.")
    parser.add_argument("--dataset-root", default=None, help="Extracted dataset root override.")
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml",
        help="RGB-T model YAML.",
    )
    parser.add_argument("--model-scale", default=None, help="Optional model scale such as n, s, m, l, x.")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "midterm_rgbt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visible-dir-name", default="visible")
    parser.add_argument("--infrared-dir-name", default="infrared")
    parser.add_argument("--start", type=int, default=1, help="1-based preset start index.")
    parser.add_argument("--end", type=int, default=10, help="1-based preset end index.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running training.")
    return parser.parse_args()


def build_base_kwargs(args: argparse.Namespace, data_yaml: Path) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": 320,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "channels": 4,
        "use_simotm": "RGBT",
        "pairs_rgb_ir": [args.visible_dir_name, args.infrared_dir_name],
        "project": args.project,
        "fraction": args.fraction,
        "exist_ok": True,
        "seed": args.seed,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "mosaic": 0.0,
        "mixup": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.2,
        "fliplr": 0.0,
    }


def adapt_pretrained_name(pretrained: str | None, model_scale: str | None) -> str | None:
    if pretrained is None:
        return None
    if not model_scale:
        return pretrained
    if pretrained.startswith("yolo11") and pretrained.endswith(".pt"):
        return f"yolo11{model_scale}.pt"
    return pretrained


def apply_overrides(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        merged[key] = value
    return merged


def manifest_path(project: Path) -> Path:
    return project.expanduser().resolve() / "sweep_manifest.json"


def main() -> None:
    args = parse_args()
    data_yaml = write_runtime_yaml_from_source(
        args.source_yaml,
        dataset_root=args.dataset_root,
        output_path=REPO_ROOT / "runs" / "tmp" / f"{Path(args.source_yaml).stem}.midterm.runtime.yaml",
        visible_dir_name=args.visible_dir_name,
        infrared_dir_name=args.infrared_dir_name,
    )
    project = Path(args.project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    base_kwargs = build_base_kwargs(args, data_yaml)
    selected = PRESETS[max(1, args.start) - 1 : min(len(PRESETS), args.end)]
    if not selected:
        raise ValueError("No presets were selected. Check --start and --end.")

    model_path = resolve_model_with_scale(args.model, args.model_scale, "midterm")

    manifest = {
        "source_yaml": args.source_yaml,
        "dataset_root": args.dataset_root,
        "runtime_yaml": str(data_yaml),
        "model": str(model_path),
        "model_scale": args.model_scale,
        "project": str(project),
        "presets": [],
    }

    current = deepcopy(base_kwargs)
    for preset in selected:
        overrides = deepcopy(preset.overrides)
        if "pretrained" in overrides:
            overrides["pretrained"] = adapt_pretrained_name(overrides["pretrained"], args.model_scale)
        current = apply_overrides(current, overrides)
        run_kwargs = deepcopy(current)
        pretrained = run_kwargs.pop("pretrained", None)
        run_kwargs["name"] = preset.name

        manifest["presets"].append(
            {
                "name": preset.name,
                "purpose": preset.purpose,
                "pretrained": pretrained,
                "train_kwargs": run_kwargs,
            }
        )

    manifest_file = manifest_path(project)
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest={manifest_file}")

    if args.dry_run:
        for item in manifest["presets"]:
            print(f"{item['name']}: {item['purpose']}")
            print(json.dumps(item["train_kwargs"], ensure_ascii=False, indent=2))
            print(f"pretrained={item['pretrained']}")
        return

    for item in manifest["presets"]:
        print(f"=== Running {item['name']} ===")
        print(item["purpose"])
        model = YOLO(model_path)
        if item["pretrained"]:
            model.load(item["pretrained"])
        results = model.train(**item["train_kwargs"])
        save_dir = Path(results.save_dir).resolve()
        summary = {
            "name": item["name"],
            "purpose": item["purpose"],
            "save_dir": str(save_dir),
            "best_weights": str(save_dir / "weights" / "best.pt"),
            "last_weights": str(save_dir / "weights" / "last.pt"),
        }
        summary_path = save_dir / "midterm_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
