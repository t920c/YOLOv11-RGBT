from __future__ import annotations

import os
from pathlib import Path

import yaml

from rgbt_dataset_runtime import REPO_ROOT, build_runtime_dataset_config, ensure_rgbt_layout


DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "YOLOV11_RGBT_DATASET_ROOT",
        Path.home() / "datasets" / "yolov11_datasets" / "Datasets" / "NewVersion" / "extracted" / "KAIST8" / "KAIST8",
    )
)
DEFAULT_RUNTIME_YAML = REPO_ROOT / "runs" / "tmp" / "KAIST8-rgbt.runtime.yaml"
LAST_RUN_FILE = REPO_ROOT / "runs" / "tmp" / "kaist_last_run.txt"


def ensure_kaist_layout(
    dataset_root: Path,
    visible_dir_name: str = "visible",
    infrared_dir_name: str = "infrared",
) -> Path:
    return ensure_rgbt_layout(dataset_root, visible_dir_name, infrared_dir_name)


def write_runtime_yaml(
    dataset_root: Path,
    output_path: Path = DEFAULT_RUNTIME_YAML,
    visible_dir_name: str = "visible",
    infrared_dir_name: str = "infrared",
) -> Path:
    dataset_root = ensure_kaist_layout(dataset_root, visible_dir_name, infrared_dir_name)
    config = build_runtime_dataset_config(
        dataset_root=dataset_root,
        names=["person"],
        nc=1,
        train_split="train",
        val_split="test",
        test_split="test",
        visible_dir_name=visible_dir_name,
        infrared_dir_name=infrared_dir_name,
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return output_path


def remember_last_run(save_dir: Path) -> None:
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(str(save_dir.expanduser().resolve()) + "\n", encoding="utf-8")


def find_latest_weights(explicit_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_weights = os.environ.get("YOLOV11_RGBT_WEIGHTS")
    if env_weights:
        candidates.append(Path(env_weights).expanduser())
    if LAST_RUN_FILE.exists():
        last_run = Path(LAST_RUN_FILE.read_text(encoding="utf-8").strip()).expanduser()
        candidates.extend([last_run / "weights" / "best.pt", last_run / "weights" / "last.pt"])

    runs_root = REPO_ROOT / "runs"
    if runs_root.exists():
        discovered = sorted(runs_root.glob("**/weights/*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        candidates.extend(discovered)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "No validation weights were found. Run train_kaist.py first or pass --weights / set YOLOV11_RGBT_WEIGHTS."
    )
