from __future__ import annotations

import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXTRACTED_ROOT = Path(
    os.environ.get(
        "YOLOV11_RGBT_EXTRACTED_ROOT",
        Path.home() / "datasets" / "yolov11_datasets" / "Datasets" / "NewVersion" / "extracted",
    )
)
DEFAULT_YAML_ROOT = Path(
    os.environ.get(
        "YOLOV11_RGBT_YAML_ROOT",
        Path.home() / "datasets" / "yolov11_datasets" / "Datasets" / "NewVersion" / "yaml",
    )
)


def canonical_dataset_name(name: str) -> str:
    return name.split("-", 1)[0]


def dataset_root_from_name(dataset_name: str, extracted_root: Path = DEFAULT_EXTRACTED_ROOT) -> Path:
    name = canonical_dataset_name(dataset_name)
    return extracted_root.expanduser().resolve() / name / name


def infer_dataset_name(source_yaml: str | Path | None = None, dataset_root: str | Path | None = None) -> str:
    if source_yaml:
        return canonical_dataset_name(Path(source_yaml).stem)
    if dataset_root:
        root = Path(dataset_root).expanduser().resolve()
        if root.name == root.parent.name:
            return root.name
        return root.name
    raise ValueError("Cannot infer dataset name without source_yaml or dataset_root.")


def ensure_rgbt_layout(
    dataset_root: str | Path,
    visible_dir_name: str = "visible",
    infrared_dir_name: str = "infrared",
    labels_dir_name: str = "labels",
    train_split: str = "train",
    val_split: str = "test",
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    required_paths = [
        root / visible_dir_name / train_split,
        root / visible_dir_name / val_split,
        root / infrared_dir_name / train_split,
        root / infrared_dir_name / val_split,
        root / labels_dir_name / train_split,
        root / labels_dir_name / val_split,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("RGBT dataset layout is incomplete. Missing:\n- " + "\n- ".join(missing))
    return root


def load_yaml(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_runtime_dataset_config(
    dataset_root: str | Path,
    *,
    names: list[str],
    nc: int | None = None,
    train_split: str = "train",
    val_split: str = "test",
    test_split: str | None = None,
    visible_dir_name: str = "visible",
    infrared_dir_name: str = "infrared",
) -> dict:
    root = ensure_rgbt_layout(
        dataset_root=dataset_root,
        visible_dir_name=visible_dir_name,
        infrared_dir_name=infrared_dir_name,
        train_split=train_split,
        val_split=val_split,
    )
    config = {
        "path": str(root),
        "train": f"{visible_dir_name}/{train_split}",
        "val": f"{visible_dir_name}/{val_split}",
        "nc": nc if nc is not None else len(names),
        "names": names,
    }
    if test_split:
        config["test"] = f"{visible_dir_name}/{test_split}"
    return config


def write_runtime_yaml(config: dict, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return output


def write_runtime_yaml_from_source(
    source_yaml: str | Path,
    *,
    dataset_root: str | Path | None = None,
    output_path: str | Path | None = None,
    train_split: str | None = None,
    val_split: str | None = None,
    test_split: str | None = None,
    visible_dir_name: str = "visible",
    infrared_dir_name: str = "infrared",
) -> Path:
    src = Path(source_yaml).expanduser().resolve()
    data = load_yaml(src)
    source_path_value = str(data.get("path", "")).strip()
    dataset_name = canonical_dataset_name(Path(source_path_value).name) if source_path_value else infer_dataset_name(source_yaml=src)
    root = Path(dataset_root).expanduser().resolve() if dataset_root else dataset_root_from_name(dataset_name)
    names = list(data.get("names", []))
    nc = data.get("nc", len(names) if names else None)

    def normalize_split(value: str | None, default: str) -> str:
        if not value:
            return default
        return Path(value).name

    runtime_config = build_runtime_dataset_config(
        dataset_root=root,
        names=names,
        nc=nc,
        train_split=normalize_split(train_split or data.get("train"), "train"),
        val_split=normalize_split(val_split or data.get("val") or data.get("test"), "test"),
        test_split=normalize_split(test_split or data.get("test"), "test") if (test_split or data.get("test")) else None,
        visible_dir_name=visible_dir_name,
        infrared_dir_name=infrared_dir_name,
    )

    target = output_path
    if target is None:
        target = REPO_ROOT / "runs" / "tmp" / f"{src.stem}.runtime.yaml"
    return write_runtime_yaml(runtime_config, target)
