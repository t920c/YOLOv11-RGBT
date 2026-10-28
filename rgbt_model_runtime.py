from __future__ import annotations

from pathlib import Path

import yaml

from rgbt_dataset_runtime import REPO_ROOT


def resolve_model_with_scale(model: str, model_scale: str | None, tag: str) -> Path | str:
    if not model_scale:
        return model

    model_path = Path(model).expanduser()
    if model_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("--model-scale is only supported when --model points to a YAML model file.")

    resolved = model_path.resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    scales = data.get("scales", {})
    if model_scale not in scales:
        raise ValueError(f"Model scale '{model_scale}' is not defined in {resolved}. Available: {sorted(scales)}")

    data["scale"] = model_scale
    output = REPO_ROOT / "runs" / "tmp" / f"{resolved.stem}.{tag}.{model_scale}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return output
