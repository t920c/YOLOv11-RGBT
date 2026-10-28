from __future__ import annotations

import argparse
from pathlib import Path

from rgbt_dataset_runtime import (
    build_runtime_dataset_config,
    dataset_root_from_name,
    infer_dataset_name,
    write_runtime_yaml,
    write_runtime_yaml_from_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local runtime YAML for an extracted RGBT dataset.")
    parser.add_argument("--source-yaml", default=None, help="Existing dataset YAML to copy names/nc from.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name such as KAIST8, FLIR_aligned, or M3FD.")
    parser.add_argument("--dataset-root", default=None, help="Extracted dataset root. Auto-derived when omitted.")
    parser.add_argument("--output", required=True, help="Where to write the generated YAML.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--visible-dir-name", default="visible")
    parser.add_argument("--infrared-dir-name", default="infrared")
    parser.add_argument("--names", nargs="*", default=None, help="Class names, required when --source-yaml is omitted.")
    parser.add_argument("--nc", type=int, default=None, help="Class count, defaults to len(names).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)

    if args.source_yaml:
        runtime_yaml = write_runtime_yaml_from_source(
            args.source_yaml,
            dataset_root=args.dataset_root,
            output_path=output,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            visible_dir_name=args.visible_dir_name,
            infrared_dir_name=args.infrared_dir_name,
        )
        print(runtime_yaml)
        return

    dataset_name = args.dataset_name or infer_dataset_name(dataset_root=args.dataset_root)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else dataset_root_from_name(dataset_name)
    if not args.names:
        raise ValueError("--names is required when --source-yaml is omitted.")

    config = build_runtime_dataset_config(
        dataset_root=dataset_root,
        names=args.names,
        nc=args.nc,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        visible_dir_name=args.visible_dir_name,
        infrared_dir_name=args.infrared_dir_name,
    )
    runtime_yaml = write_runtime_yaml(config, output)
    print(runtime_yaml)


if __name__ == "__main__":
    main()
