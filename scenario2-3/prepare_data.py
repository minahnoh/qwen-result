#!/usr/bin/env python3
"""Link/copy images and bundled case2 train/validation annotations."""

import argparse
import shutil
from pathlib import Path


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Root containing train/ and val/ images")
    parser.add_argument("--case2-train", type=Path,
                        default=Path(__file__).resolve().parent / "annotations/case2_train",
                        help="Bundled case2 training annotations")
    parser.add_argument("--case2-val", type=Path,
                        default=Path(__file__).resolve().parent / "annotations/case2_val",
                        help="Bundled case2 validation annotations")
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--copy", action="store_true", help="Copy data instead of symlinking")
    return parser.parse_args()


def replace(target: Path, source: Path, copy: bool):
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, target)
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def main():
    options = args()
    source = options.source.resolve()
    destination = options.destination.resolve()
    case2_train = options.case2_train.resolve()
    case2_val = options.case2_val.resolve()
    required = [source / "train", source / "val", case2_train, case2_val]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing source directories: " + ", ".join(missing))

    destination.mkdir(parents=True, exist_ok=True)
    replace(destination / "train", source / "train", options.copy)
    replace(destination / "val", source / "val", options.copy)
    replace(destination / "case2/train", case2_train, options.copy)
    replace(destination / "case2/val", case2_val, options.copy)
    print(
        "Prepared bundled case2 annotations: "
        f"train={len(list(case2_train.rglob('*.json')))} "
        f"val={len(list(case2_val.rglob('*.json')))}"
    )


if __name__ == "__main__":
    main()
