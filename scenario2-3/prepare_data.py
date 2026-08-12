#!/usr/bin/env python3
"""Install bundled Scenario 2-3 annotations beside case1/case1_val."""

import argparse
import shutil
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Dataset root containing train/, val/, frontier_llm/gpt/")
    parser.add_argument("--case2-train", type=Path,
                        default=Path(__file__).resolve().parent / "annotations/case2_train")
    parser.add_argument("--case2-val", type=Path,
                        default=Path(__file__).resolve().parent / "annotations/case2_val")
    return parser.parse_args()


def install(source: Path, target: Path, expected: int):
    files = list(source.rglob("*.json"))
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} JSON files in {source}, found {len(files)}")
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        existing = list(target.rglob("*.json")) if target.is_dir() else []
        if len(existing) == expected:
            print(f"Already prepared: {target} ({expected} JSON)")
            return
        raise FileExistsError(
            f"{target} already exists with {len(existing)} JSON files; "
            "move or remove that incomplete directory first"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=True)


def main():
    args = arguments()
    root = args.source.resolve()
    required = [root / "train", root / "val", root / "frontier_llm/gpt"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))

    gpt = root / "frontier_llm/gpt"
    install(args.case2_train.resolve(), gpt / "case2", 1840)
    install(args.case2_val.resolve(), gpt / "case2_val", 276)
    print(f"Prepared {gpt / 'case2'} (1840 JSON)")
    print(f"Prepared {gpt / 'case2_val'} (276 JSON)")


if __name__ == "__main__":
    main()
