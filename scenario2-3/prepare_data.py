#!/usr/bin/env python3
"""Link/copy an existing image dataset and create case2 validation labels."""

import argparse
import json
import os
import shutil
from pathlib import Path


LABELS = {"cracking": "Cracking", "layer_shifting": "Layer_shifting",
          "off_platform": "Off_platform", "stringing": "Stringing", "warping": "Warping"}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Root containing train/, val/, and frontier_llm/gpt/case1_val")
    parser.add_argument("--case2-train", type=Path,
                        default=Path(__file__).resolve().parent / "annotations/case2_train",
                        help="Bundled case2 training annotations")
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
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
    source, destination = options.source.resolve(), options.destination.resolve()
    case2_train = options.case2_train.resolve()
    required = [source / "train", source / "val", case2_train,
                source / "frontier_llm/gpt/case1_val"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing source directories: " + ", ".join(missing))
    destination.mkdir(parents=True, exist_ok=True)
    replace(destination / "train", source / "train", options.copy)
    replace(destination / "val", source / "val", options.copy)
    replace(destination / "case2/train", case2_train, options.copy)

    output = destination / "case2/val"
    output.mkdir(parents=True, exist_ok=True)
    case1 = {path.stem: path for path in (source / "frontier_llm/gpt/case1_val").rglob("*.json")}
    count = 0
    for image in (source / "val").rglob("*"):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            continue
        if image.stem not in case1:
            raise RuntimeError(f"No case1 validation JSON for {image.name}")
        label = LABELS.get(image.parent.name.lower())
        if label is None:
            raise RuntimeError(f"Unknown defect directory: {image.parent.name}")
        with case1[image.stem].open(encoding="utf-8") as handle:
            visible = json.load(handle).get("visible_features", [])
        result = {"dataset_label": label, "supporting_features": visible,
                  "contradicting_features": [], "label_consistency": "consistent"}
        (output / f"{image.stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        count += 1
    print(f"Prepared {count} validation labels in {output}")


if __name__ == "__main__":
    main()

