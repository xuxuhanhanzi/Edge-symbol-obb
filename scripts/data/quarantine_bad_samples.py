"""Quarantine invalid OBB dataset samples without deleting files.

This script handles the common dataset issues reported by check_labels.py:

- checkpoint images under `.ipynb_checkpoints`
- label files without matching images
- OBB label files containing zero-area polygons

Default mode is dry-run. Use --apply to move files into a quarantine directory.
No files are deleted.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Action:
    reason: str
    split: str
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/industrial_symbol.yaml", help="Dataset yaml path.")
    parser.add_argument(
        "--quarantine-dir",
        default=None,
        help="Quarantine directory. Defaults to <dataset-root>/quarantine_bad_samples.",
    )
    parser.add_argument("--manifest", default="docs/dataset_quarantine_manifest.csv", help="CSV action manifest.")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Default is dry-run.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset yaml must be a mapping: {path}")
    return data


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels"


def image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_files(label_dir: Path) -> list[Path]:
    if not label_dir.exists():
        return []
    return sorted(p for p in label_dir.rglob("*.txt") if p.is_file())


def polygon_area(coords: list[float]) -> float:
    points = list(zip(coords[0::2], coords[1::2]))
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def label_has_zero_area_polygon(label_path: Path) -> bool:
    for raw in label_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            continue
        try:
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if polygon_area(coords) <= 1e-12:
            return True
    return False


def unique_target(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def quarantine_target(quarantine_dir: Path, reason: str, split: str, source: Path, dataset_root: Path) -> Path:
    try:
        rel = source.relative_to(dataset_root)
    except ValueError:
        rel = Path(source.name)
    return unique_target(quarantine_dir / reason / split / rel)


def add_move(actions: list[Action], reason: str, split: str, source: Path, quarantine_dir: Path, dataset_root: Path) -> None:
    if not source.exists() or not source.is_file():
        return
    target = quarantine_target(quarantine_dir, reason, split, source, dataset_root)
    actions.append(Action(reason=reason, split=split, source=source, target=target))


def collect_actions(dataset_yaml: Path, quarantine_dir: Path) -> list[Action]:
    data = load_yaml(dataset_yaml)
    dataset_root = Path(str(data.get("path", "."))).expanduser()
    actions: list[Action] = []

    for split in ("train", "val", "test"):
        for entry in as_list(data.get(split)):
            image_dir = dataset_root / entry
            label_dir = infer_label_dir(image_dir)
            images = image_files(image_dir)
            labels = label_files(label_dir)

            label_by_key = {p.relative_to(label_dir).with_suffix("").as_posix(): p for p in labels}
            image_by_key = {p.relative_to(image_dir).with_suffix("").as_posix(): p for p in images}

            for image_path in images:
                if ".ipynb_checkpoints" in image_path.parts:
                    add_move(actions, "checkpoint_image", split, image_path, quarantine_dir, dataset_root)

            for key, label_path in label_by_key.items():
                if key not in image_by_key:
                    add_move(actions, "orphan_label", split, label_path, quarantine_dir, dataset_root)
                    continue

                if label_has_zero_area_polygon(label_path):
                    image_path = image_by_key[key]
                    add_move(actions, "zero_area_label", split, label_path, quarantine_dir, dataset_root)
                    add_move(actions, "zero_area_label", split, image_path, quarantine_dir, dataset_root)

    # Deduplicate repeated val/test actions when test points to val.
    deduped: dict[Path, Action] = {}
    for action in actions:
        deduped.setdefault(action.source, action)
    return list(deduped.values())


def write_manifest(actions: list[Action], manifest_path: Path, applied: bool) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["applied", "reason", "split", "source", "target"])
        for action in actions:
            writer.writerow([applied, action.reason, action.split, action.source, action.target])


def apply_actions(actions: list[Action]) -> None:
    for action in actions:
        action.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(action.source), str(action.target))


def main() -> int:
    args = parse_args()
    dataset_yaml = Path(args.data)
    data = load_yaml(dataset_yaml)
    dataset_root = Path(str(data.get("path", "."))).expanduser()
    quarantine_dir = Path(args.quarantine_dir).expanduser() if args.quarantine_dir else dataset_root / "quarantine_bad_samples"

    actions = collect_actions(dataset_yaml, quarantine_dir)
    write_manifest(actions, Path(args.manifest), applied=args.apply)

    if args.apply:
        apply_actions(actions)

    counts: dict[str, int] = {}
    for action in actions:
        counts[action.reason] = counts.get(action.reason, 0) + 1

    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"actions={len(actions)}")
    for reason, count in sorted(counts.items()):
        print(f"{reason}={count}")
    print(f"manifest={args.manifest}")
    print(f"quarantine_dir={quarantine_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
