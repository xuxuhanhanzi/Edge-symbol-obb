"""Check an Ultralytics OBB dataset and write a reproducible markdown report."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class SplitStats:
    name: str
    image_dirs: list[Path] = field(default_factory=list)
    label_dirs: list[Path] = field(default_factory=list)
    images: int = 0
    label_files: int = 0
    objects: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    bad_lines: int = 0
    blank_lines: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    aspect_ratios: list[float] = field(default_factory=list)
    angles_deg: list[float] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/industrial_symbol.yaml", help="Dataset yaml path.")
    parser.add_argument("--report", default="docs/dataset_report.md", help="Markdown report output path.")
    parser.add_argument("--max-examples", type=int, default=30, help="Maximum issue examples to keep.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if errors are found.")
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


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("Dataset yaml must provide names as a dict or list.")


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


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def add_example(stats: SplitStats, message: str, max_examples: int) -> None:
    if len(stats.examples) < max_examples:
        stats.examples.append(message)


def check_label_line(
    line: str,
    label_path: Path,
    line_no: int,
    names: dict[int, str],
    stats: SplitStats,
    max_examples: int,
) -> None:
    parts = line.split()
    if len(parts) != 9:
        stats.bad_lines += 1
        add_example(stats, f"{label_path}:{line_no} expected 9 values, got {len(parts)}", max_examples)
        return

    try:
        cls_float = float(parts[0])
        cls_id = int(cls_float)
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        stats.bad_lines += 1
        add_example(stats, f"{label_path}:{line_no} contains non-numeric values", max_examples)
        return

    if cls_float != cls_id or cls_id not in names:
        stats.bad_lines += 1
        add_example(stats, f"{label_path}:{line_no} invalid class id {parts[0]}", max_examples)
        return

    if any(not math.isfinite(x) for x in coords):
        stats.bad_lines += 1
        add_example(stats, f"{label_path}:{line_no} contains non-finite coordinates", max_examples)
        return

    if any(x < 0.0 or x > 1.0 for x in coords):
        stats.bad_lines += 1
        add_example(stats, f"{label_path}:{line_no} coordinates outside [0, 1]", max_examples)
        return

    points = list(zip(coords[0::2], coords[1::2]))
    area = polygon_area(points)
    if area <= 1e-12:
        stats.bad_lines += 1
        add_example(stats, f"{label_path}:{line_no} polygon area is zero", max_examples)
        return

    xs = coords[0::2]
    ys = coords[1::2]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width > 0 and height > 0:
        stats.aspect_ratios.append(max(width / height, height / width))

    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    stats.angles_deg.append(angle)
    stats.objects += 1
    stats.class_counts[cls_id] += 1


def check_split(root: Path, split_name: str, entries: list[str], names: dict[int, str], max_examples: int) -> SplitStats:
    stats = SplitStats(split_name)

    for entry in entries:
        image_dir = root / entry
        label_dir = infer_label_dir(image_dir)
        stats.image_dirs.append(image_dir)
        stats.label_dirs.append(label_dir)

        if not image_dir.exists():
            add_example(stats, f"missing image dir: {image_dir}", max_examples)
            continue
        if not label_dir.exists():
            add_example(stats, f"missing label dir: {label_dir}", max_examples)

        images = image_files(image_dir)
        labels = label_files(label_dir)
        stats.images += len(images)
        stats.label_files += len(labels)

        label_rel = {p.relative_to(label_dir).with_suffix("").as_posix(): p for p in labels}
        image_rel = {p.relative_to(image_dir).with_suffix("").as_posix(): p for p in images}

        for key, img_path in image_rel.items():
            label_path = label_rel.get(key)
            if label_path is None:
                stats.missing_labels += 1
                add_example(stats, f"missing label for image: {img_path}", max_examples)
                continue

            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            non_empty = 0
            for line_no, raw in enumerate(lines, start=1):
                line = raw.strip()
                if not line:
                    stats.blank_lines += 1
                    continue
                non_empty += 1
                check_label_line(line, label_path, line_no, names, stats, max_examples)
            if non_empty == 0:
                stats.empty_labels += 1

        for key, label_path in label_rel.items():
            if key not in image_rel:
                stats.orphan_labels += 1
                add_example(stats, f"orphan label without image: {label_path}", max_examples)

    return stats


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def split_has_errors(stats: SplitStats) -> bool:
    return any(
        [
            stats.missing_labels,
            stats.orphan_labels,
            stats.bad_lines,
            any(not p.exists() for p in stats.image_dirs),
            any(not p.exists() for p in stats.label_dirs),
        ]
    )


def write_report(path: Path, yaml_path: Path, root: Path, names: dict[int, str], split_stats: list[SplitStats]) -> None:
    total_images = sum(s.images for s in split_stats)
    total_labels = sum(s.label_files for s in split_stats)
    total_objects = sum(s.objects for s in split_stats)
    total_bad = sum(s.bad_lines for s in split_stats)
    total_missing = sum(s.missing_labels for s in split_stats)
    total_orphan = sum(s.orphan_labels for s in split_stats)
    status = "PASS" if not any(split_has_errors(s) for s in split_stats) else "FAIL"

    lines = [
        "# Dataset Report",
        "",
        f"- Dataset yaml: `{yaml_path.as_posix()}`",
        f"- Root path: `{root}`",
        f"- Status: **{status}**",
        f"- Classes: {len(names)}",
        f"- Images: {total_images}",
        f"- Label files: {total_labels}",
        f"- Objects: {total_objects}",
        f"- Missing labels: {total_missing}",
        f"- Orphan labels: {total_orphan}",
        f"- Bad label lines: {total_bad}",
        "",
        "## Splits",
        "",
        "| Split | Images | Label Files | Objects | Empty Labels | Missing Labels | Orphan Labels | Bad Lines |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stats in split_stats:
        lines.append(
            f"| {stats.name} | {stats.images} | {stats.label_files} | {stats.objects} | "
            f"{stats.empty_labels} | {stats.missing_labels} | {stats.orphan_labels} | {stats.bad_lines} |"
        )

    total_counts: Counter[int] = Counter()
    for stats in split_stats:
        total_counts.update(stats.class_counts)

    lines += [
        "",
        "## Class Distribution",
        "",
        "| Class ID | Name | Objects |",
        "|---:|---|---:|",
    ]
    for cls_id, name in names.items():
        lines.append(f"| {cls_id} | {name} | {total_counts.get(cls_id, 0)} |")

    aspect_values = [x for stats in split_stats for x in stats.aspect_ratios]
    angle_values = [x for stats in split_stats for x in stats.angles_deg]
    lines += [
        "",
        "## Geometry Summary",
        "",
        f"- Aspect ratio mean: {mean(aspect_values):.4f}",
        f"- Aspect ratio p95: {percentile(aspect_values, 0.95):.4f}",
        f"- Angle mean degree: {mean(angle_values):.4f}",
        f"- Angle p95 degree: {percentile(angle_values, 0.95):.4f}",
        "",
        "## Issue Examples",
        "",
    ]

    issue_count = 0
    for stats in split_stats:
        for example in stats.examples:
            issue_count += 1
            lines.append(f"- `{stats.name}`: {example}")
    if issue_count == 0:
        lines.append("- No issue examples recorded.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    yaml_path = Path(args.data)
    data = load_yaml(yaml_path)
    names = normalize_names(data.get("names"))
    root = Path(str(data.get("path", ".")))

    split_stats = []
    for split_name in ("train", "val", "test"):
        entries = as_list(data.get(split_name))
        if entries:
            split_stats.append(check_split(root, split_name, entries, names, args.max_examples))

    report_path = Path(args.report)
    write_report(report_path, yaml_path, root, names, split_stats)

    has_errors = any(split_has_errors(s) for s in split_stats)
    print(f"dataset_report={report_path}")
    print(f"status={'FAIL' if has_errors else 'PASS'}")
    return 1 if args.strict and has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
