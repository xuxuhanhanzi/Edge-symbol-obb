"""Analyze YOLO OBB geometry and export hard-case candidates.

This script is read-only for the source dataset. It only writes reports and CSV
files to the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
AREA_BINS = [0.0, 0.002, 0.01, 0.05, 1.0]
ASPECT_BINS = [1.0, 2.0, 5.0, 10.0, 1.0e9]
ANGLE_BINS = [0.0, 15.0, 45.0, 75.0, 105.0, 135.0, 165.0, 180.0]


@dataclass
class ObjectRecord:
    image_path: str
    label_path: str
    line_no: int
    class_id: int
    class_name: str
    area_ratio: float
    aspect_ratio: float
    angle_deg: float
    border_margin: float
    is_priority_class: bool
    is_very_small: bool
    is_small: bool
    is_extreme_aspect: bool
    is_very_extreme_aspect: bool
    is_border: bool
    is_large_rotation: bool
    hard_reasons: list[str]

    @property
    def hard_score(self) -> int:
        return len(self.hard_reasons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/industrial_symbol.yaml", help="Dataset yaml path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to analyze.")
    parser.add_argument("--out-dir", default="artifacts/local/dataset_audit", help="CSV output directory.")
    parser.add_argument("--report", default="docs/dataset_geometry_audit.md", help="Markdown report path.")
    parser.add_argument(
        "--priority-classes",
        default="QR,BARCODE,DM",
        help="Comma-separated class names prioritized for hard-val candidate mining.",
    )
    parser.add_argument("--small-area", type=float, default=0.01, help="Small-object area-ratio threshold.")
    parser.add_argument("--very-small-area", type=float, default=0.002, help="Very-small-object area-ratio threshold.")
    parser.add_argument("--extreme-aspect", type=float, default=5.0, help="Extreme aspect-ratio threshold.")
    parser.add_argument("--very-extreme-aspect", type=float, default=10.0, help="Very-extreme aspect-ratio threshold.")
    parser.add_argument("--border-margin", type=float, default=0.02, help="Border-case margin threshold.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top hard candidates to list in the report.")
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


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def obb_aspect_ratio(points: list[tuple[float, float]]) -> float:
    sides = [distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    long_side = max(sides)
    short_side = min(x for x in sides if x > 0.0) if any(x > 0.0 for x in sides) else 0.0
    if short_side <= 0.0:
        return 0.0
    return long_side / short_side


def obb_angle_deg(points: list[tuple[float, float]]) -> float:
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    return math.degrees(math.atan2(dy, dx)) % 180.0


def border_margin(points: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(min(xs), min(ys), 1.0 - max(xs), 1.0 - max(ys))


def is_large_rotation(angle: float) -> bool:
    return (15.0 <= angle <= 75.0) or (105.0 <= angle <= 165.0)


def bin_label(edges: list[float], value: float) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= value < right or (right == edges[-1] and value <= right):
            right_text = "inf" if right >= 1.0e8 else f"{right:g}"
            return f"[{left:g},{right_text})"
    return f">={edges[-1]:g}"


def parse_label_line(line: str, names: dict[int, str]) -> tuple[int, list[tuple[float, float]]] | None:
    parts = line.split()
    if len(parts) != 9:
        return None
    try:
        cls_float = float(parts[0])
        cls_id = int(cls_float)
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if cls_float != cls_id or cls_id not in names:
        return None
    if any(not math.isfinite(x) for x in coords):
        return None
    if any(x < 0.0 or x > 1.0 for x in coords):
        return None
    points = list(zip(coords[0::2], coords[1::2]))
    if polygon_area(points) <= 1e-12:
        return None
    return cls_id, points


def make_record(
    image_path: Path,
    label_path: Path,
    line_no: int,
    cls_id: int,
    points: list[tuple[float, float]],
    names: dict[int, str],
    priority_classes: set[str],
    args: argparse.Namespace,
) -> ObjectRecord:
    class_name = names[cls_id]
    area = polygon_area(points)
    aspect = obb_aspect_ratio(points)
    angle = obb_angle_deg(points)
    margin = border_margin(points)

    priority = class_name.upper() in priority_classes
    very_small = area < args.very_small_area
    small = area < args.small_area
    extreme_aspect = aspect > args.extreme_aspect
    very_extreme_aspect = aspect > args.very_extreme_aspect
    border = margin < args.border_margin
    large_rot = is_large_rotation(angle)

    reasons = []
    if priority:
        reasons.append("priority_class")
    if very_small:
        reasons.append("very_small")
    elif small:
        reasons.append("small")
    if very_extreme_aspect:
        reasons.append("very_extreme_aspect")
    elif extreme_aspect:
        reasons.append("extreme_aspect")
    if border:
        reasons.append("border")
    if large_rot:
        reasons.append("large_rotation")

    return ObjectRecord(
        image_path=image_path.as_posix(),
        label_path=label_path.as_posix(),
        line_no=line_no,
        class_id=cls_id,
        class_name=class_name,
        area_ratio=area,
        aspect_ratio=aspect,
        angle_deg=angle,
        border_margin=margin,
        is_priority_class=priority,
        is_very_small=very_small,
        is_small=small,
        is_extreme_aspect=extreme_aspect,
        is_very_extreme_aspect=very_extreme_aspect,
        is_border=border,
        is_large_rotation=large_rot,
        hard_reasons=reasons,
    )


def collect_records(data: dict[str, Any], yaml_path: Path, split: str, names: dict[int, str], args: argparse.Namespace) -> list[ObjectRecord]:
    root = Path(str(data.get("path", ".")))
    if not root.is_absolute():
        root = yaml_path.parent / root

    entries = as_list(data.get(split))
    priority_classes = {x.strip().upper() for x in args.priority_classes.split(",") if x.strip()}
    records: list[ObjectRecord] = []

    for entry in entries:
        image_dir = root / entry
        label_dir = infer_label_dir(image_dir)
        label_by_key = {
            p.relative_to(label_dir).with_suffix("").as_posix(): p
            for p in sorted(label_dir.rglob("*.txt")) if p.is_file()
        } if label_dir.exists() else {}

        for image_path in image_files(image_dir):
            key = image_path.relative_to(image_dir).with_suffix("").as_posix()
            label_path = label_by_key.get(key)
            if label_path is None:
                continue
            for line_no, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                parsed = parse_label_line(line, names)
                if parsed is None:
                    continue
                cls_id, points = parsed
                records.append(make_record(image_path, label_path, line_no, cls_id, points, names, priority_classes, args))

    return records


def record_row(record: ObjectRecord) -> dict[str, Any]:
    return {
        "image_path": record.image_path,
        "label_path": record.label_path,
        "line_no": record.line_no,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "area_ratio": f"{record.area_ratio:.8f}",
        "aspect_ratio": f"{record.aspect_ratio:.6f}",
        "angle_deg": f"{record.angle_deg:.4f}",
        "border_margin": f"{record.border_margin:.6f}",
        "is_priority_class": int(record.is_priority_class),
        "is_very_small": int(record.is_very_small),
        "is_small": int(record.is_small),
        "is_extreme_aspect": int(record.is_extreme_aspect),
        "is_very_extreme_aspect": int(record.is_very_extreme_aspect),
        "is_border": int(record.is_border),
        "is_large_rotation": int(record.is_large_rotation),
        "hard_score": record.hard_score,
        "hard_reasons": ";".join(record.hard_reasons),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_hist(path: Path, counts: Counter[str]) -> None:
    rows = [{"bin": key, "count": counts[key]} for key in counts]
    write_csv(path, rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def top_candidates(records: list[ObjectRecord], top_k: int) -> list[ObjectRecord]:
    return sorted(
        records,
        key=lambda r: (
            -r.hard_score,
            not r.is_priority_class,
            r.area_ratio,
            -r.aspect_ratio,
            r.border_margin,
            r.image_path,
        ),
    )[:top_k]


def is_hard_candidate(record: ObjectRecord) -> bool:
    """Return true when a record has at least one real difficulty signal.

    Priority classes are important for ranking, but being QR/BARCODE/DM alone
    should not make an object a hard case.
    """
    return any(reason != "priority_class" for reason in record.hard_reasons)


def write_report(
    path: Path,
    data_path: Path,
    split: str,
    records: list[ObjectRecord],
    hard_records: list[ObjectRecord],
    out_dir: Path,
    top_k: int,
) -> None:
    class_counts = Counter(r.class_name for r in records)
    hard_class_counts = Counter(r.class_name for r in hard_records)
    reason_counts = Counter(reason for r in hard_records for reason in r.hard_reasons)
    area_values = [r.area_ratio for r in records]
    aspect_values = [r.aspect_ratio for r in records]
    angle_values = [r.angle_deg for r in records]

    lines = [
        "# 数据集几何审计与困难样本候选报告",
        "",
        f"- 数据配置: `{data_path.as_posix()}`",
        f"- 分析 split: `{split}`",
        f"- 输出目录: `{out_dir.as_posix()}`",
        f"- 目标总数: {len(records)}",
        f"- 困难候选目标数: {len(hard_records)}",
        "",
        "## 1. 几何摘要",
        "",
        f"- 面积占比均值: {mean(area_values):.6f}",
        f"- 面积占比 p10: {percentile(area_values, 0.10):.6f}",
        f"- 面积占比 p50: {percentile(area_values, 0.50):.6f}",
        f"- 面积占比 p90: {percentile(area_values, 0.90):.6f}",
        f"- 长宽比均值: {mean(aspect_values):.4f}",
        f"- 长宽比 p90: {percentile(aspect_values, 0.90):.4f}",
        f"- 长宽比 p95: {percentile(aspect_values, 0.95):.4f}",
        f"- 角度均值: {mean(angle_values):.4f}",
        "",
        "## 2. 类别分布",
        "",
        "| 类别 | 目标数 | 困难候选数 |",
        "|---|---:|---:|",
    ]
    for class_name in sorted(class_counts):
        lines.append(f"| {class_name} | {class_counts[class_name]} | {hard_class_counts[class_name]} |")

    lines += [
        "",
        "## 3. 困难原因分布",
        "",
        "| 原因 | 数量 |",
        "|---|---:|",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} |")
    if not reason_counts:
        lines.append("| 无 | 0 |")

    lines += [
        "",
        "## 4. Top 困难候选样本",
        "",
        "| hard_score | 类别 | area_ratio | aspect_ratio | angle_deg | border_margin | reasons | image |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for record in top_candidates(hard_records, top_k):
        lines.append(
            f"| {record.hard_score} | {record.class_name} | {record.area_ratio:.6f} | "
            f"{record.aspect_ratio:.3f} | {record.angle_deg:.2f} | {record.border_margin:.4f} | "
            f"{';'.join(record.hard_reasons)} | `{record.image_path}` |"
        )
    if not hard_records:
        lines.append("| 0 | 无 | 0 | 0 | 0 | 0 | 无 | 无 |")

    lines += [
        "",
        "## 5. 生成文件",
        "",
        f"- 全量目标: `{(out_dir / 'all_objects.csv').as_posix()}`",
        f"- 困难候选: `{(out_dir / 'hard_candidates.csv').as_posix()}`",
        f"- 面积直方图: `{(out_dir / 'area_hist.csv').as_posix()}`",
        f"- 长宽比直方图: `{(out_dir / 'aspect_ratio_hist.csv').as_posix()}`",
        f"- 角度直方图: `{(out_dir / 'angle_hist.csv').as_posix()}`",
        "",
        "## 6. 下一步判断",
        "",
        "优先查看 QR、BARCODE、DM 的困难候选数量。如果三类候选足够，再进入 hard-val 复制与 yaml 构建；如果不足，应先搜索外部数据或补充采集。",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    data = load_yaml(data_path)
    names = normalize_names(data.get("names"))
    out_dir = Path(args.out_dir)

    records = collect_records(data, data_path, args.split, names, args)
    hard_records = [r for r in records if is_hard_candidate(r)]

    area_hist = Counter(bin_label(AREA_BINS, r.area_ratio) for r in records)
    aspect_hist = Counter(bin_label(ASPECT_BINS, r.aspect_ratio) for r in records)
    angle_hist = Counter(bin_label(ANGLE_BINS, r.angle_deg) for r in records)

    write_csv(out_dir / "all_objects.csv", [record_row(r) for r in records])
    write_csv(out_dir / "hard_candidates.csv", [record_row(r) for r in top_candidates(hard_records, len(hard_records))])
    write_hist(out_dir / "area_hist.csv", area_hist)
    write_hist(out_dir / "aspect_ratio_hist.csv", aspect_hist)
    write_hist(out_dir / "angle_hist.csv", angle_hist)
    write_report(Path(args.report), data_path, args.split, records, hard_records, out_dir, args.top_k)

    print(f"objects={len(records)}")
    print(f"hard_candidates={len(hard_records)}")
    print(f"out_dir={out_dir}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
