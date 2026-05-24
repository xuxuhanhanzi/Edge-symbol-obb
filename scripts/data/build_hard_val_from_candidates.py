"""Build a fixed hard validation dataset from OBB hard-case candidates.

The script copies selected images and labels into a new dataset directory,
then writes a dataset yaml, a manifest CSV, and a markdown report. It never
deletes source data or output data.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


STRONG_REASONS = {"very_small", "small", "border", "very_extreme_aspect", "extreme_aspect"}
REASON_WEIGHTS = {
    "priority_class": 1,
    "very_small": 7,
    "small": 4,
    "border": 4,
    "very_extreme_aspect": 6,
    "extreme_aspect": 4,
    "large_rotation": 1,
}
AUG_WEIGHTS = {
    "Aug_Occlusion": 3,
    "Aug_Quality": 3,
    "Aug_Lighting": 2,
    "Aug_Geometric": 1,
}


@dataclass
class CandidateRow:
    image_path: Path
    label_path: Path
    class_id: int
    class_name: str
    area_ratio: float
    aspect_ratio: float
    angle_deg: float
    border_margin: float
    hard_score: int
    hard_reasons: tuple[str, ...]
    source_split: str
    source_csv: str


@dataclass
class CandidateGroup:
    image_path: Path
    label_path: Path
    source_split: str
    rows: list[CandidateRow]

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(sorted({reason for row in self.rows for reason in row.hard_reasons}))

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(sorted({row.class_name for row in self.rows}))

    @property
    def primary_row(self) -> CandidateRow:
        return max(self.rows, key=lambda row: row_score(row))

    @property
    def primary_class(self) -> str:
        return self.primary_row.class_name

    @property
    def score(self) -> int:
        return max(row_score(row) for row in self.rows) + len(self.rows) - 1

    @property
    def has_strong_reason(self) -> bool:
        return any(reason in STRONG_REASONS for reason in self.reasons)

    @property
    def is_rotation_only(self) -> bool:
        return "large_rotation" in self.reasons and not self.has_strong_reason

    @property
    def is_priority(self) -> bool:
        return "priority_class" in self.reasons


@dataclass
class SelectedSample:
    group: CandidateGroup
    dest_image_path: Path
    dest_label_path: Path
    dest_rel_image: str
    dest_rel_label: str
    leakage_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True, help="One or more hard_candidates.csv files.")
    parser.add_argument("--data", default="datasets/industrial_symbol.yaml", help="Source dataset yaml for class names.")
    parser.add_argument("--out-root", required=True, help="Output dataset root, e.g. /root/autodl-tmp/yolo_dataset_gray_hard.")
    parser.add_argument("--out-yaml", default="datasets/industrial_symbol_hard.yaml", help="Output dataset yaml path.")
    parser.add_argument("--manifest", default="docs/hard_val_manifest.csv", help="Selected sample manifest CSV path.")
    parser.add_argument("--report", default="docs/hard_val_build_report.md", help="Markdown report path.")
    parser.add_argument("--split-name", default="val", help="Output split name. Default: val.")
    parser.add_argument("--max-total", type=int, default=700, help="Maximum selected images.")
    parser.add_argument(
        "--priority-classes",
        default="QR,BARCODE,DM",
        help="Comma-separated priority classes. Used for reporting and default quotas.",
    )
    parser.add_argument(
        "--class-quota",
        default="BARCODE=220,DM=130,QR=120",
        help="Comma-separated per-class maximums, e.g. BARCODE=220,DM=130,QR=120.",
    )
    parser.add_argument("--max-per-other-class", type=int, default=35, help="Maximum images for each non-priority class.")
    parser.add_argument(
        "--max-rotation-only-ratio",
        type=float,
        default=0.35,
        help="Maximum fraction of selected images that may be rotation-only hard cases.",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Allow candidates from train split. Use only for diagnostic sets unless retraining excludes these samples.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing over explicit destination files. Extra old files are never removed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select and report samples without copying files.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset yaml must be a mapping: {path}")
    return data


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("Dataset yaml must provide names as a dict or list.")


def parse_class_quota(text: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    if not text.strip():
        return quotas
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class quota item: {item}")
        name, value = item.split("=", 1)
        quotas[name.strip().upper()] = int(value)
    return quotas


def parse_reasons(value: str) -> tuple[str, ...]:
    return tuple(reason for reason in value.split(";") if reason)


def infer_source_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for marker in ("images", "labels"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return parts[idx - 1]
    for split in ("train", "val", "test"):
        if split in parts:
            return split
    return "unknown"


def row_score(row: CandidateRow) -> int:
    score = sum(REASON_WEIGHTS.get(reason, 0) for reason in row.hard_reasons)
    image_name = row.image_path.name
    for token, weight in AUG_WEIGHTS.items():
        if token in image_name:
            score += weight
    return max(score, row.hard_score)


def read_candidates(paths: list[Path]) -> tuple[list[CandidateGroup], Counter[str]]:
    grouped: dict[str, CandidateGroup] = {}
    stats: Counter[str] = Counter()

    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                stats["rows"] += 1
                try:
                    image_path = Path(raw["image_path"])
                    label_path = Path(raw["label_path"])
                    row = CandidateRow(
                        image_path=image_path,
                        label_path=label_path,
                        class_id=int(raw["class_id"]),
                        class_name=raw["class_name"],
                        area_ratio=float(raw["area_ratio"]),
                        aspect_ratio=float(raw["aspect_ratio"]),
                        angle_deg=float(raw["angle_deg"]),
                        border_margin=float(raw["border_margin"]),
                        hard_score=int(raw["hard_score"]),
                        hard_reasons=parse_reasons(raw["hard_reasons"]),
                        source_split=infer_source_split(image_path),
                        source_csv=path.as_posix(),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    stats["bad_rows"] += 1
                    stats[f"bad_row:{type(exc).__name__}"] += 1
                    continue

                key = image_path.as_posix()
                if key not in grouped:
                    grouped[key] = CandidateGroup(
                        image_path=image_path,
                        label_path=label_path,
                        source_split=row.source_split,
                        rows=[],
                    )
                grouped[key].rows.append(row)

    return list(grouped.values()), stats


def allowed_source_splits(include_train: bool) -> set[str]:
    splits = {"val", "test"}
    if include_train:
        splits.add("train")
    return splits


def filter_groups(groups: list[CandidateGroup], include_train: bool) -> tuple[list[CandidateGroup], Counter[str]]:
    allowed = allowed_source_splits(include_train)
    kept: list[CandidateGroup] = []
    stats: Counter[str] = Counter()

    for group in groups:
        if group.source_split not in allowed:
            stats[f"skipped_split:{group.source_split}"] += 1
            continue
        if not group.image_path.exists():
            stats["missing_image"] += 1
            continue
        if not group.label_path.exists():
            stats["missing_label"] += 1
            continue
        kept.append(group)
    return kept, stats


def sort_groups(groups: list[CandidateGroup]) -> list[CandidateGroup]:
    return sorted(
        groups,
        key=lambda group: (
            -int(group.has_strong_reason),
            -group.score,
            -int(group.is_priority),
            group.primary_class,
            group.image_path.as_posix(),
        ),
    )


def class_limit(class_name: str, quotas: dict[str, int], max_per_other_class: int) -> int:
    return quotas.get(class_name.upper(), max_per_other_class)


def select_groups(
    groups: list[CandidateGroup],
    max_total: int,
    quotas: dict[str, int],
    max_per_other_class: int,
    max_rotation_only_ratio: float,
) -> tuple[list[CandidateGroup], Counter[str]]:
    sorted_items = sort_groups(groups)
    selected: list[CandidateGroup] = []
    selected_paths: set[str] = set()
    class_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    rotation_only_limit = max(0, int(max_total * max_rotation_only_ratio))

    def try_add(group: CandidateGroup) -> bool:
        if len(selected) >= max_total:
            stats["skipped_max_total"] += 1
            return False
        key = group.image_path.as_posix()
        if key in selected_paths:
            stats["skipped_duplicate"] += 1
            return False
        primary = group.primary_class
        if class_counts[primary] >= class_limit(primary, quotas, max_per_other_class):
            stats[f"skipped_quota:{primary}"] += 1
            return False
        if group.is_rotation_only and stats["selected_rotation_only"] >= rotation_only_limit:
            stats["skipped_rotation_only_cap"] += 1
            return False
        selected.append(group)
        selected_paths.add(key)
        class_counts[primary] += 1
        stats["selected"] += 1
        stats[f"selected_class:{primary}"] += 1
        stats[f"selected_split:{group.source_split}"] += 1
        if group.has_strong_reason:
            stats["selected_strong"] += 1
        if group.is_rotation_only:
            stats["selected_rotation_only"] += 1
        return True

    passes = [
        lambda group: group.has_strong_reason and group.is_priority,
        lambda group: group.has_strong_reason and not group.is_priority,
        lambda group: group.is_rotation_only and group.is_priority,
        lambda group: group.is_rotation_only and not group.is_priority,
    ]

    for predicate in passes:
        for group in sorted_items:
            if predicate(group):
                try_add(group)

    return selected, stats


def safe_stem(value: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or "sample")[:max_len]


def destination_paths(group: CandidateGroup, out_root: Path, split_name: str, index: int) -> tuple[Path, Path, str, str]:
    class_name = safe_stem(group.primary_class)
    source_split = safe_stem(group.source_split)
    stem = safe_stem(group.image_path.stem)
    dest_stem = f"{index:06d}_{source_split}_{class_name}_{stem}"
    rel_image = f"{split_name}/images/{dest_stem}{group.image_path.suffix.lower()}"
    rel_label = f"{split_name}/labels/{dest_stem}.txt"
    return out_root / rel_image, out_root / rel_label, rel_image, rel_label


def ensure_can_write(paths: list[Path], exist_ok: bool) -> None:
    if exist_ok:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        preview = "\n".join(str(path) for path in existing[:10])
        raise FileExistsError(
            "Destination files already exist. Use a fresh --out-root or pass --exist-ok.\n"
            f"Existing examples:\n{preview}"
        )


def copy_selected(
    selected_groups: list[CandidateGroup],
    out_root: Path,
    split_name: str,
    exist_ok: bool,
    dry_run: bool,
) -> list[SelectedSample]:
    samples: list[SelectedSample] = []
    planned_paths: list[Path] = []

    for idx, group in enumerate(selected_groups, start=1):
        dest_image, dest_label, rel_image, rel_label = destination_paths(group, out_root, split_name, idx)
        planned_paths.extend([dest_image, dest_label])
        leakage_status = "train_source_diagnostic" if group.source_split == "train" else "formal_eval_candidate"
        samples.append(
            SelectedSample(
                group=group,
                dest_image_path=dest_image,
                dest_label_path=dest_label,
                dest_rel_image=rel_image,
                dest_rel_label=rel_label,
                leakage_status=leakage_status,
            )
        )

    if not dry_run:
        ensure_can_write(planned_paths, exist_ok)
    if dry_run:
        return samples

    for sample in samples:
        sample.dest_image_path.parent.mkdir(parents=True, exist_ok=True)
        sample.dest_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.group.image_path, sample.dest_image_path)
        shutil.copy2(sample.group.label_path, sample.dest_label_path)

    return samples


def write_yaml(path: Path, out_root: Path, split_name: str, names: dict[int, str], dry_run: bool) -> None:
    if dry_run:
        return
    image_entry = f"{split_name}/images"
    lines = [
        "# Hard validation dataset generated from OBB hard-case candidates.",
        "# Use for evaluation only; do not train on this dataset.",
        f"path: {out_root.as_posix()}",
        "train:",
        f"  - {image_entry}",
        "val:",
        f"  - {image_entry}",
        "test:",
        f"  - {image_entry}",
        "",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_row(sample: SelectedSample) -> dict[str, Any]:
    group = sample.group
    return {
        "dest_image_path": sample.dest_image_path.as_posix(),
        "dest_label_path": sample.dest_label_path.as_posix(),
        "source_image_path": group.image_path.as_posix(),
        "source_label_path": group.label_path.as_posix(),
        "source_split": group.source_split,
        "leakage_status": sample.leakage_status,
        "primary_class": group.primary_class,
        "class_names": ";".join(group.class_names),
        "score": group.score,
        "has_strong_reason": int(group.has_strong_reason),
        "is_rotation_only": int(group.is_rotation_only),
        "reasons": ";".join(group.reasons),
        "object_candidates_in_image": len(group.rows),
    }


def write_manifest(path: Path, samples: list[SelectedSample], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [manifest_row(sample) for sample in samples]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    names: dict[int, str],
    read_stats: Counter[str],
    filter_stats: Counter[str],
    select_stats: Counter[str],
    samples: list[SelectedSample],
) -> None:
    source_counts = Counter(sample.group.source_split for sample in samples)
    class_counts = Counter(sample.group.primary_class for sample in samples)
    reason_counts = Counter(reason for sample in samples for reason in sample.group.reasons)
    leakage_counts = Counter(sample.leakage_status for sample in samples)
    train_warning = any(sample.group.source_split == "train" for sample in samples)

    lines = [
        "# Hard-Val 构建报告",
        "",
        f"- 输出根目录: `{Path(args.out_root).as_posix()}`",
        f"- 输出 YAML: `{Path(args.out_yaml).as_posix()}`",
        f"- manifest: `{Path(args.manifest).as_posix()}`",
        f"- dry-run: `{args.dry_run}`",
        f"- 选择图片数: {len(samples)}",
        f"- 类别数: {len(names)}",
        f"- include_train: `{args.include_train}`",
        "",
        "## 1. 数据泄漏状态",
        "",
    ]
    if train_warning:
        lines.append(
            "本次输出包含 `train` 来源样本。除非后续训练明确排除了这些样本，否则该数据集只能作为诊断集，不能作为正式泛化验证集。"
        )
    else:
        lines.append("本次输出未包含 `train` 来源样本，可作为当前训练配置下的 hard-val 候选。")

    lines += [
        "",
        "| 状态 | 数量 |",
        "|---|---:|",
    ]
    for key, count in leakage_counts.most_common():
        lines.append(f"| {key} | {count} |")
    if not leakage_counts:
        lines.append("| 无 | 0 |")

    lines += [
        "",
        "## 2. 来源 split",
        "",
        "| split | 数量 |",
        "|---|---:|",
    ]
    for split, count in source_counts.most_common():
        lines.append(f"| {split} | {count} |")
    if not source_counts:
        lines.append("| 无 | 0 |")

    lines += [
        "",
        "## 3. 类别分布",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
    ]
    for class_name, count in class_counts.most_common():
        lines.append(f"| {class_name} | {count} |")
    if not class_counts:
        lines.append("| 无 | 0 |")

    lines += [
        "",
        "## 4. 困难原因分布",
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
        "## 5. 过程计数",
        "",
        "| 阶段 | 项 | 数量 |",
        "|---|---|---:|",
    ]
    for key, count in read_stats.items():
        lines.append(f"| read | {key} | {count} |")
    for key, count in filter_stats.items():
        lines.append(f"| filter | {key} | {count} |")
    for key, count in select_stats.items():
        lines.append(f"| select | {key} | {count} |")

    lines += [
        "",
        "## 6. 下一步",
        "",
        "先运行 `check_labels.py` 检查生成的数据集，再分别用 scalar、QG、official baseline 跑 PyTorch/ONNX/RKNN 评估。正式论文结论只使用无 train 泄漏的 hard-val 结果。",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    data = load_yaml(data_path)
    names = normalize_names(data.get("names"))
    candidate_paths = [Path(path) for path in args.candidates]
    quotas = parse_class_quota(args.class_quota)

    groups, read_stats = read_candidates(candidate_paths)
    groups, filter_stats = filter_groups(groups, args.include_train)
    selected_groups, select_stats = select_groups(
        groups=groups,
        max_total=args.max_total,
        quotas=quotas,
        max_per_other_class=args.max_per_other_class,
        max_rotation_only_ratio=args.max_rotation_only_ratio,
    )
    samples = copy_selected(
        selected_groups=selected_groups,
        out_root=Path(args.out_root),
        split_name=args.split_name,
        exist_ok=args.exist_ok,
        dry_run=args.dry_run,
    )

    write_yaml(Path(args.out_yaml), Path(args.out_root), args.split_name, names, args.dry_run)
    write_manifest(Path(args.manifest), samples, args.dry_run)
    write_report(Path(args.report), args, names, read_stats, filter_stats, select_stats, samples)

    print(f"candidate_rows={read_stats['rows']}")
    print(f"candidate_images={len(groups)}")
    print(f"selected_images={len(samples)}")
    print(f"out_root={args.out_root}")
    print(f"out_yaml={args.out_yaml}")
    print(f"manifest={args.manifest}")
    print(f"report={args.report}")
    if args.include_train:
        print("warning=train_source_included_diagnostic_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
