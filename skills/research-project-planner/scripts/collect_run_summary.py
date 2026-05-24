#!/usr/bin/env python
"""Collect a lightweight markdown summary from common experiment logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRIC_PATTERNS = {
    "mAP50": re.compile(r"mAP@?0?\.?5\s*[:=]\s*([0-9.]+)|mAP50\s+([0-9.]+)", re.I),
    "mAP50-95": re.compile(r"mAP50-95\s*[:=]?\s*([0-9.]+)|mAP@?0?\.?5:0\.95\s*[:=]\s*([0-9.]+)", re.I),
    "avg_latency": re.compile(r"Avg inference time\s*[:=]\s*([0-9.]+)", re.I),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Training run directory, e.g. runs/obb/name.")
    parser.add_argument("--log", action="append", default=[], help="Additional text log to scan.")
    return parser.parse_args()


def read_last_results_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        return {}
    row = rows[-1]
    return {k.strip(): v.strip() for k, v in row.items() if v}


def scan_text(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    found: dict[str, str] = {}
    for name, pattern in METRIC_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            groups = [g for g in matches[-1].groups() if g is not None]
            if groups:
                found[name] = groups[0]
    return found


def main() -> int:
    args = parse_args()
    summary: dict[str, str] = {}
    if args.run_dir:
        run_dir = Path(args.run_dir)
        summary.update(read_last_results_csv(run_dir / "results.csv"))
    for log in args.log:
        summary.update(scan_text(Path(log)))

    if not summary:
        print("No metrics found.")
        return 1

    print("| Metric | Value |")
    print("|---|---:|")
    for key, value in summary.items():
        print(f"| {key} | {value} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
