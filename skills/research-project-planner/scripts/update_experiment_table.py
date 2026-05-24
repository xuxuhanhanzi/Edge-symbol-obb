#!/usr/bin/env python
"""Append one row to a markdown experiment table."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_HEADER = (
    "| Run | Main variable | PyTorch mAP50 | ONNX mAP50 | Deployment mAP50 | Drop | Decision |\n"
    "|---|---|---:|---:|---:|---:|---|\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, help="Markdown table file.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--variable", default="")
    parser.add_argument("--pytorch-map50", default="")
    parser.add_argument("--onnx-map50", default="")
    parser.add_argument("--deployment-map50", default="")
    parser.add_argument("--drop", default="")
    parser.add_argument("--decision", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.table)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_HEADER, encoding="utf-8")

    row = (
        f"| {args.run} | {args.variable} | {args.pytorch_map50} | {args.onnx_map50} | "
        f"{args.deployment_map50} | {args.drop} | {args.decision} |\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(row)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
