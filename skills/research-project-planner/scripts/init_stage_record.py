#!/usr/bin/env python
"""Create a staged experiment record markdown file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Stage name, e.g. qg_stage2.")
    parser.add_argument("--run", required=True, help="Run name.")
    parser.add_argument("--docs-dir", default="docs/experiments", help="Output directory.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="Date prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.date}_{args.stage}_{args.run}.md"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {path}")

    content = f"""# 实验记录：{args.run}

## 1. 目标

## 2. 环境

- 平台：
- GPU：
- Python：
- PyTorch：
- CUDA：
- Conda 环境：
- 代码路径：
- 代码 commit / diff：
- 数据集路径：

## 3. 实验变量

- 主变量：
- 固定项：
- 对比对象：

## 4. 命令

```bash

```

## 5. 输出路径

- 权重：
- 日志：
- TensorBoard：
- ONNX：
- 部署模型：
- 可视化：

## 6. 结果

| 指标 | 数值 |
|---|---:|

## 7. 失败与异常

## 8. 结论

## 9. 下一步
"""
    path.write_text(content, encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
