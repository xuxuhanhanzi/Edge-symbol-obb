# 阶段二：困难样本候选分析记录

## 1. 阶段目标

阶段一 `check_labels.py` 已经通过基础完整性检查，输出状态为 `PASS`。阶段二的目标是基于现有验证集，先找出可解释、可复核的困难样本候选，而不是立即修改数据集。

本阶段只读取原始图片和标签，输出 CSV 与 Markdown 报告，不移动、不删除、不复制源数据。

## 2. 输入

数据配置：

```text
datasets/industrial_symbol.yaml
```

默认分析 split：

```text
val
```

优先关注类别：

```text
QR, BARCODE, DM
```

## 3. 困难候选规则

一个目标进入困难候选，至少需要命中一个真实困难信号：

- 小目标或极小目标。
- 极端长宽比。
- 接近图片边界。
- 大角度旋转。

`QR/BARCODE/DM` 是优先类别，会提升候选排序优先级，但不会单独构成困难样本。

## 4. AutoDL 执行命令

```bash
conda activate base
cd ~/ultralytics_yolov8-main/ultralytics_yolov8-main

python -B scripts/data/analyze_obb_hard_cases.py \
  --data datasets/industrial_symbol.yaml \
  --split val \
  --out-dir artifacts/local/dataset_audit \
  --report docs/dataset_geometry_audit.md
```

## 5. 预期输出

```text
docs/dataset_geometry_audit.md
artifacts/local/dataset_audit/all_objects.csv
artifacts/local/dataset_audit/hard_candidates.csv
artifacts/local/dataset_audit/area_hist.csv
artifacts/local/dataset_audit/aspect_ratio_hist.csv
artifacts/local/dataset_audit/angle_hist.csv
```

## 6. 执行后检查命令

```bash
wc -l artifacts/local/dataset_audit/hard_candidates.csv
head -n 20 artifacts/local/dataset_audit/hard_candidates.csv
cat docs/dataset_geometry_audit.md
```

## 7. 决策门槛

如果 QR、BARCODE、DM 三类在困难候选中数量足够，并且样本来源没有明显泄漏风险，则进入阶段三：构建固定 `industrial_symbol_hard.yaml`。

如果候选数量不足，或者候选集中大量样本仍然过于简单，则阶段三改为外部数据集搜索与补充采集，暂缓 hard-val 固化。
