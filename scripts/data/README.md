# Data Scripts

计划放置数据转换、标签检查和数据集统计脚本。

阶段 1 需要补齐：

- `convert_to_yolo_obb.py`
- `analyze_dataset.py`

当前可用：

- `check_labels.py`: 检查 Ultralytics OBB polygon 标签格式、类别范围、坐标范围、图片/标签配对，并生成数据集报告。
- `analyze_obb_hard_cases.py`: 统计 OBB 面积、长宽比、角度、贴边情况，并导出困难样本候选 CSV 和中文审计报告。该脚本只读源数据，不移动或删除图片/标签。
- `build_hard_val_from_candidates.py`: 从一个或多个困难候选 CSV 中按配额选择图片，复制到新的 hard-val 数据集目录，并生成 YAML、manifest 和中文报告。默认排除 train 来源；如显式使用 `--include-train`，结果只能作为诊断集，不能直接作为正式泛化验证集。
- `quarantine_bad_samples.py`: 将 checkpoint 图片、孤儿标签、零面积 OBB 样本移动到隔离目录；默认 dry-run，不删除文件。

示例：

```text
python -B scripts\data\check_labels.py --data datasets\industrial_symbol.yaml --report docs\dataset_report.md
```

阶段二困难样本候选分析：

```text
python -B scripts\data\analyze_obb_hard_cases.py --data datasets\industrial_symbol.yaml --split val --out-dir artifacts\local\dataset_audit --report docs\dataset_geometry_audit.md
```

阶段三 hard-val 构建：

```text
python -B scripts/data/build_hard_val_from_candidates.py --candidates artifacts/local/dataset_audit/hard_candidates.csv --data datasets/industrial_symbol.yaml --out-root /root/autodl-tmp/yolo_dataset_gray_hard --out-yaml datasets/industrial_symbol_hard.yaml --manifest docs/hard_val_manifest.csv --report docs/hard_val_build_report.md
```

先生成隔离清单：

```text
python -B scripts\data\quarantine_bad_samples.py --data datasets\industrial_symbol.yaml
```

确认清单后再移动到隔离目录：

```text
python -B scripts\data\quarantine_bad_samples.py --data datasets\industrial_symbol.yaml --apply
```
