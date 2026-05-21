# Data Scripts

计划放置数据转换、标签检查和数据集统计脚本。

阶段 1 需要补齐：

- `convert_to_yolo_obb.py`
- `analyze_dataset.py`

当前可用：

- `check_labels.py`: 检查 Ultralytics OBB polygon 标签格式、类别范围、坐标范围、图片/标签配对，并生成数据集报告。
- `quarantine_bad_samples.py`: 将 checkpoint 图片、孤儿标签、零面积 OBB 样本移动到隔离目录；默认 dry-run，不删除文件。

示例：

```text
python -B scripts\data\check_labels.py --data datasets\industrial_symbol.yaml --report docs\dataset_report.md
```

先生成隔离清单：

```text
python -B scripts\data\quarantine_bad_samples.py --data datasets\industrial_symbol.yaml
```

确认清单后再移动到隔离目录：

```text
python -B scripts\data\quarantine_bad_samples.py --data datasets\industrial_symbol.yaml --apply
```
