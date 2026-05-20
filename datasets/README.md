# Datasets

本目录保存数据集入口文件和数据检查脚本说明。

## Required Layout

推荐工业符号 OBB 数据集使用以下结构：

```text
yolo_dataset_gray/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

标签格式必须是 Ultralytics OBB polygon format：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

坐标应归一化到 `[0, 1]`。

## Current Dataset Entry

- Primary yaml: `datasets/industrial_symbol.yaml`
- Historical yaml: `ultralytics/cfg/datasets/My_project.yaml`

`industrial_symbol.yaml` 当前保留训练服务器路径 `/root/autodl-tmp/yolo_dataset_gray`。如果在 Windows 本机运行，需要先改成实际数据路径。
