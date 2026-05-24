# QG-OBB e100 与 RV1106-M2 e100 RKNN 对比记录

## 1. 实验目标

验证 QG-OBB 是否在保持接近 scalar angle head 检测精度的同时，降低 ONNX 到 RKNN INT8 的精度掉点。

本阶段不把 QG-OBB 表述为单纯精度提升模块，而重点验证其作为量化友好角度表示优化的价值。

## 2. 实验环境

| 项 | 内容 |
|---|---|
| 平台 | AutoDL |
| GPU | NVIDIA GeForce RTX 5090 32111MiB |
| 训练环境 | `base` |
| RKNN 环境 | `rknn232` |
| 训练 Python / Torch | Python 3.12.3 / torch 2.7.0+cu128 |
| RKNN toolkit | `rknn-toolkit2==2.3.2` |
| RKNN ONNX | `onnx==1.16.1` |
| RKNN numpy | `numpy==1.26.4` |
| 输入尺寸 | `256` |
| 数据集 | `/root/autodl-tmp/yolo_dataset_gray` |
| 项目路径 | `/root/ultralytics_yolov8-main/ultralytics_yolov8-main` |

## 3. 对比模型

| 模型 | Run | 说明 |
|---|---|---|
| RV1106-M2 scalar e100 | `rv1106_m2_e100_b512` | 工程部署 baseline，scalar angle head |
| QG-OBB e100 | `rv1106_qg_sincos_e100_b512_selected` | RV1106-M2 + QG sin-cos unit-cycle angle branch |

QG-OBB e100 训练参数：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --data datasets/industrial_symbol.yaml \
  --name rv1106_qg_sincos_e100_b512_selected \
  --epochs 100 \
  --batch 512 \
  --workers 16 \
  --device 0 \
  --imgsz 256 \
  --angle 5.0 \
  --qg-angle-align 0.25 \
  --qg-angle-unit 0.05
```

## 4. ONNX 输出结构

### RV1106-M2 scalar e100

```text
loc_head0 [1, 48, 32, 32]
loc_head1 [1, 48, 16, 16]
loc_head2 [1, 48, 8, 8]
angle_head [1, 1, 1344]
```

### QG-OBB e100

```text
loc_head0 [1, 48, 32, 32]
loc_head1 [1, 48, 16, 16]
loc_head2 [1, 48, 8, 8]
angle_sin_head [1, 1, 1344]
angle_cos_head [1, 1, 1344]
```

QG-OBB 使用 5-head split-angle ONNX 输出，RKNN 图中不包含 `atan2`，角度在 CPU 后处理中解码。

## 5. 实验结果总表

| 模型 | PyTorch P | PyTorch R | PyTorch mAP50 | PyTorch mAP50-95 | ONNX P | ONNX R | ONNX mAP50 | ONNX mAP50-95 | RKNN mAP50 | ONNX->RKNN drop | RKNN avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RV1106-M2 scalar e100 | 0.986 | 0.976 | 0.991 | 0.960 | 0.9847 | 0.9796 | 0.9905 | 0.9596 | 0.9843 | 0.0062 | 45.200 ms |
| QG-OBB e100 | 0.982 | 0.975 | 0.990 | 0.957 | 0.9857 | 0.9735 | 0.9895 | 0.9577 | 0.9854 | 0.0041 | 44.574 ms |

drop 计算：

```text
Scalar: 0.9904943591 - 0.9843 = 0.0062
QG:     0.9895153820 - 0.9854 = 0.0041
```

## 6. RKNN per-class AP@0.5 对比

| 类别 | Scalar RKNN AP50 | QG RKNN AP50 | 差值 QG-Scalar |
|---|---:|---:|---:|
| BARCODE | 0.9776 | 0.9764 | -0.0012 |
| DM | 0.9605 | 0.9531 | -0.0074 |
| HANXIN | 0.9907 | 0.9950 | +0.0043 |
| QR | 0.9324 | 0.9414 | +0.0090 |
| PDF | 0.9856 | 0.9938 | +0.0082 |
| AZTEC | 0.9939 | 0.9938 | -0.0001 |
| CODEONE | 0.9950 | 0.9950 | 0.0000 |
| DOT | 0.9776 | 0.9863 | +0.0087 |
| GM | 0.9950 | 0.9906 | -0.0044 |
| MAXI | 0.9950 | 0.9950 | 0.0000 |
| MPDF | 0.9905 | 0.9946 | +0.0041 |
| MQR | 0.9950 | 0.9949 | -0.0001 |
| RMQR | 0.9946 | 0.9906 | -0.0040 |
| ULTRA | 0.9854 | 0.9858 | +0.0004 |
| UPN | 0.9950 | 0.9950 | 0.0000 |

QG 改善较明显的类别包括 `QR`、`PDF`、`DOT`、`HANXIN`、`MPDF`。下降较明显的类别包括 `DM`、`GM`、`RMQR`。

## 7. 结论

1. RV1106-M2 scalar e100 在 PyTorch 和 ONNX 上略高于 QG-OBB e100。
2. QG-OBB e100 在 RKNN INT8 上高于 scalar e100，`0.9854` vs `0.9843`。
3. QG-OBB 将 ONNX 到 RKNN 的 mAP50 掉点从 `0.0062` 降低到 `0.0041`。
4. 当前证据支持将 QG-OBB 定位为量化友好的 unit-cycle angle branch，而不是单纯精度提升模块。

可用于论文/报告的保守表述：

```text
QG-OBB 在保持与 scalar angle head 接近的 FP32/ONNX 检测精度时，降低了 ONNX 到 RKNN INT8 的精度掉点，并在 RKNN INT8 评估中取得略高的 mAP50，说明双角 unit-cycle 表示对部署量化稳定性有积极作用。
```

不建议表述为：

```text
QG-OBB 显著提升检测精度。
```

## 8. 下一步

1. 将本记录同步到 `docs/qg_stage2_experiment_plan.md` 和 `docs/qg_obb_head_design.md`。
2. 准备官方 YOLOv8-OBB baseline 审计。
3. 建立官方 YOLOv8n-OBB 与 QG-OBB 的论文主对比链路。
