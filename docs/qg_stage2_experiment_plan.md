# QG-OBB 阶段 2 实验计划

本文档记录阶段 A/B 的已完成结果，并给出下一阶段正式实验的固定方案。

## 1. 阶段 A/B 结论

当前已完成三组 20 epoch 对照：

| 方案 | ONNX mAP50 | ONNX mAP50-95 | RKNN INT8 mAP50 | ONNX 到 RKNN 掉点 | RKNN 平均耗时 |
|---|---:|---:|---:|---:|---:|
| Scalar RV1106-M2 | 0.9561 | 0.8832 | 0.9400 | 0.0161 | 44.5 ms |
| QG 原始 0.25/0.05 | 0.9548 | 0.8859 | 0.9427 | 0.0121 | 45.6 ms |
| QG decode-only 0/0 | 0.9535 | 0.8864 | 0.9405 | 0.0130 | 43.4 ms |

阶段 A/B 的正式候选选择为：`QG 原始 0.25/0.05`。

选择理由：

- RKNN INT8 mAP50 最高，为 `0.9427`。
- ONNX 到 RKNN 的掉点最小，为 `0.0121`。
- 符合 QG-OBB 当前阶段的主要目标：改善角度表示连续性和 INT8 部署稳定性，而不是只追求 FP32 mAP。

`decode-only` 保留为备选消融项；weak 版本淘汰。

## 2. 固定训练超参数

QG 相关损失现在由命令行参数控制：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--angle` | 5.0 | OBB 角度损失整体权重 |
| `--qg-angle-align` | 0.25 | QG unit-cycle 对齐损失权重 |
| `--qg-angle-unit` | 0.05 | QG 向量模长约束损失权重 |

对应关系：

| 实验名 | `--qg-angle-align` | `--qg-angle-unit` |
|---|---:|---:|
| QG 原始 | 0.25 | 0.05 |
| QG decode-only | 0.0 | 0.0 |
| QG weak | 0.10 | 0.02 |

## 3. 正式 100 Epoch 命令

下一阶段优先只跑主候选：

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

## 4. 部署验证命令

训练完成后导出 ONNX：

```bash
RUN=rv1106_qg_sincos_e100_b512_selected

python -B scripts/export/export_gray_obb_onnx.py \
  --weights runs/obb/$RUN/weights/best.pt \
  --imgsz 256 \
  --opset 19 \
  --device 0
```

验证 ONNX：

```bash
python -B scripts/eval/val_onnx_gray.py \
  --weights runs/obb/$RUN/weights/best.onnx \
  --data datasets/industrial_symbol.yaml \
  --imgsz 256 \
  --batch 1
```

验证 RKNN INT8：

```bash
conda activate rknn232

python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --onnx runs/obb/$RUN/weights/best.onnx \
  --rknn runs/obb/$RUN/weights/best_int8_rv1106_v232.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --dataset-txt artifacts/local/dataset_rknn_qg_e100_v232.txt \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_qg_e100_v232_full \
  --target-platform rv1106 \
  --imgsz 256 \
  --quant-images 500 \
  --no-save-inference
```

## 5. RKNN 环境要求

后续 QG 实验统一使用：

| 组件 | 版本 |
|---|---|
| `rknn-toolkit2` | 2.3.2 |
| `onnx` | 1.16.1 |
| `numpy` | 1.26.4 |
| `opencv-python` | 4.11.0.86 |
| `setuptools` | `<81` |

旧 `rknn-toolkit2 1.6.0` 在 QG 模型 simulator 初始化阶段会卡住，不再作为 QG 部署评估环境。

## 6. 下一步判定标准

正式 100 epoch 后优先观察：

- PyTorch / ONNX mAP50 是否接近或超过 e20 结果。
- ONNX mAP50-95 是否保持稳定。
- RKNN INT8 mAP50 是否继续优于 scalar RV1106-M2。
- ONNX 到 RKNN 掉点是否继续小于 scalar 对照。
- 推理耗时是否保持在 RV1106-M2 同级范围内。

如果 `QG 原始 0.25/0.05` 在 e100 上 RKNN mAP50 和掉点仍优于 scalar 对照，则作为 QG-OBB 第一阶段正式结果。

## 7. 正式 100 Epoch 结果

`rv1106_qg_sincos_e100_b512_selected` 与 `rv1106_m2_e100_b512` 已完成同环境对比。RKNN 均使用 `rknn-toolkit2==2.3.2`。

| 模型 | PyTorch mAP50 | PyTorch mAP50-95 | ONNX mAP50 | ONNX mAP50-95 | RKNN INT8 mAP50 | ONNX 到 RKNN 掉点 | RKNN 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RV1106-M2 scalar e100 | 0.991 | 0.960 | 0.9905 | 0.9596 | 0.9843 | 0.0062 | 45.200 ms |
| QG-OBB e100 | 0.990 | 0.957 | 0.9895 | 0.9577 | 0.9854 | 0.0041 | 44.574 ms |

结论：

- QG-OBB e100 并未显著提升 PyTorch / ONNX 精度。
- QG-OBB e100 在 RKNN INT8 评估中略高于 scalar baseline。
- QG-OBB e100 的 ONNX 到 RKNN 掉点更小，`0.0041` vs `0.0062`。
- 当前证据支持将 QG-OBB 定位为 **量化友好的角度表示优化**。

阶段记录见：

```text
docs/experiments/20260524_qg_e100_rknn_comparison.md
```

## 8. 下一阶段任务

下一阶段进入官方 YOLOv8-OBB baseline 审计和对比准备。

优先级：

1. 审计当前仓库中的官方 YOLOv8-OBB 配置是否仍可作为官方 baseline。
2. 若当前官方配置已被本地修改污染，则创建独立灰度官方 baseline 配置。
3. 先跑 official YOLOv8n-OBB gray e20 quick，对齐训练和评估链路。
4. quick 通过后再跑 official YOLOv8n-OBB gray e100。
5. 后续再设计 Official YOLOv8n-OBB + QG Head 的主对比实验。
