# Edge-symbol-obb 项目阶段完成报告

## 1. 报告目的

本文档记录截至当前阶段，Edge-symbol-obb 项目已经完成的代码工作、实验工作、实验数据、实验结论和遗留事项。当前阶段的核心目标是建立一个可复现、可导出、可量化、可部署验证的 RV1106 轻量化 OBB baseline。

## 2. 当前阶段结论

项目已经完成从数据检查到模型训练，再到 ONNX 导出、ONNX 验证、RKNN INT8 转换和 RKNN simulator 精度验证的完整闭环。

当前正式 baseline 为：

```text
Model: RV1106-M2 lightweight OBB baseline
Model YAML: configs/rv1106/yolov8n_obb_rv1106_m2.yaml
Data YAML: datasets/industrial_symbol.yaml
Input: grayscale single-channel, 256 x 256
Classes: 15
```

核心结论：

- PyTorch FP32 baseline 精度成立。
- ONNX FP32 导出没有造成明显精度损失。
- RKNN INT8 转换成功，且连续 AP 复评后 INT8 精度掉点很小。
- RV1106 实机速度测试暂缓，因为当前没有端侧设备；现有 RKNN 速度仅为 simulator 结果，不能作为论文最终部署速度。

## 3. 已完成工作

### 3.1 项目结构整理

已将原本散落在根目录的大量脚本整理到功能目录中：

| 目录 | 用途 |
|---|---|
| `scripts/data/` | 数据检查、坏样本隔离 |
| `scripts/train/` | 训练和 smoke train |
| `scripts/export/` | ONNX 导出 |
| `scripts/eval/` | ONNX 验证 |
| `scripts/deploy_rv1106/` | RKNN 转换和部署评估 |
| `configs/rv1106/` | RV1106 相关模型配置 |
| `datasets/` | 数据集入口 YAML |
| `docs/` | 实验记录和报告 |

项目结构整理的主要收益：

- 训练、导出、验证和部署脚本职责清晰。
- 后续实验可以按阶段复用统一入口。
- 避免历史脚本继续写死旧路径。

### 3.2 数据集检查与清理

数据集入口：

```text
datasets/industrial_symbol.yaml
```

已完成数据标签检查脚本：

```text
scripts/data/check_labels.py
```

已完成坏样本隔离脚本：

```text
scripts/data/quarantine_bad_samples.py
```

服务器上首次检查结果为 FAIL，随后使用隔离脚本处理异常样本：

| 问题类型 | 数量 |
|---|---:|
| checkpoint image | 6 |
| orphan label | 3 |
| zero-area label | 74 |
| total actions | 83 |

处理后重新检查：

```text
status=PASS
```

数据集规模：

| Split | Images | Instances |
|---|---:|---:|
| train | 50049 | - |
| val | 2635 | 2693 |

### 3.3 训练入口

已完成稳妥训练入口：

```text
scripts/train/train_rv1106_smoke.py
```

该脚本支持命令行参数，能够进行 1 epoch smoke train，也能进行正式 baseline 训练。

正式 baseline 训练命令：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --name rv1106_m2_baseline_e100_b256 \
  --epochs 100 \
  --batch 512 \
  --workers 16
```

注意：run name 中包含 `b256`，但实际训练 batch 是 `512`。

### 3.4 PyTorch FP32 baseline

训练环境：

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CUDA | torch `2.7.0+cu128` |
| Ultralytics | `8.2.82` |
| Epochs | 100 |
| Batch | 512 |
| Image size | 256 |
| Optimizer | SGD |
| AMP | False |
| Workers | 16 |

模型规模：

| Metric | Value |
|---|---:|
| Parameters | 2.14M |
| FLOPs | 9.0G |
| PyTorch weight size | 4.6 MB |

PyTorch FP32 结果：

| Metric | Value |
|---|---:|
| Precision | 0.986 |
| Recall | 0.975 |
| mAP50 | 0.991 |
| mAP50-95 | 0.960 |
| Training time | 1.154 h |

权重路径：

```text
runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt
```

### 3.5 ONNX 导出与验证

已完成参数化导出脚本：

```text
scripts/export/export_gray_obb_onnx.py
```

导出命令：

```bash
python -B scripts/export/export_gray_obb_onnx.py \
  --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt \
  --imgsz 256 \
  --opset 19
```

ONNX 输出：

```text
runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx
```

ONNX 文件大小：

```text
8.2 MB
```

已完成参数化 ONNX 验证脚本：

```text
scripts/eval/val_onnx_gray.py
```

该脚本已经修复两个关键问题：

- 灰度单通道模型的输入适配。
- OBB 4-head ONNX 输出在进入 Ultralytics NMS 前的解码适配。

ONNX 验证命令：

```bash
python -B scripts/eval/val_onnx_gray.py \
  --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --data datasets/industrial_symbol.yaml \
  --imgsz 256
```

ONNX FP32 验证结果：

| Metric | Value |
|---|---:|
| Precision | 0.9846879004808976 |
| Recall | 0.9795905864006227 |
| mAP50 | 0.9904943591129942 |
| mAP50-95 | 0.9596001726240346 |
| Fitness | 0.9626895912729306 |

速度统计，AutoDL RTX 5090：

| Stage | Time |
|---|---:|
| preprocess | 0.49 ms/image |
| inference | 14.96 ms/image |
| postprocess | 6.40 ms/image |

ONNX 结论：

- ONNX mAP50-95 为 `0.959600`，PyTorch mAP50-95 为 `0.960`。
- ONNX 导出没有造成明显精度损失。

### 3.6 RKNN INT8 转换与验证

已完成 RKNN 主入口参数化：

```text
scripts/deploy_rv1106/convert_eval_rknn.py
```

该脚本支持：

- ONNX 路径参数化。
- RKNN 输出路径参数化。
- 数据路径参数化。
- calibration 数量参数化。
- debug images 参数。
- simulator build + eval。
- real hardware `--eval-only --runtime-target`。
- 连续 precision-envelope AP 计算。

RKNN 环境：

```bash
conda activate rknn_env
python -c "from rknn.api import RKNN; print('rknn_ok')"
```

结果：

```text
rknn_ok
RKNN Toolkit2 version: 1.6.0+81f21f4d
```

RKNN INT8 转换和验证命令：

```bash
export OMP_NUM_THREADS=1

python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --onnx runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --rknn runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --dataset-txt artifacts/local/dataset_rknn.txt \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_inference_results \
  --target-platform rv1106 \
  --imgsz 256 \
  --quant-images 500
```

RKNN 输出：

```text
runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn
```

RKNN 连续 AP 复评结果：

| Item | Value |
|---|---:|
| Log | `rknn_logs/rknn_eval_20260521_004405.txt` |
| Eval images | 2635 |
| Calibration images | 500 |
| Runtime mode | RKNN simulator |
| AP method | continuous precision-envelope integration |
| mAP@0.5 | 0.9846 |
| Avg inference time | 76.679 ms/image simulator |
| Min inference time | 39.273 ms/image simulator |
| Max inference time | 34098.166 ms/image simulator |

逐类 AP@0.5：

| Class | AP@0.5 |
|---|---:|
| BARCODE | 0.9782 |
| DM | 0.9605 |
| HANXIN | 0.9907 |
| QR | 0.9375 |
| PDF | 0.9856 |
| AZTEC | 0.9939 |
| CODEONE | 0.9950 |
| DOT | 0.9776 |
| GM | 0.9950 |
| MAXI | 0.9950 |
| MPDF | 0.9905 |
| MQR | 0.9950 |
| RMQR | 0.9946 |
| ULTRA | 0.9854 |
| UPN | 0.9950 |

RKNN 精度分析：

| Compare | Value |
|---|---:|
| ONNX FP32 mAP50 | 0.990494 |
| RKNN INT8 AP@0.5 | 0.9846 |
| Absolute drop | 0.0059 |

结论：

- RKNN INT8 精度掉点约 `0.0059`。
- 掉点远小于 `0.03` 的初始可接受阈值。
- 当前 RV1106-M2 lightweight baseline 的 RKNN INT8 精度验证通过。

注意：

- 当前速度来自 RKNN simulator，不是 RV1106 实机速度。
- 因无端侧设备，RV1106 实机 latency/FPS 测试暂缓。
- simulator latency 不得作为论文最终部署速度。

## 4. 实验数据分析

### 4.1 PyTorch 到 ONNX

PyTorch mAP50-95 为 `0.960`，ONNX mAP50-95 为 `0.959600`，差异可以忽略。说明当前灰度 OBB 模型导出路径是可靠的。

关键修复点是 ONNX 4-head 输出。默认 Ultralytics validator 不能直接处理当前 ONNX 输出，需要在 NMS 前完成 4-head decode。该问题已经在 `val_onnx_gray.py` 中解决。

### 4.2 ONNX 到 RKNN INT8

RKNN INT8 连续 AP 结果为 `0.9846`，相比 ONNX FP32 mAP50 `0.990494` 仅下降约 `0.0059`。这说明当前模型结构对 INT8 量化较稳定。

早期 RKNN 结果 `0.9134` 不应作为正式结果，因为当时使用的是 11-point AP，评估口径偏低。修正为连续 precision-envelope AP 后，结果回到 `0.9846`。

### 4.3 速度数据

当前可记录但不能作为最终部署结论的数据：

```text
RKNN simulator avg inference: 76.679 ms/image
```

该值只能说明 simulator 环境下链路可运行，不能代表 RV1106 NPU 的真实吞吐。

## 5. 当前遗留事项

| Item | Status | Reason |
|---|---|---|
| RV1106 实机 latency/FPS | Deferred | 暂无端侧设备 |
| RKNN model size | Pending | 需要服务器输出 `.rknn` 文件大小 |
| QG-OBB Head | Pending | 下一阶段模型改进主任务 |
| SOF-FPN | Pending | QG-OBB 后进行 |
| GIS-Aug | Pending | 结构实验后进行 |

## 6. 下一阶段方向

下一阶段进入模型改进阶段，优先级如下：

1. QG-OBB Head：先做角度分支接口审计和最小 sin-cos 版本。
2. SOF-FPN：在 QG 成立后做轻量方向保持 neck。
3. GIS-Aug：在结构稳定后做可控数据增强。
4. Full model：组合 QG-OBB Head、SOF-FPN 和 GIS-Aug。

每个新模型必须重复以下链路：

```text
PyTorch train
-> PyTorch val
-> ONNX export
-> ONNX val
-> RKNN INT8 conversion
-> RKNN INT8 val
```

## 7. 当前阶段判断

当前 baseline 阶段可以视为完成。后续模型改进可以在此基础上推进。
