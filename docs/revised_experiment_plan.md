# Edge-symbol-obb 实验计划修正版

## 1. 修正目的

原始 `Edge-symbol-obb_实验计划.md` 给出了项目目标、模块方向和论文实验构想。当前项目已经完成 baseline 阶段，因此本修正版不再停留在初始设想，而是基于已完成实验结果，对后续阶段做可执行修正。

本计划的核心变化：

1. baseline 阶段已完成，不再作为待办项。
2. RV1106 实机速度测试暂缓，因为当前没有端侧设备。
3. 下一阶段优先进入模型改进，先做 QG-OBB Head。
4. 每个模型改动都必须经过 PyTorch、ONNX、RKNN INT8 三段验证。
5. simulator latency 只能作为链路调试参考，不作为最终部署速度。

## 2. 当前项目定位

项目名称：

```text
Edge-symbol-obb
```

建议论文方向：

```text
Deployment-aware Lightweight Oriented Object Detection
for Industrial Symbol Recognition on Resource-constrained Edge NPUs
```

当前任务：

```text
Industrial barcode / QR / 2D symbol oriented object detection
using grayscale single-channel OBB models.
```

当前部署目标：

```text
Rockchip RV1106 NPU
```

当前现实约束：

- 已完成 RKNN INT8 simulator 验证。
- 暂无 RV1106 实机设备。
- 因此端侧 FPS/latency 结论暂不进入最终论文结果。

## 3. 已完成内容

### 3.1 项目工程化整理

已完成：

- 项目脚本按功能归入 `scripts/data/`, `scripts/train/`, `scripts/export/`, `scripts/eval/`, `scripts/deploy_rv1106/`。
- RV1106 模型配置放入 `configs/rv1106/`。
- 固定数据集入口为 `datasets/industrial_symbol.yaml`。
- 关键实验记录归入 `docs/`。

结论：

项目已经具备可持续实验的基本工程结构。

### 3.2 数据集验证

已完成：

- `scripts/data/check_labels.py`
- `scripts/data/quarantine_bad_samples.py`

处理结果：

| Issue | Count |
|---|---:|
| checkpoint image | 6 |
| orphan label | 3 |
| zero-area label | 74 |
| total quarantine actions | 83 |

最终状态：

```text
status=PASS
```

### 3.3 RV1106-M2 PyTorch baseline

训练命令：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --name rv1106_m2_baseline_e100_b256 \
  --epochs 100 \
  --batch 512 \
  --workers 16
```

正式结果：

| Metric | Value |
|---|---:|
| Params | 2.14M |
| FLOPs | 9.0G |
| mAP50 | 0.991 |
| mAP50-95 | 0.960 |
| Training time | 1.154 h |
| Weight size | 4.6 MB |

结论：

当前轻量 baseline 的 FP32 精度已经足够高，可以作为后续改进实验的对照基线。

### 3.4 ONNX FP32 baseline

已完成：

- 参数化导出脚本 `scripts/export/export_gray_obb_onnx.py`
- 参数化验证脚本 `scripts/eval/val_onnx_gray.py`
- 灰度输入适配
- OBB 4-head ONNX 输出 decode 适配

ONNX 结果：

| Metric | Value |
|---|---:|
| ONNX size | 8.2 MB |
| Precision | 0.9846879004808976 |
| Recall | 0.9795905864006227 |
| mAP50 | 0.9904943591129942 |
| mAP50-95 | 0.9596001726240346 |

结论：

ONNX 导出几乎无精度损失，导出链路成立。

### 3.5 RKNN INT8 baseline

已完成：

- 参数化 RKNN 转换和评估脚本 `scripts/deploy_rv1106/convert_eval_rknn.py`
- RKNN Toolkit2 环境验证
- RKNN INT8 转换
- RKNN simulator full validation
- AP 计算从 11-point AP 修正为 continuous precision-envelope AP

RKNN INT8 正式复评结果：

| Metric | Value |
|---|---:|
| Log | `rknn_logs/rknn_eval_20260521_004405.txt` |
| Eval images | 2635 |
| Calibration images | 500 |
| AP method | continuous precision-envelope integration |
| RKNN INT8 AP@0.5 | 0.9846 |
| ONNX FP32 mAP50 | 0.990494 |
| Absolute drop | 0.0059 |

结论：

当前 RV1106-M2 lightweight baseline 的 RKNN INT8 精度验证通过。

### 3.6 暂缓内容

以下内容暂缓：

```text
Physical RV1106 board latency/FPS testing
```

原因：

```text
当前没有端侧 RV1106 设备。
```

记录要求：

- simulator latency 不能写成最终部署速度。
- 后续获得设备后再补测真实 NPU latency/FPS。

## 4. 修正后的总实验路线

修正后的实验路线如下：

```text
Stage 0: 工程整理与数据检查       Done
Stage 1: PyTorch FP32 baseline     Done
Stage 2: ONNX FP32 baseline        Done
Stage 3: RKNN INT8 baseline        Done
Stage 4: QG-OBB Head               Next
Stage 5: SOF-FPN                   Pending
Stage 6: GIS-Aug                   Pending
Stage 7: Full model                Pending
Stage 8: RV1106 real-device test   Deferred
```

## 5. 下一阶段：QG-OBB Head

### 5.1 目标

QG-OBB Head 的目标不是单纯增加精度，而是提升角度预测和 INT8 量化后的稳定性。

核心问题：

- 当前 OBB angle branch 使用单值角度预测。
- 角度存在周期边界。
- 量化后 angle branch 可能成为精度不稳定来源。
- ONNX/RKNN 输出和后处理强依赖 angle 输出格式。

因此 QG-OBB Head 必须从接口审计开始。

### 5.2 第一阶段：接口审计

需要分析的文件：

| File | Purpose |
|---|---|
| `ultralytics/nn/modules/head.py` | 当前 OBB head 和 ONNX 4-head export |
| OBB loss 相关文件 | angle loss 和训练监督 |
| `scripts/eval/val_onnx_gray.py` | ONNX 4-head decode |
| `scripts/deploy_rv1106/convert_eval_rknn.py` | RKNN output decode |
| `configs/rv1106/yolov8n_obb_rv1106_m2.yaml` | 当前 baseline 模型结构 |

产出：

```text
docs/qg_obb_head_design.md
```

该文档必须记录：

- 当前 OBB head 输入输出 shape。
- 当前 angle 输出范围。
- 当前 angle decode 方式。
- 当前 ONNX 4-head 输出格式。
- 当前 RKNN 后处理如何使用 angle。
- 改成 sin-cos 后需要同步修改的所有位置。

### 5.3 第二阶段：最小 sin-cos Head

优先做最小版本，不直接做 Gaussian / Cholesky。

设计目标：

```text
angle scalar -> sin(theta), cos(theta)
```

基本策略：

- box branch 不变。
- cls branch 不变。
- angle branch 输出从 1 channel 改为 2 channel。
- decode 使用 `atan2(sin, cos)`。
- loss 增加 unit-cycle 或 sin-cos 对齐约束。
- ONNX/RKNN 后处理同步修改。

原因：

- sin-cos 是周期角度的低风险改法。
- 便于调试。
- 容易观察是否改善 INT8 稳定性。
- 比 Gaussian/Cholesky 更适合第一轮实验。

### 5.4 第三阶段：QG-OBB Head 完整实验

实验流程：

```text
shape test
-> 1 epoch smoke train
-> 20 epoch quick train
-> 100 epoch full train
-> ONNX export
-> ONNX validation
-> RKNN INT8 conversion
-> RKNN INT8 validation
```

建议实验命名：

| Experiment | Run Name |
|---|---|
| smoke | `rv1106_qg_sincos_smoke` |
| quick | `rv1106_qg_sincos_e20_b512` |
| full | `rv1106_qg_sincos_e100_b512` |

通过标准：

| Metric | Requirement |
|---|---:|
| PyTorch mAP50 | >= baseline - 0.003 |
| PyTorch mAP50-95 | >= baseline - 0.005 |
| ONNX export | PASS |
| ONNX mAP drop | <= 0.005 |
| RKNN INT8 AP@0.5 drop vs ONNX | <= 0.03 |
| Params/FLOPs | 不明显增加 |

若 QG-OBB Head 没有带来明显收益，但没有破坏 RKNN，则可以保留为 ablation 项；若破坏导出或 RKNN 后处理，则回退。

## 6. 后续阶段：SOF-FPN

### 6.1 目标

提升多尺度符号、低质量图像和方向敏感目标的特征表达能力。

### 6.2 实施原则

先做轻量方向保持分支，不直接引入复杂频域算子。

优先插入：

```text
P3 / P4
```

原因：

- P3 对小符号和边缘细节敏感。
- P4 兼顾语义和空间分辨率。
- P5 改动风险更高，优先级较低。

### 6.3 实验组

| Experiment | Purpose |
|---|---|
| baseline neck | 当前结构 |
| + orientation branch P3 | 小目标方向增强 |
| + orientation branch P3/P4 | 多尺度方向增强 |
| + lightweight fusion gate | 控制额外分支权重 |
| full SOF-FPN | 完整 neck 改进 |

建议命名：

| Experiment | Run Name |
|---|---|
| smoke | `rv1106_sof_p3_smoke` |
| quick | `rv1106_sof_p3_e20_b512` |
| full | `rv1106_sof_p3_e100_b512` |

通过标准：

| Metric | Requirement |
|---|---:|
| mAP50-95 | baseline + 0.003 以上才算有效 |
| Params increase | <= 10% |
| FLOPs increase | <= 10% |
| ONNX/RKNN | PASS |

## 7. 后续阶段：GIS-Aug

### 7.1 目标

提升工业符号在复杂成像条件下的鲁棒性。

### 7.2 增强类型

优先做可控增强：

- random rotation
- perspective transform
- motion blur
- low light
- reflection / glare
- partial occlusion
- small object scaling
- large aspect-ratio variation

暂不优先做复杂 diffusion synthetic data，因为那会引入额外变量，不适合作为当前下一步。

### 7.3 实验组

| Group | Augmentation |
|---|---|
| G0 | baseline augmentation |
| G1 | geometry only |
| G2 | image quality only |
| G3 | occlusion and scale |
| G4 | full GIS-Aug |

通过标准：

- 全量 val 不下降。
- `QR`, `DM`, `BARCODE` 三类不能明显下降。
- 如果建立 hard-case subset，则 hard-case 提升优先级高于普通 val 的微小提升。

## 8. Full Model 组合实验

组合顺序：

```text
Baseline
-> Baseline + QG-OBB Head
-> Baseline + SOF-FPN
-> Baseline + QG-OBB Head + SOF-FPN
-> Full Model + GIS-Aug
```

每一步必须记录：

- model yaml
- train command
- PyTorch metrics
- ONNX metrics
- RKNN INT8 metrics
- Params
- FLOPs
- model size
- known issues

## 9. 修正后的论文实验表

### 9.1 主结果表

| Model | Params | FLOPs | PyTorch mAP50 | PyTorch mAP50-95 | ONNX mAP50 | RKNN INT8 AP@0.5 | INT8 Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| RV1106-M2 baseline | 2.14M | 9.0G | 0.991 | 0.960 | 0.990494 | 0.9846 | 0.0059 |
| + QG-OBB Head | | | | | | | |
| + SOF-FPN | | | | | | | |
| Full Model | | | | | | | |

### 9.2 消融实验表

| QG-OBB | SOF-FPN | GIS-Aug | mAP50 | mAP50-95 | RKNN AP@0.5 | INT8 Drop |
|---|---|---|---:|---:|---:|---:|
| No | No | No | 0.991 | 0.960 | 0.9846 | 0.0059 |
| Yes | No | No | | | | |
| No | Yes | No | | | | |
| Yes | Yes | No | | | | |
| Yes | Yes | Yes | | | | |

### 9.3 部署实验表

| Model | Format | Precision | Size | mAP50 / AP@0.5 | Drop | Latency |
|---|---|---|---:|---:|---:|---:|
| Baseline | PyTorch | FP32 | 4.6 MB | 0.991 | 0.000 | |
| Baseline | ONNX | FP32 | 8.2 MB | 0.990494 | 0.0005 | 14.96 ms/image GPU |
| Baseline | RKNN | INT8 | pending | 0.9846 | 0.0059 | simulator only |
| Full Model | PyTorch | FP32 | | | | |
| Full Model | RKNN | INT8 | | | | |

说明：

- RKNN latency 当前只记录 simulator，不作为实机部署结论。
- 真实 RV1106 latency/FPS 等有设备后补测。

## 10. 当前立即执行计划

下一步先做 QG-OBB Head 设计审计。

具体任务：

1. 创建 `docs/qg_obb_head_design.md`。
2. 阅读 `ultralytics/nn/modules/head.py` 中当前 OBB head。
3. 记录当前训练输出、推理输出、ONNX 输出和 RKNN decode 的 shape。
4. 明确 sin-cos angle branch 要修改的文件清单。
5. 给出最小 sin-cos QG-OBB Head 的实现方案。
6. 方案确认后再写代码。

## 11. 修正后的风险清单

| Risk | Impact | Mitigation |
|---|---|---|
| QG-OBB 改动破坏 ONNX 输出 | ONNX/RKNN 无法验证 | 先做设计审计和 shape test |
| sin-cos loss 不收敛 | PyTorch mAP 下降 | 先 1 epoch smoke，再 20 epoch quick |
| RKNN 不支持新增算子 | 部署链路中断 | 推理图只用基础算子 |
| SOF-FPN 增加计算过多 | 不适合 RV1106 | Params/FLOPs 增幅限制 10% |
| GIS-Aug 造成数据分布偏移 | val mAP 下降 | 分组实验，不一次开启全部增强 |
| 无实机速度 | 部署结论不完整 | 明确记录 deferred，后续有设备补测 |

## 12. 修正后阶段判断

baseline 阶段已经完成。当前项目应进入模型改进阶段，第一优先级是 QG-OBB Head 的接口审计和最小 sin-cos 实验。
