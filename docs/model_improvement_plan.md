# Model Improvement Plan

本文档记录 baseline 完成后的模型改进阶段计划。当前 RV1106 实机速度测试暂缓，因为暂时没有端侧设备；后续模型改进先以 AutoDL 上可闭环验证的 PyTorch、ONNX、RKNN simulator 结果为准。

## 0. Current Fixed Baseline

当前可作为后续所有实验对照的正式 baseline：

| Stage | Artifact | Metric |
|---|---|---:|
| PyTorch FP32 | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt` | mAP50 `0.991`, mAP50-95 `0.960` |
| ONNX FP32 | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx` | mAP50 `0.990494`, mAP50-95 `0.959600` |
| RKNN INT8 | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn` | continuous AP@0.5 `0.9846` |

暂缓项：

- Physical RV1106 board latency/FPS testing is deferred because no endpoint device is currently available.
- Simulator latency must not be used as final deployment latency.

## 1. Improvement Principles

模型改进必须遵守以下原则：

1. 每次只改一个主要因素，避免无法判断收益来源。
2. 每个候选结构必须先通过 smoke train，再进入完整训练。
3. 每个有效候选都必须完成 PyTorch -> ONNX -> RKNN INT8 三段验证。
4. 参数量和 FLOPs 不能明显膨胀；目标是保持 RV1106 友好的轻量结构。
5. RKNN INT8 掉点优先级高于单纯 PyTorch mAP 提升。若 PyTorch 提升但 INT8 掉点过大，该方案不能作为最终模型。
6. 没有实机设备前，只记录 simulator latency，不把它写成最终部署速度。

## 2. Stage A: QG-OBB Head

目标：提升角度预测稳定性，降低 INT8 量化后角度分支的敏感性。

### A1. Code Audit

需要先阅读并记录：

- `ultralytics/nn/modules/head.py`
- OBB head 当前 forward/export 分支
- 当前 angle 输出格式
- OBB loss 中 angle loss 的计算方式
- ONNX 4-head 输出和 RKNN 后处理脚本之间的格式约定

产出：

- `docs/qg_obb_head_design.md`
- 明确当前 head 输出形状、angle decode 公式、ONNX/RKNN 后处理依赖。

### A2. Minimal Sin-Cos Angle Branch

先做最小可验证版本，不直接上 Gaussian/Cholesky。

候选设计：

- 将 angle 标量预测扩展为 `sin(theta), cos(theta)` 或等价 unit-cycle 表达。
- loss 中加入单位圆约束或归一化 decode。
- 推理时从 `atan2(sin, cos)` 解码角度。
- 保持 OBB box 分支和 cls 分支不变。

预期收益：

- 减少角度周期边界带来的不连续问题。
- 提升量化后 angle branch 稳定性。

风险：

- 需要同步修改训练、推理、ONNX export 和 RKNN decode。
- 若只改 head 而 loss 没有配套，可能收敛变差。

验证顺序：

1. 单元级 shape 检查。
2. 1 epoch smoke train。
3. 20 epoch 快速训练，观察 loss 和 mAP 是否正常。
4. 100 epoch 正式训练。
5. ONNX export + ONNX validation。
6. RKNN INT8 conversion + full validation。

成功标准：

| Metric | Requirement |
|---|---:|
| PyTorch mAP50 | >= baseline - 0.003 |
| PyTorch mAP50-95 | >= baseline - 0.005 |
| RKNN INT8 AP@0.5 drop vs ONNX | <= 0.03 |
| Params/FLOPs | 不明显增加 |
| ONNX/RKNN export | 必须通过 |

### A3. QG-OBB Advanced Head

在 A2 成立后再考虑 Gaussian/Cholesky-style 表达。

候选方向：

- 使用 Gaussian covariance 或 Cholesky 参数描述旋转框不确定性。
- 在训练期增加几何一致性约束。
- 推理期仍输出 RV1106 友好的简单张量，避免复杂算子进入 ONNX/RKNN。

约束：

- 不允许引入 RKNN 不支持的复杂算子到导出图。
- 复杂几何计算尽量保留在 loss 或训练辅助分支中，推理图保持简单。

## 3. Stage B: SOF-FPN

目标：提升符号检测的多尺度、方向保持和低质量图像鲁棒性，同时保持轻量。

### B1. Neck Baseline Profiling

先分析当前 neck：

- P3/P4/P5 三层输入输出通道
- 当前 PAN/FPN concat 位置
- 每层特征图尺寸
- FLOPs 主要集中位置

产出：

- `docs/sof_fpn_design.md`
- 当前 neck 的结构图和潜在插入点。

### B2. Orientation-Preserving Branch

优先做轻量方向保持分支，而不是复杂 spectral branch。

候选设计：

- 在 P3 或 P4 上加入轻量方向增强模块。
- 使用 depthwise conv 或小核方向卷积。
- 输出通过轻量 gate 融合回原特征。

推荐先只插入 P3/P4：

- P3 对小符号和细节最敏感。
- P5 尺寸小但语义强，改动风险较大。

验证：

1. 新 YAML：`configs/rv1106/yolov8n_obb_rv1106_sof_p3.yaml`
2. smoke train。
3. 20 epoch quick train。
4. 若提升明显，再 100 epoch full train。

成功标准：

| Metric | Requirement |
|---|---:|
| PyTorch mAP50-95 | baseline + 0.003 以上才认为有意义 |
| Params | 增加不超过 10% |
| FLOPs | 增加不超过 10% |
| ONNX/RKNN | 必须通过 |

### B3. Spectral Enhancement Branch

仅在 B2 通过后尝试。

原则：

- 不使用 FFT 等部署风险高的算子进入推理图。
- 若使用频域思想，优先实现为训练期增强或可部署的卷积近似。

## 4. Stage C: GIS-Aug

目标：提升工业符号在真实复杂成像条件下的鲁棒性。

先不生成复杂 synthetic data，先做可控增强。

候选增强：

- random rotation
- perspective transform
- motion blur
- low light
- reflection/glare
- partial occlusion
- small object scaling
- large aspect-ratio variation

实施策略：

1. 不直接改默认训练 pipeline。
2. 新增可开关训练脚本参数或配置。
3. 每个增强组独立实验。

实验组：

| Group | Augmentation |
|---|---|
| G0 | baseline augmentation |
| G1 | geometry only: rotation + perspective |
| G2 | image quality only: blur + low light + reflection |
| G3 | occlusion/scale only |
| G4 | full GIS-Aug |

成功标准：

- 全量 val mAP 不下降。
- 重点类别 `QR`, `DM`, `BARCODE` 不下降。
- 若可构造 hard-case subset，则 hard-case subset 提升优先级高于普通 val 微小提升。

## 5. Stage D: Combined Model

组合顺序：

1. Baseline
2. Baseline + QG-OBB Head
3. Baseline + SOF-FPN
4. Baseline + QG-OBB Head + SOF-FPN
5. Full model + GIS-Aug

每一步都必须记录：

- model yaml
- train command
- PyTorch result
- ONNX result
- RKNN INT8 result
- Params/FLOPs
- model size
- known issues

## 6. Experiment Naming

建议命名：

| Experiment | Run Name |
|---|---|
| QG minimal sin-cos smoke | `rv1106_qg_sincos_smoke` |
| QG minimal sin-cos quick | `rv1106_qg_sincos_e20_b512` |
| QG minimal sin-cos full | `rv1106_qg_sincos_e100_b512` |
| SOF P3 smoke | `rv1106_sof_p3_smoke` |
| SOF P3 quick | `rv1106_sof_p3_e20_b512` |
| SOF P3 full | `rv1106_sof_p3_e100_b512` |
| Full model | `rv1106_qg_sof_e100_b512` |
| Full model + GIS-Aug | `rv1106_qg_sof_gisaug_e100_b512` |

## 7. Immediate Next Step

最先执行的代码任务：

1. 创建 `docs/qg_obb_head_design.md`。
2. 阅读并记录当前 OBB head、loss、export、RKNN decode 的张量约定。
3. 做 QG-OBB Head 的最小 sin-cos 设计，不直接改训练主流程。
4. 设计通过后，再新建独立 head/module 和 YAML，避免污染 baseline。

当前不建议立刻做 SOF-FPN 或 GIS-Aug，因为 QG-OBB Head 影响 angle branch、ONNX 输出和 RKNN 后处理，是最需要先厘清接口的部分。
