# QG-OBB Head 设计审计文档

## 1. 文档目的

本文档基于当前项目文档和 `C:\Users\27475\Desktop\paper` 中的论文规划材料，重新整理 QG-OBB Head 的设计依据、论文出处、与官方 YOLOv8-OBB 的对比关系，以及下一步可实现的最小方案。

本次修订后的核心原则：

1. 对比对象应为 **官方标准 YOLOv8-OBB**，不是历史魔改模型。
2. 当前 RV1106-M2 轻量模型只作为工程实现起点和部署 baseline，不作为论文中新方案的唯一对比对象。
3. QG-OBB 的主论文依据改为 `paper` 目录中明确规划的 GauCho、RSAR、InlierQ。
4. 旧文档中引用的 Biternion、Complex-YOLO、CSL、ProbIoU 只保留为背景知识，不再作为本项目主创新出处。

## 2. 当前参考材料来源

新增纳入项目参考的外部资料目录：

```text
C:\Users\27475\Desktop\paper
```

已阅读并用于本次修订的关键文件：

| 文件 | 作用 |
|---|---|
| `innovation_points_upgrade_2025_2026.md` | 重新定义 2025/2026 顶会导向的创新点组合 |
| `Edge-symbol-obb_实验计划.md` | 记录 QG-OBB、SOF-FPN、RV1106-aware、GIS-Aug 的参考论文与实现顺序 |
| `top_conference_paper_selection_report.md` | 给出顶会论文筛选、模块归属和落地建议 |
| `项目论文大纲.docx` | 论文主线：轻量 OBB、方向纹理、量化友好 head、RV1106 部署 |
| `期刊参考论文_reference_papers/README_参考论文下载清单.md` | 期刊风格参考论文清单 |
| `HACMatch_ Semi-supervised rotation regression...pdf` | rotation regression 补充参考，相关但不是 QG 主出处 |

相关性审计见：

```text
docs/paper_reference_relevance_audit.md
```

## 3. QG-OBB 的主论文出处

根据 `paper` 目录中的创新点升级方案，QG-OBB Head 的直接依据应为以下三篇：

| 论文 | 出处 | 与 QG-OBB 的关系 | 相关性判断 |
|---|---|---|---|
| `GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection` | CVPR 2025 | 用 Cholesky 分解直接回归 Gaussian 表示，理论上缓解 OBB 角度边界不连续 | 强相关，完整 QG 版本主出处 |
| `RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark` | CVPR 2025 | 提出 Unit Cycle Resolver，用单位圆约束改善旋转目标角度预测 | 强相关，最小 sin-cos / unit-cycle 版本主出处 |
| `Inlier-Centric Post-Training Quantization for Object Detection Models` | ICLR 2026 | 针对检测模型 PTQ 区分 informative inliers 和 anomalies，改善量化范围选择 | 强相关，INT8 校准与量化稳定性主出处 |

补充相关文献：

| 论文 | 出处 | 与本项目关系 | 相关性判断 |
|---|---|---|---|
| `HACMatch: Semi-supervised rotation regression with hardness-aware curriculum pseudo labeling` | CVIU 2026 | rotation regression、几何完整性增强、hardness-aware curriculum | 中等相关，可作为角度学习和困难样本训练背景 |
| `Reg-PTQ: Regression-specialized Post-training Quantization for Fully Quantized Object Detector` | CVPR 2024 | 关注 detection regression 分支的 PTQ 问题 | 中等相关，可作为量化背景 |
| `Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection` | arXiv 2021 / TIP 2024 | Gaussian bbox 与 ProbIoU，是 GauCho 的背景脉络之一 | 背景相关，不作为最新主出处 |

为方便后续阅读，已生成中文论文解读：

| 文档 | 作用 |
|---|---|
| `docs/paper_gaucho_cn.md` | 理解 GauCho 的 Cholesky Gaussian OBB 表示 |
| `docs/paper_rsar_cn.md` | 理解 RSAR / Unit Cycle Resolver 的角度周期处理 |
| `docs/paper_inlierq_cn.md` | 理解 InlierQ 的检测 PTQ 校准思想 |
| `docs/paper_hacmatch_cn.md` | 理解 HACMatch 与 rotation regression 的补充关系 |
| `docs/paper_reg_ptq_cn.md` | 理解 Reg-PTQ 与检测回归分支量化的关系 |
| `docs/paper_gaussian_probiou_cn.md` | 理解 Gaussian bbox / ProbIoU 作为 GauCho 的背景脉络 |

## 4. 论文对比对象：官方 YOLOv8-OBB

论文中 QG-OBB 的主要对比对象应是官方标准 YOLOv8-OBB，而不是本仓库已有的历史魔改模型。

推荐对比层次：

| 层次 | 对比对象 | 用途 |
|---|---|---|
| 官方结构 baseline | YOLOv8n-OBB / YOLOv8s-OBB | 证明新方案相对标准 OBB head 的算法收益 |
| 当前工程 baseline | RV1106-M2 lightweight OBB | 证明方案可落到资源受限 NPU 友好模型上 |
| 部署 baseline | ONNX / RKNN INT8 的当前模型 | 证明量化和部署稳定性 |

需要注意：

- `configs/baseline/yolov8n_obb_official_arch.yaml` 只是官方架构参考；当前仓库的 head/loss 已修改，因此严格官方 baseline 需要在干净上游代码或独立分支中复现。
- 论文表格中不能把当前 `ultralytics/cfg/models/v8/yolov8-obb.yaml` 称为官方 baseline，因为该文件已经被本项目修改过。
- 新方案的消融应写成：`Official YOLOv8-OBB Head` vs `QG-OBB Head`，再额外报告 `RV1106-M2 + QG-OBB` 的部署版本。

## 5. 官方 YOLOv8-OBB 的角度分支问题

官方 YOLOv8-OBB 的核心输出可概括为：

```text
box branch: 位置与尺寸相关分布
cls branch: 类别
angle branch: 单标量角度 theta
```

这种单标量角度回归存在三个与本项目强相关的问题：

1. **角度边界不连续**
   旋转框角度有周期性，标量 theta 在定义域边界附近可能出现数值跳变。

2. **旋转状态约束不足**
   二维码/条形码的朝向具有方向轴等价关系，尤其细长条码对角度误差敏感。单标量回归没有显式使用单位圆或 Gaussian 约束。

3. **INT8 量化敏感**
   OBB 的角度分支是回归分支，微小量化误差可能通过 `sin/cos` 解码放大为框中心和方向偏移。

这些问题正对应：

| 问题 | 对应论文依据 |
|---|---|
| 角度边界和表示不连续 | GauCho, RSAR |
| 单标量角度状态约束弱 | RSAR |
| 检测模型回归分支量化敏感 | InlierQ, Reg-PTQ |

## 6. 新方案总体定位

QG-OBB Head 的完整目标是：

```text
Quantization-stable Gaussian Oriented Bounding Box Head
```

中文定位：

```text
面向 INT8 量化稳定性的连续高斯旋转框检测头
```

完整理想版本：

```text
官方 YOLOv8-OBB: (x, y, w, h, theta)
QG-OBB 完整版:  (x, y, l11, l21, l22) 或等价 Gaussian / Cholesky 表示
```

但从工程风险看，第一阶段不直接替换整个 OBB 参数化，而采用 RSAR 风格的最小实现：

```text
官方 YOLOv8-OBB: scalar theta
QG-OBB 第一阶段: unit-cycle / sin-cos angle encoding
```

本项目第一阶段建议：

```text
theta -> [sin(2theta), cos(2theta)]
theta = 0.5 * atan2(sin2theta, cos2theta)
```

原因：

- RSAR 支持单位圆约束处理旋转目标角度。
- OBB 矩形方向轴具有 `pi` 周期，`theta` 与 `theta + pi` 表示同一方向轴。
- 双角表达 `[sin(2theta), cos(2theta)]` 比普通 `[sin(theta), cos(theta)]` 更贴合 OBB 几何。
- 该方案只增加 angle branch 最后一层 1 个通道，便于 smoke test、ONNX、RKNN 验证。

## 7. 新方案与官方 YOLOv8-OBB 对比

| 对比项 | 官方标准 YOLOv8-OBB | QG-OBB 第一阶段 |
|---|---|---|
| 角度输出 | 单标量 theta | 双角单位圆向量 `[sin(2theta), cos(2theta)]` |
| 理论依据 | 常规 OBB angle regression | RSAR 的 unit-cycle 思想；GauCho 的连续 OBB 表示动机 |
| 周期处理 | 主要依赖 loss / 后处理吸收周期性 | 输出空间本身显式表达周期 |
| 边界连续性 | 定义域边界附近可能跳变 | 单位圆空间中边界连续 |
| 对细长条码 | 角度误差会明显影响框偏移 | 通过连续方向表达降低边界误差风险 |
| 量化稳定性 | 回归分支对 INT8 较敏感 | 输出分量被约束在单位圆附近，便于分析和校准 |
| ONNX 输出 | scalar angle | angle vector，后处理解码回 scalar |
| RKNN 实现 | 直接读取 theta | CPU 后处理 `atan2` 解码，避免 RKNN 图中复杂算子 |
| 参数/FLOPs | 原始 head | 仅 angle 最后一层多 1 个输出通道，增量极小 |
| 后续扩展 | 标量 OBB head | 可升级到 GauCho 风格 Cholesky Gaussian Head |

具体优化方式：

1. **角度表示优化**
   从普通标量 theta 改为单位圆向量，显式处理角度周期。

2. **OBB 周期适配优化**
   用 `2theta` 处理 OBB 的 `pi` 周期，避免把 `theta` 和 `theta + pi` 当成相反方向。

3. **量化鲁棒性优化**
   将角度预测约束到更稳定的向量空间，并为后续 InlierQ 风格校准提供 angle branch 误差分析入口。

4. **部署风险控制**
   ONNX/RKNN 图只输出原始双通道向量，`atan2` 放在 CPU 后处理，避免引入 RKNN 不支持算子。

5. **可扩展性**
   第一阶段 sin-cos 是 RSAR 风格最小可验证版本；若成立，再进一步尝试 GauCho 的 Cholesky Gaussian 表示。

## 8. 当前代码接口审计

当前项目工程 baseline 已经不是官方 YOLOv8-OBB，而是 RV1106-M2 轻量化版本。它的作用是提供可复现部署链路。

当前配置：

```text
configs/rv1106/yolov8n_obb_rv1106_m2.yaml
nc: 15
ch: 1
最后一层 head: [[15, 18, 21], 1, OBB, [15, 1]]
```

在 `imgsz=256` 时：

| 层级 | Stride | 网格 | Anchor 数 |
|---|---:|---:|---:|
| P3 | 8 | `32 x 32` | 1024 |
| P4 | 16 | `16 x 16` | 256 |
| P5 | 32 | `8 x 8` | 64 |
| 合计 | | | 1344 |

当前 `ultralytics/nn/modules/head.py` 中本地修改版 `OBB`：

- `reg_max = 8`
- `self.no = nc + reg_max * 4 = 47`
- bbox 分支每层输出 32 通道
- cls 分支每层输出 15 通道
- angle 分支当前输出 `ne=1`
- `onnx_4head` 每层输出 `32 + 15 + 1 = 48` 通道

训练输出：

```text
feats:
  P3: [B, 47, 32, 32]
  P4: [B, 47, 16, 16]
  P5: [B, 47, 8, 8]
angle:
  [B, 1, 1344]
```

ONNX 4-head 当前输出：

```text
out_p3: [B, 48, 32, 32]
out_p4: [B, 48, 16, 16]
out_p5: [B, 48, 8, 8]
angle:  [B, 1, 1344]
```

QG 第一阶段应变为：

```text
angle_vec: [B, 2, 1344]
```

其余输出尽量不变。

## 9. 必需代码改动

### 9.1 新增模型配置

新增：

```text
configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml
```

从当前 RV1106-M2 复制，只改最后一层：

```text
- [[15, 18, 21], 1, OBB, [15, 2]]
```

这不是论文对比对象的替代，只是 QG 方案在当前部署工程中的落地版本。

### 9.2 `ultralytics/nn/modules/head.py`

需要在现有 `OBB` 类中加入兼容逻辑：

- `self.ne == 1` 时完全保持当前 scalar angle 行为。
- `self.ne == 2` 时启用双角向量。
- 增加 `_decode_angle(angle)`，输出 scalar theta。
- 训练时返回原始 `angle_vec`，loss 内部解码。
- 普通推理时，在 `super().forward(x)` 前把 `self.angle` 设置为 decoded theta。
- `onnx_4head` 返回原始 `angle_vec`，后处理解码。

### 9.3 `ultralytics/utils/loss.py`

更新 `v8OBBLoss`：

- 兼容 1 通道 scalar angle。
- 兼容 2 通道 angle vector。
- 2 通道时先解码 theta，再参与 `bbox_decode`、assigner 和当前 angle loss。
- 增加 unit-cycle / vector alignment loss。

建议：

```text
target_vec = [sin(2 * target_theta), cos(2 * target_theta)]
pred_unit = normalize(pred_vec)
loss_vec = 1 - dot(pred_unit, target_vec)
loss_unit = (norm(pred_vec) - 1)^2
loss_angle = current_periodic_theta_loss + alpha * loss_vec + beta * loss_unit
```

初始系数：

```text
alpha = 0.25
beta = 0.05
```

### 9.4 `scripts/eval/val_onnx_gray.py`

更新 4-head decode：

- 支持 `[B, 1, 1344]`。
- 支持 `[B, 2, 1344]`。
- 2 通道时解码到 scalar theta 后再调用 `dist2rbox`。
- 返回给 Ultralytics validator 的形状保持：

```text
[B, 4 + nc + 1, 1344]
```

### 9.5 `scripts/deploy_rv1106/convert_eval_rknn.py`

更新 RKNN 后处理：

- 支持 scalar angle 长度 `1344`。
- 支持 QG angle vector 长度 `2688` 或 shape `[1, 2, 1344]`。
- CPU 后处理逐 anchor 解码 theta。
- 最终 detection 格式保持：

```text
[cls_id, cx, cy, w, h, angle, score]
```

## 10. 实验计划

### 10.1 官方 YOLOv8-OBB 对比实验

论文主对比必须包含：

| 模型 | 目的 |
|---|---|
| Official YOLOv8n-OBB | 最小官方标准 baseline |
| Official YOLOv8s-OBB | 更强官方标准 baseline |
| YOLOv8n-OBB + QG-OBB Head | 验证 head 改造收益 |
| RV1106-M2 lightweight OBB | 工程轻量部署 baseline |
| RV1106-M2 + QG-OBB Head | 验证部署场景下收益 |

### 10.2 Shape Test

```text
model: configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml
input: [1, 1, 256, 256]
expected training angle output: [1, 2, 1344]
expected ONNX 4-head angle output: [1, 2, 1344]
```

### 10.3 Smoke Train

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --name rv1106_qg_sincos_smoke \
  --epochs 1 \
  --batch 128 \
  --workers 8
```

通过标准：

- 无 shape error。
- loss 没有 NaN。
- `angle_loss` 有限。
- validation 能跑完。

### 10.4 Quick / Full Train

Quick：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --name rv1106_qg_sincos_e20_b512 \
  --epochs 20 \
  --batch 512 \
  --workers 16
```

Full：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --name rv1106_qg_sincos_e100_b512 \
  --epochs 100 \
  --batch 512 \
  --workers 16 \
  --angle 5.0 \
  --qg-angle-align 0.25 \
  --qg-angle-unit 0.05
```

门槛：

| 指标 | 要求 |
|---|---:|
| PyTorch mAP50 | 不低于官方 YOLOv8-OBB 或当前部署 baseline 的合理误差范围 |
| PyTorch mAP50-95 | 不明显下降 |
| ONNX export | PASS |
| RKNN INT8 AP drop | 小于 scalar angle 对照 |
| angle MAE / boundary-angle error | 应优于 scalar angle 对照 |

## 11. 当前结论

本项目 QG-OBB 的主创新逻辑应从旧的“sin-cos 角度表达”升级为：

```text
以 GauCho 为完整连续 Gaussian OBB 表示目标，
以 RSAR 的 Unit Cycle Resolver 作为第一阶段可落地角度编码，
以 InlierQ 作为 INT8 检测量化校准和部署稳定性依据。
```

第一阶段仍建议实现双角 sin-cos，因为它是最小可验证、最容易兼容 ONNX/RKNN 的路径；但论文表述中应明确它是 QG-OBB 的保守实现阶段，而不是最终完整 Gaussian/Cholesky Head。

## 12. 阶段 A/B 后的实验更新

阶段 A/B 已完成 20 epoch 快速对照。当前正式候选为 `QG 原始 0.25/0.05`，不是 `decode-only`。

| 方案 | ONNX mAP50 | ONNX mAP50-95 | RKNN INT8 mAP50 | ONNX 到 RKNN 掉点 |
|---|---:|---:|---:|---:|
| Scalar RV1106-M2 | 0.9561 | 0.8832 | 0.9400 | 0.0161 |
| QG 原始 0.25/0.05 | 0.9548 | 0.8859 | 0.9427 | 0.0121 |
| QG decode-only 0/0 | 0.9535 | 0.8864 | 0.9405 | 0.0130 |

后续正式实验应使用 `docs/qg_stage2_experiment_plan.md` 中的固定命令执行，避免继续手动修改 `loss.py`。

## 13. QG-OBB e100 正式结果与结论边界

QG-OBB e100 已完成 PyTorch、ONNX、RKNN INT8 全链路验证。对比对象为同一工程 baseline 下的 `RV1106-M2 scalar e100`。

| 模型 | PyTorch mAP50 | PyTorch mAP50-95 | ONNX mAP50 | ONNX mAP50-95 | RKNN INT8 mAP50 | ONNX 到 RKNN 掉点 |
|---|---:|---:|---:|---:|---:|---:|
| RV1106-M2 scalar e100 | 0.991 | 0.960 | 0.9905 | 0.9596 | 0.9843 | 0.0062 |
| QG-OBB e100 | 0.990 | 0.957 | 0.9895 | 0.9577 | 0.9854 | 0.0041 |

当前可以成立的结论：

- QG-OBB 保持了接近 scalar angle head 的 FP32/ONNX 检测精度。
- QG-OBB 在 RKNN INT8 评估中取得略高 mAP50。
- QG-OBB 降低了 ONNX 到 RKNN INT8 的精度掉点。
- 因此，QG-OBB 第一阶段应定位为量化友好的 unit-cycle angle branch。

当前不应过度主张：

- 不应写作“QG-OBB 显著提升检测精度”。
- 不应把 simulator 耗时作为真实 RV1106 板端速度结论。
- 不应把 RV1106-M2 工程 baseline 替代为论文主对比对象。

下一步需要补官方 YOLOv8-OBB baseline，用于论文主对比。
