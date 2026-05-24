# 论文解读：Reg-PTQ

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | `Reg-PTQ: Regression-specialized Post-training Quantization for Fully Quantized Object Detector` |
| 会议 | CVPR 2024 |
| 作者 | Yifu Ding, Weilun Feng, Chuyan Chen, Jinyang Guo, Xianglong Liu |
| 官方页面 | https://cvpr.thecvf.com/virtual/2024/poster/30629 |
| OpenAccess | https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html |
| 项目相关性 | 中等相关，检测回归分支量化背景文献 |

## 2. 论文要解决的问题

常规 PTQ 方法很多来自分类模型。分类模型主要关注类别判别，对小幅数值误差相对不那么敏感。

目标检测不同。检测模型除了分类分支，还有 bbox regression 分支。回归分支输出的是位置、尺寸、偏移或分布参数，小的量化误差可能直接造成框偏移、尺度错误或 IoU 下降。

Reg-PTQ 的核心问题是：

```text
检测模型的回归分支对量化误差更敏感，不能简单套用分类模型 PTQ。
```

## 3. 核心创新点

Reg-PTQ 的重点不是提出新的检测 head，而是提出 regression-specialized 的 PTQ 思路。

可以从三个角度理解：

1. **区分分类与回归的量化需求**  
   分类分支关心类别排序，回归分支关心连续数值精度。两者不能完全使用同一套量化策略。

2. **为回归分支设计更友好的量化处理**  
   论文关注回归输出在低比特量化下的误差传播，并尝试降低量化对框定位的破坏。

3. **推动 fully quantized detector**  
   论文目标是让检测模型尽可能完整量化，而不是只量化部分卷积层。

## 4. 为什么与 QG-OBB 有关

本项目的 QG-OBB 也面对一个回归分支量化问题：

```text
官方 YOLOv8-OBB:
box regression + scalar angle regression

QG-OBB 第一阶段:
box regression + unit-cycle angle vector regression
```

无论使用 scalar theta 还是 `[sin(2theta), cos(2theta)]`，角度分支本质上仍是连续回归。INT8 量化误差可能导致角度漂移，进而影响细长工业符号的 OBB 定位。

因此 Reg-PTQ 可以支撑以下论述：

- 检测模型回归分支确实比分类分支更量化敏感。
- QG-OBB 需要单独统计 angle branch 的 FP32 到 INT8 漂移。
- 量化实验不能只看整体 mAP，还应分析角度误差、长宽比目标误差和困难样本误差。

## 5. 为什么不是 QG-OBB 主出处

Reg-PTQ 与本项目相关，但不是 QG-OBB 的主创新出处。

原因：

- 它不解决 OBB 角度周期不连续。
- 它不提出 Gaussian OBB 或 Cholesky OBB head。
- 它不直接给出 unit-cycle angle encoding。
- 它主要是检测 PTQ 框架，不是 OBB head 结构设计论文。

因此在本文中更合适的定位是：

```text
Reg-PTQ: 检测回归分支量化敏感性的背景依据
InlierQ: 后续 INT8 calibration 的主要量化依据
RSAR: 第一阶段 unit-cycle angle branch 的主要角度表示依据
GauCho: 完整 QG-OBB Gaussian head 的主要表示依据
```

## 6. 本项目可借鉴的实验设计

| Reg-PTQ 思想 | 本项目对应设计 |
|---|---|
| 回归分支更量化敏感 | 单独统计 box branch 与 angle branch 量化误差 |
| 分类和回归不能混在一起分析 | mAP 之外增加 angle MAE / corner error / slender-object AP |
| fully quantized detector 更接近部署需求 | 除 PyTorch 指标外，必须报告 ONNX 和 RKNN INT8 指标 |
| 低比特量化会放大定位误差 | 重点观察细长符号、密集符号和边界角度样本 |

## 7. 阅读时重点关注

1. 作者如何证明 regression branch 比 classification branch 更敏感。
2. 论文如何设计回归友好的 PTQ。
3. 它的校准数据选择、量化粒度和误差度量。
4. 它的实验是否报告 localization error，而不只是整体 mAP。
5. 哪些思想能迁移到 RKNN INT8，而哪些需要训练框架或量化器支持。

## 8. 结论

Reg-PTQ 适合作为 QG-OBB 的量化背景文献，尤其用来解释为什么 angle branch 不能只按普通分类分支方式量化。

但它不是本项目 QG-OBB 的主结构出处。主结构出处仍应是 GauCho 和 RSAR，量化稳定性的主出处应是 InlierQ，Reg-PTQ 作为检测回归 PTQ 的补充依据。
