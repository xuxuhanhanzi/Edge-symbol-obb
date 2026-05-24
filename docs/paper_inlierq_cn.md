# 论文解读：InlierQ

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | `Inlier-Centric Post-Training Quantization for Object Detection Models` |
| 简称 | InlierQ |
| 出处 | ICLR 2026 |
| 方向 | Post-training quantization, object detection, calibration |
| 项目相关性 | 强相关，QG-OBB 量化稳定性主出处 |
| OpenReview | https://openreview.net/forum?id=GN9otzf5o6 |
| arXiv | https://arxiv.org/abs/2602.03472 |

## 2. 论文要解决的问题

目标检测模型的 PTQ 量化不同于分类模型。

检测图像中往往存在：

- 大量背景区域。
- 反光、纹理、噪声等异常激活。
- 少量真正与检测任务相关的目标区域激活。

如果 PTQ 校准时被背景异常值主导，量化范围会被拉大，真正有用的目标区域特征反而被压缩。

本项目也有同类问题：

```text
工业二维码/条形码图像
-> 背景纹理、反光、低照度、噪声
-> 可能干扰 INT8 校准
-> angle branch 对量化误差敏感
```

## 3. 核心创新

InlierQ 的核心思想是：

```text
不要让异常背景激活主导量化范围
而是围绕 informative inliers 选择和优化量化范围
```

文档中总结为：

```text
inlier-centric calibration
```

即关注真正对检测任务有贡献的激活，而不是简单随机抽样或使用全局 min/max。

## 4. 为什么与 QG-OBB 强相关

QG-OBB 不只是角度表示问题，还包含：

```text
Quantization-stable
```

也就是要证明：

```text
FP32 -> INT8 后，角度分支更稳定
```

InlierQ 正好提供了量化校准层面的理论依据：

- 校准集不能只随机抽样。
- 应覆盖目标区域、困难角度、小目标、细长条码、反光样本。
- 应单独统计 angle branch 的 FP32/INT8 输出漂移。

## 5. 本项目如何使用 InlierQ

第一阶段不实现完整 InlierQ 算法，而是采用其思想设计校准和分析流程。

建议校准集覆盖：

```text
极端角度样本
细长条形码样本
小二维码样本
反光样本
低照度样本
遮挡样本
密集多码样本
```

量化分析指标：

```text
FP32 mAP
INT8 AP@0.5
INT8 drop
angle MAE before/after INT8
boundary-angle error before/after INT8
elongated-object AP drop
```

## 6. 对 QG-OBB 的具体启发

| InlierQ 思想 | 本项目对应设计 |
|---|---|
| 背景异常激活不应主导 PTQ | 校准集按目标和困难样本构造 |
| informative inliers 更重要 | 关注目标框区域和 angle branch |
| 检测任务 PTQ 需区别于分类 | 单独报告 OBB angle quantization drift |
| 少量校准样本也可有效 | 构造覆盖性强的 calibration subset |

## 7. 阅读重点

阅读时建议重点关注：

1. Inlier / anomaly 如何定义。
2. detection PTQ 为什么比 classification PTQ 更难。
3. 论文如何选择 calibration samples。
4. 它如何度量激活对检测任务的重要性。
5. 是否有可以简化迁移到 RKNN INT8 校准的流程。

## 8. 当前阶段结论

InlierQ 是 QG-OBB 中 “Quantization-stable” 的主要依据。当前第一阶段先实现 unit-cycle angle branch；后续 RKNN INT8 验证时，应补充 InlierQ-inspired calibration 和 angle branch 量化漂移分析。
