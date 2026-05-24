# 论文解读：RSAR / Unit Cycle Resolver

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | `RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark` |
| 出处 | CVPR 2025 |
| 方向 | Rotated object detection, angle representation, SAR benchmark |
| 项目相关性 | 强相关，QG-OBB 第一阶段主出处 |
| PDF | https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_RSAR_Restricted_State_Angle_Resolver_and_Rotated_SAR_Benchmark_CVPR_2025_paper.pdf |
| arXiv | https://arxiv.org/abs/2501.04440 |

## 2. 论文要解决的问题

旋转目标检测中的角度估计有周期性和状态约束问题。

当目标旋转接近角度定义边界时，标量角度回归可能发生不连续；当目标存在不同等价状态时，模型可能学习到不稳定的角度表示。

这与本项目中的问题高度一致：

```text
二维码/条形码任意旋转
-> OBB angle branch 需要稳定预测方向
-> INT8 后微小角度误差可能影响框定位和裁剪
```

## 3. 核心创新

RSAR 提出 Restricted State Angle Resolver，并包含 Unit Cycle Resolver 思想。

可理解为：

```text
不把角度只当普通标量
而是通过单位圆约束处理角度周期状态
```

这和 QG-OBB 第一阶段的做法直接对应：

```text
theta -> [sin(2theta), cos(2theta)]
```

## 4. 为什么与 QG-OBB 强相关

当前项目第一阶段的实现目标是最小改动：

```text
官方 YOLOv8-OBB scalar angle
-> unit-cycle angle branch
```

RSAR 正好提供了单位圆角度约束的理论依据。

相比完整 GauCho，RSAR 更适合作为第一阶段原因：

- 改动小。
- 只改 angle branch。
- 更容易兼容 YOLOv8-OBB 解码。
- 更容易导出 ONNX。
- RKNN 后处理只需 CPU `atan2`。

## 5. 本项目如何适配 RSAR

RSAR 面向旋转 SAR 目标，本项目面向工业符号 OBB。两者任务不同，但角度周期问题相同。

本项目适配方式：

```text
RSAR / UCR:
  unit circle constraint for angle

QG-OBB:
  angle branch outputs [sin(2theta), cos(2theta)]
  decode theta = 0.5 * atan2(sin2, cos2)
```

使用 `2theta` 的原因：

- OBB 矩形方向轴是 `pi` 周期。
- `theta` 与 `theta + pi` 表示同一个方向轴。
- `[sin(2theta), cos(2theta)]` 正好把 `pi` 周期映射成单位圆上的 `2pi` 周期。

## 6. 对 QG-OBB 的具体启发

| RSAR 思想 | 本项目对应设计 |
|---|---|
| 角度状态需要约束 | 增加 unit-cycle angle branch |
| 单位圆处理角度周期 | 使用 sin-cos vector |
| 限制角度状态不稳定 | vector alignment + unit norm penalty |
| 可插入现有旋转检测器 | 在现有 OBB head 中用 `ne=2` 兼容实现 |

## 7. 阅读重点

阅读时建议重点关注：

1. Unit Cycle Resolver 如何表达角度。
2. 它如何缓解角度边界问题。
3. loss 中如何约束单位圆。
4. 它是否改变原检测器主体结构。
5. 论文如何证明角度预测改善。

## 8. 当前阶段结论

RSAR 是当前 QG-OBB 第一阶段最直接、最重要的出处。我们现在要实现的双角 sin-cos 方案，应表述为：

```text
RSAR-inspired unit-cycle OBB angle branch
```

而不是泛泛地说“sin-cos 编码”。
