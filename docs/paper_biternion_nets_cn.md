# 论文解读：Biternion Nets

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | `Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels` |
| 作者 | Lucas Beyer, Alexander Hermans, Bastian Leibe |
| 会议 | GCPR 2015 |
| 链接 | https://dblp.org/rec/conf/dagm/BeyerHL15 |
| DOI | `10.1007/978-3-319-24947-6_13` |

## 2. 论文要解决的问题

这篇论文关注的是头部姿态角度估计。角度预测有一个典型难点：角度不是普通实数，而是周期变量。

例如：

```text
359 度 和 0 度在数值上差 359
但在方向意义上只差 1 度
```

如果模型直接回归一个角度标量，普通 L1/L2 loss 会把这种边界附近的预测看成巨大误差，从而造成训练不稳定。

## 3. 核心创新

论文提出用单位圆上的二维向量表达角度，也就是 biternion 表达。

一般形式：

```text
angle phi -> [cos(phi), sin(phi)]
```

这样一来，角度预测不再是直线上的标量回归，而是圆上的方向预测。

边界附近的例子：

```text
0 度     -> [1, 0]
359 度   -> [cos(359), sin(359)]，非常接近 [1, 0]
```

因此，角度边界在向量空间中变得连续。

## 4. 为什么对本项目有用

本项目当前 OBB head 直接输出一个角度标量：

```text
theta = angle_raw
```

这种方式虽然配合了周期 angle loss，但输出空间本身仍然是普通实数轴。对 INT8 量化来说，边界附近或幅值漂移可能带来不稳定。

Biternion 的启发是：

```text
不要直接让网络学习角度数值
而是让网络学习角度在单位圆上的方向向量
```

这正是 QG-OBB 最小方案采用 sin-cos angle branch 的理论来源。

## 5. 与本项目方案的区别

Biternion 原论文用于头部朝向，通常是完整 360 度方向问题，因此使用：

```text
[cos(theta), sin(theta)]
```

但 OBB 旋转框中，`theta` 和 `theta + pi` 表示同一条矩形方向轴，因此本项目不直接照搬普通 sin-cos，而是改为双角表达：

```text
[sin(2theta), cos(2theta)]
```

这是对 Biternion 思想的 OBB 适配。

## 6. 阅读时重点关注

阅读这篇论文时建议重点看：

1. 为什么角度不能按普通实数直接回归。
2. 单位圆向量如何解决周期边界问题。
3. loss 如何鼓励预测向量接近目标方向。
4. 预测向量是否需要单位长度约束。

## 7. 对 QG-OBB 的直接启发

| Biternion 思想 | QG-OBB 用法 |
|---|---|
| 用二维向量表示角度 | angle branch 从 1 通道变 2 通道 |
| 用圆空间处理周期 | 用 `[sin(2theta), cos(2theta)]` 处理 OBB 的 `pi` 周期 |
| 避免角度边界跳变 | 减少 scalar theta 在边界附近的不连续 |
| 需要向量归一化或约束 | 增加 unit-cycle penalty |

## 8. 本项目不采用的部分

Biternion Nets 解决的是头部姿态估计，不是目标检测；它没有涉及：

- OBB box decode
- DFL
- rotated assigner
- ONNX/RKNN 部署
- INT8 量化

因此本项目只借鉴角度表达方式，不照搬整体网络结构。
