# 论文解读：CSL 与旋转框角度周期问题

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 会议论文 | `Arbitrary-Oriented Object Detection with Circular Smooth Label` |
| 作者 | Xue Yang, Junchi Yan |
| 会议 | ECCV 2020 |
| 链接 | https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/666_ECCV_2020_paper.php |
| 扩展版 | `On the Arbitrary-Oriented Object Detection: Classification based Approaches Revisited` |
| 扩展版链接 | https://arxiv.org/abs/2003.05597 |

## 2. 论文要解决的问题

旋转目标检测中，角度参数化经常带来边界问题。

一个典型例子：

```text
角度定义范围是 [-90, 0)
真实角度接近 -90
预测角度接近 0
```

从几何上看，这两个框可能很接近；但从标量角度看，它们处在定义域两端，loss 会认为误差很大。

CSL 论文指出，这类问题的根源是：

- angle periodicity：角度本身是周期变量。
- corner ordering：四点顺序也会造成表示不连续。
- ideal prediction may be outside defined range：理想预测可能落到定义域之外。

## 3. 核心创新

CSL 的核心做法是把角度预测从回归变成分类，并用 circular smooth label 处理周期边界。

普通 one-hot 分类：

```text
目标角度类别附近只有一个类别为 1
其他类别为 0
```

CSL：

```text
目标角度附近一段类别按窗口函数平滑赋值
并且类别序列首尾相接，形成圆形标签
```

这样，角度 0 附近和角度最大类别附近会被视为相邻，而不是断开的两端。

## 4. 为什么对本项目有用

本项目不准备第一版使用 angle classification，因为：

- 分类角度会增加输出通道。
- 高精度角度分类需要较多 bins。
- 对 RV1106/RKNN 来说输出和后处理成本更高。
- 我们当前 baseline 的 scalar OBB 链路已经成立，第一步应尽量小改动。

但 CSL 对本项目非常重要，因为它证明了一个关键事实：

```text
旋转框检测的角度边界问题是真实存在的，不能把角度当普通实数处理。
```

这为 QG-OBB 的 sin-cos 周期表达提供了问题依据。

## 5. 与本项目方案的关系

CSL 的解决路线：

```text
角度回归 -> 角度分类 + 圆形平滑标签
```

本项目 QG-OBB 的解决路线：

```text
角度标量回归 -> 双角向量回归 [sin(2theta), cos(2theta)]
```

两者都在解决：

```text
角度周期边界不连续
```

但实现方式不同：

| 项目 | CSL | QG-OBB |
|---|---|---|
| 角度输出 | 多类别 bins | 2 通道向量 |
| 周期处理 | circular label 首尾相接 | 双角单位圆表达 |
| 输出通道 | 较多 | 只增加 1 个通道 |
| 部署复杂度 | 分类后解码 | CPU `atan2` 解码 |
| 适合 RV1106 | 成本偏高 | 更轻量 |

## 6. 阅读时重点关注

建议重点读：

1. 论文如何定义 rotation detector 的 boundary problem。
2. 为什么边界问题由 angle periodicity 或 corner ordering 引起。
3. CSL 如何让角度类别首尾相接。
4. window radius 对容错性的影响。
5. 为什么角度分类能缓解边界不连续，但会增加输出维度。

## 7. 对 QG-OBB 的直接启发

| CSL 启发 | QG-OBB 用法 |
|---|---|
| 角度有周期边界 | 不再把 theta 当普通标量直接优化 |
| 定义域边界导致 loss 异常 | 用圆空间向量消除硬边界 |
| 相邻角度应有容错 | vector alignment 对角度误差连续变化 |
| 旋转检测需要专门角度设计 | 单独改造 angle branch，而不是只调 box loss |

## 8. 本项目不采用的部分

当前不采用 CSL 的角度分类形式，也不引入多 bin circular label。

原因是当前项目优先级是：

```text
低成本、可导出、可 RKNN INT8 验证
```

因此第一版采用 2 通道双角向量。如果后续发现 sin-cos 收益不足，可以再把 CSL/DCL 作为独立 ablation 方向。
