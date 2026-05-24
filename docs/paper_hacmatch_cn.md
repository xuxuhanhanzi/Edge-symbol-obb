# 论文解读：HACMatch

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | `HACMatch: Semi-supervised rotation regression with hardness-aware curriculum pseudo labeling` |
| 出处 | Computer Vision and Image Understanding, 2026 |
| DOI | `10.1016/j.cviu.2026.104742` |
| 本地 PDF | `C:\Users\27475\Desktop\paper\期刊参考论文_reference_papers\07_Computer Vision and Image Understanding\HACMatch_ Semi-supervised rotation regression with hardness-aware curriculum pseudo labeling.pdf` |
| 项目相关性 | 中等相关，rotation regression 和困难样本训练补充参考 |

## 2. 论文要解决的问题

HACMatch 研究的是从 2D 图像进行 3D rotation regression，并且重点关注半监督场景。

它指出：

- rotation regression 需要大量标注数据。
- 伪标签过滤如果只用固定阈值，难以区分可靠和不可靠样本。
- 训练应从容易样本逐步过渡到困难样本。

## 3. 核心创新

论文提出 hardness-aware curriculum pseudo labeling。

简化理解：

```text
先选择容易且可靠的旋转伪标签
再逐步加入更困难的样本
```

同时论文提出结构化数据增强，目的是增加特征多样性，同时保持关键几何完整性。

## 4. 与本项目的关系

HACMatch 与 QG-OBB 有共同点：

- 都关注 rotation / angle prediction。
- 都强调几何完整性。
- 都关注困难样本对角度学习的影响。

但它不是 QG-OBB 的主出处，原因：

- 它不是 OBB detection head。
- 它不是 YOLOv8-OBB。
- 它不关注 RKNN / INT8 部署。
- 它主要解决半监督 rotation regression，而非 OBB 角度参数化。

## 5. 可借鉴点

| HACMatch 思想 | 本项目可能用法 |
|---|---|
| hardness-aware curriculum | 对极端角度、反光、遮挡样本做分阶段训练或重采样 |
| rotation regression | 角度误差分析可借鉴 |
| structured augmentation | GIS-Aug 应保持符号几何完整性 |
| low-data regimes | 如果工业困难样本少，可用 curriculum 增强训练稳定性 |

## 6. 当前阶段结论

HACMatch 是补充相关文献。它适合放在 Related Work 或训练策略讨论中，但不应作为 QG-OBB Head 的主创新出处。
