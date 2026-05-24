# `paper` 目录参考文献相关性审计

## 1. 审计目的

本文件记录 `C:\Users\27475\Desktop\paper` 中与项目创新点相关的文档和论文，并判断它们是否适合作为当前项目创新点的依据。

审计结论用于修订：

```text
docs/qg_obb_head_design.md
```

## 2. 已阅读的关键文件

| 文件 | 结论 |
|---|---|
| `innovation_points_upgrade_2025_2026.md` | 最重要。明确将 QG-OBB 主出处升级为 GauCho、RSAR、InlierQ |
| `Edge-symbol-obb_实验计划.md` | 最重要。记录四个创新点及其论文出处 |
| `top_conference_paper_selection_report.md` | 重要。提供更完整的顶会论文筛选和模块落地建议 |
| `项目论文大纲.docx` | 重要。明确论文主线是轻量 OBB、方向保持、量化友好 head、RV1106 部署 |
| `Neurocomputing投稿模板与论文大纲/neurocomputing_project_outline.md` | 主要是写作和投稿结构参考，不直接定义算法 |
| `期刊参考论文_reference_papers/README_参考论文下载清单.md` | 主要是期刊风格参考，不全是创新点依据 |

已跳过：

```text
deepseek api_key.txt
```

原因：该文件是敏感信息，不属于论文或算法参考材料。

## 3. 创新点与论文相关性总表

| 创新点 | 主参考论文 | 发表时间 | 相关性 | 判断 |
|---|---|---:|---|---|
| QG-OBB Head | GauCho | CVPR 2025 | 强相关 | 直接针对 OBB 边界不连续，提出 Cholesky Gaussian head |
| QG-OBB Head | RSAR | CVPR 2025 | 强相关 | Unit Cycle Resolver 直接支撑 sin-cos / unit-cycle angle encoding |
| QG-OBB Head | InlierQ | ICLR 2026 | 强相关 | 支撑 INT8 PTQ 校准和检测模型量化稳定性分析 |
| QG-OBB Head | HACMatch | CVIU 2026 | 中等相关 | rotation regression 和结构增强相关，但不是 OBB head 主出处 |
| QG-OBB Head | Reg-PTQ | CVPR 2024 | 中等相关 | 支撑检测回归分支 PTQ 背景，但不是 OBB 角度表示主出处 |
| QG-OBB Head | Gaussian Bounding Boxes / ProbIoU | arXiv 2021 / TIP 2024 | 背景相关 | 支撑 Gaussian bbox 脉络，是 GauCho 的前置背景之一 |
| SOF-FPN | SET | CVPR 2025 | 强相关 | 支撑 spectral enhancement 和 tiny object feature enhancement |
| SOF-FPN | YOLOv12 | NeurIPS 2025 | 中强相关 | 支撑实时检测中的轻量注意力设计，但需验证 RV1106 友好性 |
| SOF-FPN | YOLO-RD | ICLR 2025 | 中强相关 | 支撑符号纹理 prototype / dictionary 思想 |
| RV1106-aware co-design | RF-DETR | ICLR 2026 | 中等相关 | 支撑 latency-aware Pareto 搜索思想，但不建议完整迁移 DETR |
| RV1106-aware co-design | InlierQ | ICLR 2026 | 强相关 | 支撑 inlier-centric calibration |
| RV1106-aware co-design | Efficient Test-time Adaptive Object Detection | CVPR 2025 | 中等相关 | 可借鉴 sensitivity-guided pruning |
| GIS-Aug | AeroGen | CVPR 2025 | 中强相关 | 支撑 OBB layout-guided synthetic data |
| GIS-Aug | Object Fidelity Diffusion | ICLR 2026 | 中强相关 | 支撑目标形态保真生成 |
| GIS-Aug | Unbiased Object Detection Beyond Frequency | ICLR 2026 | 中等相关 | 支撑长尾困难样本补齐 |

## 4. QG-OBB 相关文献判断

### 4.1 GauCho：强相关

相关原因：

- 直接面向 Oriented Object Detection。
- 明确讨论 OBB 的角度边界不连续。
- 提出用 Cholesky 分解直接回归 Gaussian distribution。
- 与项目中“Gaussian OBB Head”的完整版本高度一致。

落地建议：

```text
第一阶段先做 RSAR 风格 unit-cycle angle encoding。
第二阶段再评估 GauCho 风格 Cholesky Gaussian representation。
```

### 4.2 RSAR：强相关

相关原因：

- 提出 Unit Cycle Resolver。
- 用单位圆约束改善旋转目标角度估计。
- 与当前最小 sin-cos 方案直接对应。

落地建议：

```text
将 QG-OBB 第一阶段表述为 RSAR-inspired unit-cycle angle branch。
```

### 4.3 InlierQ：强相关

相关原因：

- 面向 object detection 的 post-training quantization。
- 区分 anomaly 和 informative inlier。
- 与工业场景中的反光、背景纹理、噪声拉大量化范围的问题高度相关。

落地建议：

```text
后续量化实验中设计 inlier-centric calibration subset。
重点统计 angle branch FP32 -> INT8 漂移。
```

### 4.4 HACMatch：中等相关

相关原因：

- 论文主题是 rotation regression。
- 提出 hardness-aware curriculum pseudo labeling 和结构化增强。
- 对困难角度样本、低标注数据、角度泛化有启发。

不作为主出处的原因：

- 它不是 OBB detection head。
- 它主要解决半监督 rotation regression，不是 INT8 部署或 OBB Gaussian 表示。

### 4.5 Reg-PTQ：中等相关

相关原因：

- 面向 object detector 的 post-training quantization。
- 特别关注检测模型中的 regression 分支，这一点与 OBB 的 box / angle 回归敏感性相关。
- 可为 QG-OBB 后续 INT8 部署实验提供量化误差分析背景。

不作为主出处的原因：

- 它不解决 OBB 角度周期或 Gaussian OBB 表示。
- 它更适合作为 InlierQ 之前的检测 PTQ 背景文献。

### 4.6 Gaussian Bounding Boxes / ProbIoU：背景相关

相关原因：

- 将 bounding box 与 Gaussian representation 联系起来。
- ProbIoU 是旋转框评估与损失设计中的重要背景。
- GauCho 的 Cholesky Gaussian head 可以看作沿着 Gaussian bbox 思路进一步把 head 输出也改成连续 Gaussian 参数。

不作为主出处的原因：

- 它不是本项目当前第一阶段 unit-cycle angle branch 的直接依据。
- 它的主要作用是帮助理解 GauCho，而不是直接定义本项目 QG-OBB 的最小实现。

## 5. 期刊参考论文判断

`期刊参考论文_reference_papers` 下的 PDF 大多用于学习期刊写作风格，并不都适合作为创新点依据。

| 论文 | 相关性 | 用途 |
|---|---|---|
| Neurocomputing SNN edge detection | 中等 | 学习“算法 + 硬件部署 + benchmark”的写法 |
| SO-YOLOv8 | 中等 | 学习小目标 YOLO 改法和 ESWA 写法，可辅助 SOF-FPN |
| TOE-YOLO | 中等 | tiny / rotated small object 与轻量融合相关，可辅助 SOF-FPN |
| HACMatch | 中等偏高 | rotation regression，与 QG 角度学习补充相关 |
| Pattern Recognition 工业缺陷检测 | 低到中 | 工业视觉写法参考，不直接支撑 QG |
| Object detection survey | 低到中 | Related Work 组织参考 |

## 6. 本次修订决策

1. `qg_obb_head_design.md` 不再把 Biternion / Complex-YOLO / CSL 作为主出处。
2. QG-OBB 主出处改为 GauCho、RSAR、InlierQ。
3. 对比对象改为官方标准 YOLOv8-OBB。
4. 当前 RV1106-M2 只作为工程落地 baseline。
5. HACMatch 作为补充相关文献记录，不进入主贡献出处表。
