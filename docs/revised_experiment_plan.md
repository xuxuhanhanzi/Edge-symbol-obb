# 修正版实验方案

## 1. 方向修正

原实验方案过度强调在当前验证集上继续做更多 full-epoch 对比。根据目前结果，这已经不是最优先方向。

当前证据表明，现有验证集已经接近饱和：

- RV1106-M2 scalar e100 约为 `mAP50=0.991`。
- QG-OBB e100 约为 `mAP50=0.990`。
- Official-style YOLOv8n-OBB gray e20 已达到约 `mAP50=0.977`、`mAP50-95=0.924`。
- 多个类别已经接近满分，小幅指标差异不足以强力证明创新点有效。

因此，后续方向修正为：

1. 先完成所有计划创新点，并证明每个创新点都能跑通完整部署链路。
2. 再建立更有区分度的评估协议，包括困难子集和外部数据集。
3. 最后再启动正式、大规模对比实验。

## 2. 修正后的研究目标

本项目不应主张“当前创新点能在简单验证集上显著提升普通 FP32 精度”。

更稳妥、更有论文价值的目标是：

```text
在边缘设备约束下，提高灰度工业符号旋转检测的鲁棒性和部署可靠性，
尤其关注 ONNX 导出和 RV1106 INT8 RKNN 量化后的稳定性。
```

正式结论应优先围绕以下方面建立：

- 全链路可部署性：`PyTorch -> ONNX -> RKNN -> 后处理 -> 评估`。
- INT8 保持率：ONNX mAP50 到 RKNN mAP50 的掉点。
- 困难样本表现：QR、BARCODE、DM 在旋转、模糊、低对比、透视、小目标、复杂背景、部分裁切下的表现。
- 边缘代价：模型大小、参数量、FLOPs、RKNN simulator 延迟，以及后续真实设备延迟。

## 3. 停止、保留、推迟

| 事项 | 决策 | 原因 |
|---|---|---|
| 立刻继续 official YOLOv8n-OBB gray e100 | 推迟 | 当前数据集已接近饱和，e100 结果后续有用，但不是当前优先事项 |
| 保留 official YOLOv8n-OBB gray e20 | 保留为 sanity baseline | 已证明官方风格 baseline 可以训练，并验证灰度链路修复有效 |
| 继续 QG 全链路验证 | 保留 | QG 已有 PyTorch/ONNX/RKNN 证据，可作为其他创新点闭环模板 |
| 现在继续堆正式结果表 | 停止 | 会过度依赖容易验证集，结论证明力弱 |
| 构建 hard validation set | 立即开始 | 这是判断创新点真实效果的关键 |
| 搜索外部数据集 | 在数据审计方案稳定后开始 | 用于补充泛化证据 |

## 4. 阶段 A：创新点全链路闭环

目的：在正式实验前，先完成每个计划创新点的工程闭环。

每个创新点都使用同一套闭环检查表：

| 检查项 | 必要产物 |
|---|---|
| 论文来源与原理 | `docs/paper_<name>_cn.md` 或创新点映射记录 |
| 代码实现 | 修改模块路径和配置路径 |
| 构建检查 | 模型 summary、参数量、FLOPs |
| 短训练 | 1 到 20 epoch smoke run |
| ONNX 导出 | ONNX 文件、输出名称和 shape |
| ONNX 验证 | 全量或小规模验证结果 |
| RKNN 转换 | RKNN 文件、toolkit 版本、build log |
| RKNN 评估 | 小规模 debug，再全量验证 |
| 失败记录 | 错误日志和诊断 |

通过标准：

- 该创新点暂时不要求超过 baseline。
- 必须能训练、导出、转换、推理和解码。
- 如果无法通过 RKNN 转换，应标记为 research-only，而不是 deployment-ready。

当前 QG 状态：

- QG sin/cos head 已基本闭环。
- 后续创新点应参考 QG 的闭环方式执行。

## 5. 阶段 B：现有数据集审计

构建新数据前，必须先审计当前数据集。

必要分析包括：

- 每类图片数和实例数。
- PyTorch、ONNX、RKNN 下的 per-class AP。
- 目标面积分布。
- 长宽比分布。
- 角度分布。
- train/val 是否有重复或近重复风险。
- QR、BARCODE、DM 的失败案例。

类别分组：

| 分组 | 类别 | 作用 |
|---|---|---|
| 主要困难类别 | QR、BARCODE、DM | 作为创新点效果判断的主要依据 |
| 监控类别 | PDF、MPDF、MQR、RMQR | 用于回归检查 |
| 饱和类别 | AP 接近满分的类别 | 保留在 full-val 中，但不作为主要创新证明 |

输出产物：

- `docs/dataset_audit_current.md`
- `docs/experiments/<date>_dataset_audit.md`
- 如果生成失败案例图，则保存到 `artifacts/local/` 下。

## 6. 阶段 C：困难验证集 hard-val

先从现有数据中构建一个固定的困难验证集。

建议名称：

```text
datasets/industrial_symbol_hard.yaml
```

建议组成：

| 来源 | 目标数量 |
|---|---:|
| QR 困难样本 | 200-400 张 |
| BARCODE 困难样本 | 200-400 张 |
| DM 困难样本 | 200-400 张 |
| 其他回归类别 | 200-500 张 |

困难样本标准：

- 低对比度。
- 小目标。
- 大角度旋转。
- 透视变形。
- 运动模糊或失焦模糊。
- 部分裁切或贴近边界。
- 背景纹理干扰。
- 类间相似样本。

重要规则：

- 不要把训练图像的增强副本作为主要 hard-val 证据。
- 合成退化样本可以作为单独的鲁棒性测试集，但必须标记为 synthetic robustness，不能等同于自然泛化。

评估指标：

- mAP50 和 mAP50-95。
- QR、BARCODE、DM 的 per-class AP。
- 固定置信度阈值下的召回率。
- ONNX 到 RKNN 的 mAP50 掉点。
- 错误案例：误检、漏检、角度错误、定位错误。

## 7. 阶段 D：外部数据集搜索与适配

目的：测试当前方法是否只适配自建数据集。

搜索优先级：

1. Barcode / QR / DataMatrix 检测数据集。
2. 工业包装、标签、符号数据集。
3. 带 polygon 或 rotated bbox 标注的数据集。
4. 只有普通 bbox 的数据集仅在目标接近矩形、方向可恢复时作为辅助。
5. 通用 OBB 数据集只作为补充测试，不作为符号领域主要证据。

数据集接收标准：

- 可公开访问，允许科研使用。
- 标注格式清晰。
- 至少具备验证规模，最好有 500 个以上目标实例。
- 能映射到 QR、BARCODE、DM，或有清晰的子集映射规则。
- 转换过程可复现。

输出产物：

- `docs/external_dataset_search.md`
- 每个接受数据集对应一份转换说明。
- 格式检查通过后，再在 `datasets/` 下新增对应 yaml。

## 8. 阶段 E：正式实验

正式实验只在阶段 A-C 完成后启动。

评估划分：

| 数据集 | 作用 |
|---|---|
| `industrial_symbol.yaml` full val | 确认整体任务不退化 |
| `industrial_symbol_hard.yaml` hard val | 作为创新点主要证明 |
| external dataset yaml | 泛化能力检查 |

模型分组：

| 分组 | 模型 |
|---|---|
| 官方结构 baseline | Official YOLOv8n-OBB gray |
| 部署 baseline | RV1106-M2 scalar |
| 单创新点方法 | RV1106-M2 + 每个创新点 |
| 组合方法 | RV1106-M2 + 兼容创新点组合 |

每个正式模型记录：

- 训练命令和环境。
- 参数量、FLOPs、模型大小。
- PyTorch mAP50 / mAP50-95。
- ONNX mAP50 / mAP50-95。
- RKNN mAP50 和延迟。
- ONNX 到 RKNN 的掉点。
- QR、BARCODE、DM 的 per-class AP。
- hard-val 失败案例。

## 9. 阶段 F：消融实验

只对稳定创新点做消融。

QG 相关消融：

| 变量 | 取值 |
|---|---|
| 角度表示 | scalar vs sin/cos |
| QG alignment loss | 0、0.10、0.25 |
| QG unit loss | 0、0.02、0.05 |
| 量化校准图片数量 | 100、500、1000 |
| 评估集 | full-val、hard-val、external-val |

规则：

- 一次只改变一个主要变量。
- 先做 20 epoch 消融。
- 只有有希望的设置才晋级 100 epoch。

## 10. 论文结论边界

当前允许的结论：

- 方法兼容 RV1106 部署链路。
- 方法降低 ONNX 到 RKNN INT8 的精度掉点。
- 方法在 QR、BARCODE、DM 困难样本上提升鲁棒性。
- 方法保持 full-val 精度，同时改善部署稳定性。

当前不允许的结论：

- 达到通用 OBB 检测 SOTA。
- 对所有符号类别都有普遍提升。
- 在真实设备速度提升，除非完成真实硬件测试。
- 具备外部泛化能力，除非完成外部数据集验证。

## 11. 立即任务

1. 将当前结果冻结为阶段性证据，而不是最终论文证据。
2. 列出所有剩余创新点，并为每个创新点建立全链路闭环检查表。
3. 用 short run 实现或补齐剩余创新点。
4. 制定 `industrial_symbol_hard.yaml` 的设计和选择规则。
5. 审计当前数据集分布和失败案例。
6. 搜索并筛选外部数据集。
7. hard-val 和 external-val 准备好后，再恢复正式对比实验。
