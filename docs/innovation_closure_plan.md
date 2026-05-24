# 创新点闭环计划

## 1. 文档目的

本文档将当前研究方向转化为每个计划创新点的闭环检查表。

下一阶段目标不是获得最终论文指标，而是先证明每个计划创新点都能跑通完整工程链路：

```text
论文来源 -> 代码实现 -> PyTorch 短训练 -> ONNX 导出
-> ONNX 验证 -> RKNN 转换 -> RKNN 推理 -> 结果记录
```

只有通过该链路的创新点，才进入 full-val、hard-val 和 external-val 的正式实验。

## 2. 当前创新点总表

| ID | 创新点 | 主要参考论文 | 模块位置 | 预期价值 | 当前状态 | 下一关口 |
|---|---|---|---|---|---|---|
| I1 | QG-OBB 单位圆角度分支 | GauCho、RSAR、InlierQ、Reg-PTQ | OBB head、loss、ONNX/RKNN 解码 | 量化友好的角度表示 | 已实现并完成全链路测试 | 作为模板冻结，后续在 hard-val 上复测 |
| I2 | SOF-FPN 符号导向特征增强 | SET、YOLO-RD、YOLOv12、TOE-YOLO | Neck/FPN、轻量特征融合 | 提升 QR/BARCODE/DM 困难样本特征提取 | 未实现 | 设计 RV1106 友好的最小模块 |
| I3 | RV1106-aware 量化与部署协同 | InlierQ、Reg-PTQ、RF-DETR、efficient TTA detection | 量化校准、模型选择、输出分支稳定性 | 降低 ONNX 到 RKNN 的 INT8 掉点 | 已通过 QG/RKNN 做了部分探索 | 定义校准集和分支敏感性实验 |
| I4 | GIS-Aug / 困难样本策略 | AeroGen、Object Fidelity Diffusion、HACMatch、frequency-bias papers | 数据集构建、鲁棒性测试 | 构建更有效的 QR/BARCODE/DM 困难评估 | 未实现 | 先构建 hard-val，再考虑训练增强 |

## 3. 通用闭环检查表

每个创新点在正式训练前必须通过以下检查：

| 步骤 | 必要证据 | 通过标准 |
|---|---|---|
| 论文依据 | 论文笔记或相关性审计 | 能清楚说明论文思想与本地模块的关系 |
| 最小设计 | 设计文档或本文档中的章节 | 范围足够小，能用 short run 验证 |
| 配置 | YAML 路径或脚本参数 | 模型/数据配置可复现 |
| 构建 | 模型 summary | 参数量/FLOPs 已记录，无 shape error |
| 短训练 | 1 到 20 epoch run | 无 NaN，val 能跑完，记录 run 路径 |
| ONNX 导出 | ONNX 文件和输出 shape | 导出成功，输出名称和 shape 已记录 |
| ONNX 验证 | mAP 或小规模结果 | 解码链路可用 |
| RKNN 构建 | RKNN 文件和 toolkit log | 转换成功，或失败原因明确 |
| RKNN 验证 | AP@0.5 和延迟 | 推理和后处理能完成 |
| 结论边界 | 保守结论说明 | 不做无证据的精度提升声明 |

## 4. I1：QG-OBB 单位圆角度分支

### 4.1 当前状态

QG-OBB 是第一个已完成闭环的创新点模板。

已实现产物：

- `configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml`
- `ultralytics/nn/modules/head.py`
- `ultralytics/utils/loss.py`
- `scripts/export/export_gray_obb_onnx.py`
- `scripts/eval/val_onnx_gray.py`
- `scripts/deploy_rv1106/convert_eval_rknn.py`

主要结果：

| 模型 | ONNX mAP50 | ONNX mAP50-95 | RKNN mAP50 | ONNX->RKNN 掉点 |
|---|---:|---:|---:|---:|
| RV1106-M2 scalar e100 | 0.9905 | 0.9596 | 0.9843 | 0.0062 |
| QG-OBB e100 | 0.9895 | 0.9577 | 0.9854 | 0.0041 |

当前可成立结论：

```text
在当前较容易验证集上，QG-OBB 不是已经证明的 FP32 精度提升模块。
它更适合作为量化友好的单位圆角度分支，能略微改善 RKNN INT8 精度保持率。
```

### 4.2 剩余工作

- hard-val 构建完成后重新评估。
- 可行时补充角度边界误差分析。
- 补充校准图片数量消融：100、500、1000。

## 5. I2：SOF-FPN 符号导向特征增强

### 5.1 假设

QR、BARCODE、DM 是纹理和结构主导的符号。轻量级特征增强模块可能改善低对比、模糊、小目标和复杂背景下的识别。

### 5.2 设计边界

第一版必须 RV1106 友好：

- 避免不支持或昂贵算子。
- 优先使用 Conv、Depthwise Conv、Pointwise Conv、简单 pooling、add/concat。
- 不引入 Transformer attention，除非已经证明 ONNX/RKNN 安全。
- 参数增量要足够小，便于和 RV1106-M2 公平比较。

### 5.3 最小闭环计划

| 步骤 | 操作 |
|---|---|
| 设计 | 定义一个轻量增强块，用于 P3/P4/P5 或仅 P3/P4 |
| 配置 | 创建 `configs/rv1106/yolov8n_obb_rv1106_soffpn.yaml` 或等价配置 |
| smoke | 当前数据集上训练 1 epoch |
| 导出 | ONNX 输出必须继续兼容现有解码脚本 |
| RKNN | 至少完成 build-only，再做一张图片 debug 推理 |
| 决策 | 只有完整链路可跑通且参数/FLOPs 合理时保留 |

### 5.4 晋级标准

闭环通过标准：

- train/val 无 shape error。
- ONNX 导出成功。
- RKNN build 成功。

进入正式实验标准：

- 在 hard-val 上提升 QR/BARCODE/DM，或在 RKNN 延迟代价较小的前提下提升召回率。

## 6. I3：RV1106-aware 量化与校准

### 6.1 假设

当前最强证据并不是 FP32 精度提升，而是 RKNN INT8 保持率。因此需要单独建立量化友好实验线。

### 6.2 最小实验变量

| 变量 | 取值 |
|---|---|
| 校准图片数量 | 100、500、1000 |
| 校准子集 | random、QR/BARCODE/DM-heavy、hard-case-heavy |
| 模型 | scalar RV1106-M2、QG-OBB、后续 SOF-FPN 变体 |
| 指标 | ONNX mAP50、RKNN mAP50、ONNX->RKNN 掉点、延迟 |

### 6.3 闭环计划

| 步骤 | 操作 |
|---|---|
| 脚本支持 | 确认 RKNN 转换可接受固定 calibration list |
| dataset txt | 保存并版本化每个 calibration txt |
| 构建 | 使用同一校准策略转换 scalar 和 QG |
| 评估 | 先 full-val，后 hard-val |
| 分析 | 比较掉点和 per-class AP 变化 |

### 6.4 成功标准

如果能找到一种校准策略，在不增加模型代价的前提下降低 ONNX-to-RKNN 掉点，或稳定 QR/BARCODE/DM AP，则该方向成立。

## 7. I4：GIS-Aug 与困难样本策略

### 7.1 正确定位

该方向一开始不应直接作为训练增强主张，而应先服务于评估：

1. 从自然样本或现有样本中构建 hard-val。
2. 合成退化样本只作为单独 robustness stress test。
3. 固定评估集后，再考虑用增强方法参与训练。

### 7.2 闭环计划

| 步骤 | 操作 |
|---|---|
| 数据集审计 | 找出 QR/BARCODE/DM 困难样本 |
| hard-val 设计 | 明确选择规则和目标数量 |
| synthetic stress | 可选生成退化样本集，并明确标记为合成 |
| 训练增强 | 仅在评估集固定后尝试 |

### 7.3 成功标准

GIS-Aug 只有在以下条件满足时，才可作为正式创新点：

- 生成或选择的样本在视觉上有意义。
- 标签几何关系仍然有效。
- 训练收益出现在自然 hard-val 或 external-val 上，而不是只出现在 synthetic-val 上。

## 8. 执行顺序

后续工作应按如下顺序推进：

1. 完成数据集审计方案和 hard-val 规则。
2. 建立脚本或流程，生成数据集审计报告。
3. 先构建 hard-val，再继续正式训练实验。
4. 实现 SOF-FPN 最小版本，只做 smoke 和全链路闭环。
5. 基于 scalar 和 QG 实施量化校准实验。
6. hard-val 定义稳定后，搜索外部数据集。
7. 上述步骤完成后，再恢复正式 100 epoch 实验。

## 9. 当前立即任务

| 优先级 | 任务 | 输出 |
|---:|---|---|
| P0 | 建立数据集审计方案 | `docs/dataset_audit_plan.md` |
| P0 | 在 AutoDL 上运行当前数据集审计 | `docs/dataset_audit_current.md` |
| P0 | 定义 hard-val 选择规则 | 写入数据集审计方案 |
| P1 | 创建 SOF-FPN 最小设计 | `docs/soffpn_design.md` |
| P1 | 创建量化校准计划 | `docs/rknn_calibration_plan.md` |
| P2 | 搜索外部数据集 | `docs/external_dataset_search.md` |
