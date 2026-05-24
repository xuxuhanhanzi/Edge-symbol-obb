# 数据集审计与 Hard-Val 构建计划

## 1. 文档目的

当前验证集对创新点证明力不足。多个类别已经接近 mAP50 满分，继续在同一验证集上做 full-epoch 实验，会得到很多数字，但难以证明创新点真实有效。

本阶段的数据工作路线是：

```text
审计当前数据集 -> 找出困难类别和困难样本
-> 构建固定 hard-val -> 搜索外部数据集 -> 重启正式实验
```

## 2. 当前数据集

数据集配置：

```text
datasets/industrial_symbol.yaml
```

AutoDL 上的数据集根路径：

```text
/root/autodl-tmp/yolo_dataset_gray
```

当前类别：

| ID | 类别 |
|---:|---|
| 0 | BARCODE |
| 1 | DM |
| 2 | HANXIN |
| 3 | QR |
| 4 | PDF |
| 5 | AZTEC |
| 6 | CODEONE |
| 7 | DOT |
| 8 | GM |
| 9 | MAXI |
| 10 | MPDF |
| 11 | MQR |
| 12 | RMQR |
| 13 | ULTRA |
| 14 | UPN |

hard-val 优先类别：

```text
QR, BARCODE, DM
```

原因：

- 这些类别实际应用价值高。
- 更容易暴露模糊、低对比、小目标、旋转和结构相似性问题。
- 当前 per-class 结果显示，它们相较某些接近满分的类别更能体现模型差异。

## 3. 必要审计输出

本阶段应产出：

| 输出 | 作用 |
|---|---|
| `docs/dataset_audit_current.md` | 当前数据集的人类可读审计报告 |
| `docs/experiments/<date>_dataset_audit.md` | 阶段记录，包含命令和结论 |
| `artifacts/local/dataset_audit/` | 可选保存图表和失败样本 |
| `datasets/industrial_symbol_hard.yaml` | 只有 hard-val 样本确定后才创建 |

注意：生成的图表和 hard-val 候选样本，在选择规则被记录前，不应视为最终数据集。

## 4. 审计维度

### 4.1 基础完整性

先使用已有标签检查脚本：

```bash
conda activate base
cd ~/ultralytics_yolov8-main/ultralytics_yolov8-main

python -B scripts/data/check_labels.py \
  --data datasets/industrial_symbol.yaml \
  --report docs/dataset_audit_current.md
```

记录内容：

- train/val 图片数。
- train/val 目标实例数。
- 缺失标签。
- 孤立标签。
- 错误标签行。
- 类别分布。
- 基础角度和长宽比分布。

### 4.2 几何分布

当前 `check_labels.py` 只能给出基础长宽比和角度摘要。下一步应扩展该脚本，或新增脚本统计：

| 指标 | 意义 |
|---|---|
| 目标面积占比 | 区分小/中/大目标 |
| 长宽比 | 对 BARCODE、RMQR 等细长目标重要 |
| 角度直方图 | 判断旋转多样性是否真实存在 |
| 边界接触 | 找出裁切、贴边困难样本 |
| 每类 split 数量 | 检查类别均衡 |
| 困难类别数量 | 确认 QR/BARCODE/DM 是否足够 |

建议分桶：

- 面积占比：`<0.2%`、`0.2-1%`、`1-5%`、`>5%`。
- 长宽比：`1-2`、`2-5`、`5-10`、`>10`。
- 角度：`0-15`、`15-45`、`45-75`、`75-105`、`105-135`、`135-165`、`165-180`。

### 4.3 基于模型表现的困难度

如果能保存逐图预测结果，应收集以下困难信号：

| 困难信号 | 含义 |
|---|---|
| 漏检 | 目标没有被检测到 |
| 低置信度真阳性 | 识别不稳 |
| IoU 接近阈值 | 定位不稳定 |
| 分类错误 | 类别混淆 |
| 角度误差大 | 角度分支问题 |
| 仅 RKNN 失败 | 量化或部署问题 |

优先用于挖掘困难样本的模型：

1. RV1106-M2 scalar e100。
2. QG-OBB e100。
3. Official YOLOv8n-OBB gray e20，如果后续完成 ONNX/RKNN 评估。

## 5. Hard-Val 定义

推荐配置路径：

```text
datasets/industrial_symbol_hard.yaml
```

推荐 AutoDL 文件布局：

```text
/root/autodl-tmp/yolo_dataset_gray_hard/
  val/images/
  val/labels/
```

建议组成：

| 来源 | 目标数量 |
|---|---:|
| QR 困难样本 | 200-400 张 |
| BARCODE 困难样本 | 200-400 张 |
| DM 困难样本 | 200-400 张 |
| 其他回归类别 | 200-500 张 |

hard-val 不需要匹配 full dataset 的类别分布。它应有意强调困难且实际重要的样本。

## 6. 困难样本选择规则

样本满足至少一项困难条件，即可进入 hard-val 候选：

| 条件 | 判断规则 |
|---|---|
| 小目标 | 目标面积占比低于设定阈值 |
| 低对比 | 由图像统计或人工复核判断前景/背景对比低 |
| 大角度旋转 | 角度远离简单主方向 |
| 极端长宽比 | 尤其是 BARCODE/RMQR |
| 贴边或裁切 | polygon 接近图像边界 |
| 模糊 | 人工复核或 blur metric 判断 |
| 透视变形 | 符号形状明显倾斜或扭曲 |
| 复杂背景 | 背景纹理会干扰符号结构 |
| 模型失败 | scalar/QG/official baseline 漏检或低置信度 |

允许人工复核，但必须记录：

```text
image path -> reason -> source split -> selector/date
```

## 7. 数据泄漏规则

hard-val 必须只用于评估。

规则：

- 不选择训练图像的完全重复样本。
- 不把训练图像的增强副本当作自然 hard-val。
- 如果创建合成退化样本，必须放到单独 synthetic stress set。
- 不在 hard-val 上反复调参，除非记录清楚。
- hard-val 一旦固定，不应持续修改以迎合某个方法。

建议额外拆分：

```text
datasets/industrial_symbol_synthetic_stress.yaml
```

该数据集只用于受控鲁棒性压力测试，不作为主要泛化证据。

## 8. 外部数据集搜索计划

当前数据集审计规则稳定后，再开始搜索外部数据集。

优先顺序：

1. QR code detection 数据集。
2. Barcode detection 数据集。
3. DataMatrix detection 数据集。
4. 工业包装、标签、符号数据集。
5. 带 polygon 或 rotated-bbox 标注的数据集。
6. 只有普通 bbox 的数据集仅作为辅助证据，前提是方向可可靠恢复。

接受标准：

| 标准 | 要求 |
|---|---|
| License | 允许科研使用 |
| Format | 标注格式清晰、可复现 |
| Domain | 优先符号、条码、工业标签场景 |
| Size | 至少具有验证规模 |
| Mapping | 可映射到 QR/BARCODE/DM 或清晰子集 |
| Conversion | 可无歧义转换为 YOLO OBB |

输出：

```text
docs/external_dataset_search.md
```

## 9. Hard-Val 后的评估协议

hard-val 准备好后，每个模型都应在以下数据集上评估：

| 数据集 | 作用 |
|---|---|
| full-val | 整体不退化检查 |
| hard-val | 创新点主要证明 |
| external-val | 泛化能力证明 |
| synthetic-stress，可选 | 受控鲁棒性压力测试 |

每个数据集记录：

- mAP50。
- mAP50-95。
- QR/BARCODE/DM 的 per-class AP。
- QR/BARCODE/DM 的召回率。
- ONNX 结果。
- RKNN 结果，如果支持。
- ONNX-to-RKNN 掉点。
- 代表性失败案例。

## 10. 立即执行命令

### 10.1 当前数据集完整性审计

在 AutoDL 上执行：

```bash
conda activate base
cd ~/ultralytics_yolov8-main/ultralytics_yolov8-main

python -B scripts/data/check_labels.py \
  --data datasets/industrial_symbol.yaml \
  --report docs/dataset_audit_current.md
```

### 10.2 几何难例分析脚本

完整性审计通过后，执行已经新增的只读分析脚本：

```bash
python -B scripts/data/analyze_obb_hard_cases.py \
  --data datasets/industrial_symbol.yaml \
  --split val \
  --out-dir artifacts/local/dataset_audit \
  --report docs/dataset_geometry_audit.md
```

预期输出：

- `docs/dataset_geometry_audit.md`
- `artifacts/local/dataset_audit/area_hist.csv`
- `artifacts/local/dataset_audit/aspect_ratio_hist.csv`
- `artifacts/local/dataset_audit/angle_hist.csv`
- `artifacts/local/dataset_audit/hard_candidates.csv`

该脚本不能移动或删除数据，只能读取图片/标签并写出报告。

困难候选的定义为：至少命中一个真实困难信号，包括小目标、极端长宽比、贴边或大角度旋转。`QR/BARCODE/DM` 作为优先类别参与排序，但不会单独构成困难样本。

如果需要同时挖掘 train 中的困难样本，单独输出 train 候选，不能直接混入正式 hard-val：

```bash
python -B scripts/data/analyze_obb_hard_cases.py \
  --data datasets/industrial_symbol.yaml \
  --split train \
  --out-dir artifacts/local/dataset_audit_train \
  --report docs/dataset_geometry_audit_train.md
```

### 10.3 固定 Hard-Val 数据集

正式 hard-val 默认只使用 `val/test` 来源样本，避免和已经训练过的模型产生训练集泄漏。

```bash
python -B scripts/data/build_hard_val_from_candidates.py \
  --candidates artifacts/local/dataset_audit/hard_candidates.csv \
  --data datasets/industrial_symbol.yaml \
  --out-root /root/autodl-tmp/yolo_dataset_gray_hard \
  --out-yaml datasets/industrial_symbol_hard.yaml \
  --manifest docs/hard_val_manifest.csv \
  --report docs/hard_val_build_report.md \
  --max-total 700
```

如果只是为了诊断或后续训练数据增强，可以显式加入 train 候选，输出到单独诊断数据集：

```bash
python -B scripts/data/build_hard_val_from_candidates.py \
  --candidates artifacts/local/dataset_audit/hard_candidates.csv artifacts/local/dataset_audit_train/hard_candidates.csv \
  --data datasets/industrial_symbol.yaml \
  --out-root /root/autodl-tmp/yolo_dataset_gray_hard_train_diagnostic \
  --out-yaml datasets/industrial_symbol_hard_train_diagnostic.yaml \
  --manifest docs/hard_val_train_diagnostic_manifest.csv \
  --report docs/hard_val_train_diagnostic_build_report.md \
  --max-total 1000 \
  --include-train
```

含 train 的输出只能用于诊断和样本复核。除非后续重新训练时明确排除这些 train 样本，否则不能作为正式泛化验证集。

## 11. 下一决策关口

生成 `docs/dataset_audit_current.md` 后，需要判断：

1. 现有 val split 中 QR/BARCODE/DM 是否有足够困难样本。
2. 是否能直接从现有数据中选出 hard-val。
3. 正式实验前是否需要额外采集或引入外部数据集。
4. 是否需要为模糊、低对比、透视等创建 synthetic stress test。

在通过该决策关口前，正式大规模实验继续暂停。
