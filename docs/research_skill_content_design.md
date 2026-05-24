# 科研项目 Skill 内容设计稿

> 当前阶段只补全 skill 内容，不创建真正的 skill 目录。下一阶段再根据本文档生成 `SKILL.md`、`references/`、`scripts/` 等文件。

检索日期：2026-05-24  
默认项目场景：计算机视觉 / 目标检测 / OBB / 轻量化部署 / AutoDL 训练平台

## 1. GitHub 调研对象与可借鉴点

GitHub star 数会随时间变化，下表记录的是本次调研时页面显示或检索到的量级，用于筛选优先级，不作为长期固定事实。

| 项目 | Star 量级 | 类型 | 可借鉴内容 |
|---|---:|---|---|
| [anthropics/skills](https://github.com/anthropics/skills) | 140k+ | Skill 组织范式 | skill 应以 `SKILL.md` 为入口，详细流程拆到 `references/`，可重复脚本放到 `scripts/`，避免把所有内容塞进主文件 |
| [bytedance/deer-flow](https://github.com/bytedance/deer-flow) | 69k+ | 研究型 agent / deep research | 借鉴其公开 skills 中的系统综述、论文评审、报告生成流程：先定义研究问题，再检索、筛选、抽取证据、综合结论 |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | 26k+ | ML 实验管理 | 借鉴实验 tracking 思想：每次 run 记录参数、指标、artifact、模型、环境，最终形成可比较实验表 |
| [iterative/dvc](https://github.com/iterative/dvc) | 15k+ | 数据/模型版本管理 | 借鉴数据集、权重、实验 artifact 的版本化思路，尤其适合记录数据版本、导出模型、RKNN 文件和评估日志 |
| [optuna/optuna](https://github.com/optuna/optuna) | 14k+ | 超参搜索 | 借鉴 trial 记录方式：每组实验必须明确变量、固定项、评价指标和终止条件 |
| [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | 13k+ | 自动科研 agent | 借鉴 idea -> experiment -> result -> paper 的闭环，尤其是“先有假设，再做实验，再写结论”的顺序 |
| [Future-House/paper-qa](https://github.com/Future-House/paper-qa) | 8k+ | 论文问答 / 科学文献 RAG | 借鉴 citation-grounded answer：所有创新点、论文结论和实验设置都必须能回指到论文或项目记录 |
| [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) | 6k+ | 自动科研 agent v2 | 借鉴更细的研究循环：假设、实现、实验、分析、稿件整理分离 |
| [checkpoints-lab/claude-scholar](https://github.com/checkpoints-lab/claude-scholar) | 3k+ | 科研辅助 agent | 借鉴面向真实科研项目的目录化知识库和多阶段工作流 |

本 skill 不应复制上述项目的复杂 agent 架构，而应吸收它们的共性：

- `SKILL.md` 保持短小，只放触发条件、核心原则、流程入口。
- 详细模板放入 `references/`。
- 可重复、易错的检查动作放入 `scripts/`。
- 所有科研结论都必须能追溯到论文、代码、实验日志或评估表。
- 每个阶段结束后必须产生“阶段记录”，否则后续论文写作会失去证据链。

## 2. Skill 目标

建议 skill 名称暂定为：

```text
research-project-planner
```

核心目标：

为机器学习/计算机视觉科研项目提供从创新点准备、论文调研、实验设计、AutoDL 实验执行、实验留痕、结果分析到论文论证的完整工作流。

适用任务：

- 用户需要为一个科研项目寻找创新点。
- 用户需要筛选近三年顶会/高水平论文。
- 用户需要把论文创新点转化为可实现实验方案。
- 用户需要设计 baseline、ablation、deployment evaluation。
- 用户在 AutoDL 上训练、导出、评估模型，需要记录实验链路。
- 用户需要把阶段性结果整理成论文表格、报告或答辩材料。

不适用任务：

- 单纯代码 bug 修复。
- 单纯论文翻译，不涉及项目创新点。
- 非科研型产品开发。
- 无需证据链的临时试验。

## 3. 核心原则

1. **先证据，后结论**
   不允许先写“提升精度/提升稳定性”，再倒找证据。必须先记录论文依据、实验设计、实际结果，再确定能主张什么。

2. **近三年顶会优先**
   创新点准备默认优先检索近三年相关领域顶会论文。计算机视觉优先关注 CVPR、ICCV、ECCV、NeurIPS、ICLR、ICML、AAAI、IJCAI、ACM MM。若项目偏部署/量化，也可补充 MLSys、DAC、DATE、ASPLOS、RTAS、TPAMI、TIP、TNNLS 等。

3. **官方 baseline 优先**
   新方案的主要对比对象应优先选择官方标准模型或公认 baseline。历史魔改模型只能作为工程 baseline 或消融项，不能替代主对比对象。

4. **每阶段必须留痕**
   每个阶段至少产生一个阶段记录文档，包含目标、假设、代码改动、命令、环境、结果、失败记录、下一步决策。

5. **实验变量单一化**
   每组实验必须明确唯一主要变量。若同时改 head、neck、loss、augmentation，就不能把收益归因到某一个创新点。

6. **AutoDL 是默认实验平台**
   默认记录 AutoDL 镜像、GPU、conda 环境、数据路径、代码路径、训练命令、导出命令、日志路径和权重路径。

7. **部署链路单独评估**
   对部署导向项目，PyTorch 指标不是终点。必须补充 ONNX、RKNN/目标平台、INT8 掉点、速度、模型大小等指标。

8. **失败实验也要记录**
   失败实验是论文消融和方法修正的重要依据。不能只保留成功结果。

## 4. 推荐 Skill 文件结构

下一阶段真正生成 skill 时，建议结构如下：

```text
research-project-planner/
├── SKILL.md
├── references/
│   ├── paper_discovery.md
│   ├── paper_reading_template.md
│   ├── innovation_mapping.md
│   ├── experiment_planning.md
│   ├── experiment_record_template.md
│   ├── autodl_runbook.md
│   ├── result_analysis.md
│   └── paper_claims.md
└── scripts/
    ├── init_stage_record.py
    ├── collect_run_summary.py
    └── update_experiment_table.py
```

`SKILL.md` 只保留入口流程和何时读取哪个 reference。详细模板放在 `references/`。

## 5. 总体工作流

### Phase 0：项目接入与资料扫描

目标：先理解项目，不急着设计创新点。

必须完成：

- 扫描 `docs/` 中已有计划、实验记录、阶段总结。
- 扫描 `configs/`、`scripts/`、核心模型代码，确定当前真实实现。
- 读取用户提供的论文目录、record 文件、实验日志。
- 明确项目任务：检测、分割、分类、部署、量化、轻量化或其他。
- 明确目标平台：默认 AutoDL 训练，若有部署则记录目标硬件。

输出：

```text
docs/project_context_summary.md
```

最低内容：

- 项目目标
- 当前 baseline
- 数据集情况
- 已完成实验
- 当前风险
- 下一阶段候选方向

### Phase 1：创新点准备与论文发现

目标：为创新点寻找近三年高质量出处。

检索策略：

1. 明确关键词组合：
   - 任务词：oriented object detection、rotated object detection、OBB、industrial symbol detection
   - 方法词：angle representation、Gaussian bounding box、quantization、lightweight detection、feature pyramid
   - 部署词：edge deployment、INT8、RKNN、NPU、post-training quantization

2. 优先检索：
   - 顶会官网论文列表
   - arXiv
   - OpenReview
   - Google Scholar / Semantic Scholar
   - Papers with Code
   - GitHub 官方代码仓库

3. 筛选条件：
   - 近三年优先。
   - 领域强相关优先。
   - 有代码或实验设置清晰优先。
   - 能映射到当前项目某个模块优先。
   - 不能只因为标题新就选入。

每篇候选论文需要记录：

| 字段 | 内容 |
|---|---|
| 论文标题 | 完整标题 |
| 年份/会议 | 如 CVPR 2025 |
| 链接 | arXiv / OpenReview / CVF / DOI |
| 代码链接 | GitHub 或官方项目页 |
| 核心问题 | 解决什么问题 |
| 核心创新 | 方法的本质变化 |
| 与项目关系 | 强相关/中相关/背景相关/不相关 |
| 可落地模块 | head / neck / loss / augmentation / quantization / deployment |
| 预期收益 | 精度、稳定性、速度、量化鲁棒性、可解释性 |
| 风险 | 算子不支持、训练不稳定、复现成本高、数据不匹配 |

输出：

```text
docs/paper_reference_relevance_audit.md
```

### Phase 2：单篇论文深读

目标：为每篇选中论文生成可复用中文解读。

每篇论文应生成独立文档：

```text
docs/paper_<short_name>_cn.md
```

推荐结构：

1. 基本信息
   - 标题
   - 作者
   - 年份/会议
   - 论文链接
   - 代码链接
   - 相关性等级

2. 论文要解决的问题
   - 原方法的问题是什么
   - 为什么这个问题重要
   - 在本项目中是否存在相同问题

3. 核心创新点
   - 用项目语言解释创新点
   - 关键模块结构
   - 数学表达或伪代码
   - 与已有方法的差异

4. 方法原理详细解释
   - 输入输出
   - 网络结构或算法流程
   - loss / assignment / postprocess / calibration
   - 为什么理论上有效

5. 实验设置
   - 数据集
   - baseline
   - 指标
   - 训练设置
   - 消融设置
   - 部署或速度设置

6. 论文结果
   - 主表结果
   - 消融结果
   - 失败或限制
   - 作者声称的收益

7. 对本项目的启发
   - 可以借鉴什么
   - 不适合照搬什么
   - 最小可实现版本
   - 需要新增哪些实验

8. 结论与引用方式
   - 作为主出处、补充出处还是背景出处
   - 论文中可以如何表述

### Phase 3：创新点映射与方案设计

目标：把论文创新点转化为本项目可实现方案。

需要输出一张创新点映射表：

| 创新模块 | 主参考论文 | 项目问题 | 新方案 | 对比对象 | 预期收益 | 验证指标 |
|---|---|---|---|---|---|---|
| QG-OBB Head | GauCho / RSAR / InlierQ | angle 边界与 INT8 稳定性 | unit-cycle angle branch | 官方 YOLOv8-OBB | 角度稳定、INT8 掉点小 | mAP、RKNN drop、angle error |
| SOF-FPN | 待定 | 小目标/方向纹理 | 方向保持 neck | 官方 neck / RV1106-M2 neck | 特征表达增强 | per-class AP、小目标 AP |
| GIS-Aug | 待定 | 工业符号扰动 | 几何/灰度增强 | 默认增强 | 泛化增强 | val/test AP、鲁棒性集 |

设计方案文档建议结构：

```text
docs/<module>_design.md
```

必须包括：

- 为什么做这个模块
- 论文出处
- 与官方 baseline 的差异
- 与当前工程 baseline 的差异
- 最小实现方案
- 完整实现方案
- 预计代码改动
- 风险和回退方案
- 实验计划

### Phase 4：实验计划制定

目标：每个阶段先写计划，再跑实验。

每阶段实验计划必须包含：

| 字段 | 要求 |
|---|---|
| 阶段目标 | 一句话说明本阶段验证什么 |
| 假设 | 例如“QG angle branch 会减小 ONNX 到 RKNN 掉点” |
| 主变量 | 本阶段唯一主要变化 |
| 固定项 | 数据、imgsz、epoch、batch、增强、优化器等 |
| 对比对象 | 官方 baseline、工程 baseline、旧方案 |
| 评价指标 | mAP50、mAP50-95、per-class AP、INT8 drop、latency |
| 成功门槛 | 进入下一阶段的条件 |
| 失败处理 | 回退、降级、改参数或淘汰 |
| 预计产物 | 权重、日志、表格、文档 |

实验命名规则：

```text
<platform>_<module>_<variant>_e<epochs>_b<batch>_<tag>
```

示例：

```text
rv1106_qg_sincos_e100_b512_selected
rv1106_m2_e20_b512_compare
official_yolov8n_obb_e100_b512
```

### Phase 5：实验执行与留痕

每次实验开始前必须创建阶段记录：

```text
docs/experiments/<YYYYMMDD>_<stage>_<run_name>.md
```

阶段记录模板：

```markdown
# 实验记录：<run_name>

## 1. 目标

## 2. 环境

- 平台：
- GPU：
- Python：
- Conda 环境：
- 代码 commit / diff：
- 数据集路径：

## 3. 实验变量

- 主变量：
- 固定项：
- 对比对象：

## 4. 命令

## 5. 输出路径

- 权重：
- 日志：
- TensorBoard：
- ONNX：
- RKNN：

## 6. 结果

| 指标 | 数值 |
|---|---:|

## 7. 失败与异常

## 8. 结论

## 9. 下一步
```

必须记录失败：

- 报错堆栈
- 卡死位置
- 版本冲突
- 性能异常
- 指标崩溃
- 采取过的修复方案

### Phase 6：AutoDL 默认平台规范

AutoDL 上每次实验需要记录：

| 项 | 示例 |
|---|---|
| 工作目录 | `/root/ultralytics_yolov8-main/ultralytics_yolov8-main` |
| 数据目录 | `/root/autodl-tmp/yolo_dataset_gray` |
| 训练环境 | `base` 或项目训练环境 |
| RKNN 环境 | `rknn232` |
| GPU | 如 RTX 5090 |
| Python / Torch | 训练前打印 |
| 输出目录 | `runs/obb/<run_name>` |
| 日志目录 | `rknn_logs/`、`artifacts/local/` |

推荐每次训练前运行：

```bash
pwd
python - <<'PY'
import torch, sys
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

部署链路默认顺序：

```text
PyTorch val -> ONNX export -> ONNX val -> RKNN build -> RKNN simulator/full eval
```

RKNN 环境必须记录：

```bash
python - <<'PY'
import numpy, onnx, cv2, importlib.metadata as md
from rknn.api import RKNN
print("numpy", numpy.__version__)
print("onnx", onnx.__version__)
print("cv2", cv2.__version__)
print("rknn-toolkit2", md.version("rknn-toolkit2"))
rknn = RKNN(verbose=False)
print("RKNN init ok")
rknn.release()
PY
```

### Phase 7：结果分析

不要只看最终 mAP，需要拆解：

- PyTorch 与 ONNX 的差异
- ONNX 与 RKNN INT8 的差异
- per-class AP
- recall 是否下降
- mAP50 与 mAP50-95 是否方向一致
- 速度与模型大小
- 部署平台是否卡住或算子不兼容

结果表建议：

| 模型 | PyTorch mAP50 | PyTorch mAP50-95 | ONNX mAP50 | ONNX mAP50-95 | RKNN mAP50 | INT8 drop | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|

消融表建议：

| 模块 | Variant | 主变量 | mAP50 | mAP50-95 | RKNN mAP50 | 结论 |
|---|---|---|---:|---:|---:|---|

### Phase 8：论文主张边界

根据证据等级写结论：

| 证据 | 可以主张 | 不能主张 |
|---|---|---|
| 只有 PyTorch 提升 | 提升训练/FP32 精度 | 不能说部署稳定 |
| ONNX/RKNN 掉点更小 | 改善量化部署稳定性 | 不能说所有硬件都有效 |
| per-class 个别提升 | 对某些类别有效 | 不能说整体显著提升 |
| e20 快速实验 | 早期趋势 | 不能作为最终论文结果 |
| e100 + 多链路验证 | 正式实验结果 | 仍需说明数据集和平台限制 |

每个创新点最终需要归纳为：

```text
我们提出 X，用于解决 Y。该设计受 A/B/C 论文启发。
实验显示，在 Z 设置下，X 相比 baseline 在指标 M 上取得 N 的结果；
同时在部署链路中表现为 P，因此我们将其表述为 Q。
```

## 6. 计划中的 references 内容

下一阶段生成 skill 时，建议把以下内容拆成 reference 文件：

### `references/paper_discovery.md`

包含：

- 顶会和期刊优先级
- 近三年检索规则
- 关键词组合模板
- 论文筛选表模板
- GitHub 代码仓库筛选标准

### `references/paper_reading_template.md`

包含：

- 单篇论文中文解读模板
- 方法原理拆解模板
- 实验设置记录模板
- 与项目相关性判断标准

### `references/innovation_mapping.md`

包含：

- 创新点到论文出处映射
- 创新点到代码模块映射
- 可落地性评分
- 风险评分

### `references/experiment_planning.md`

包含：

- baseline 选择规则
- ablation 设计规则
- 变量控制规则
- 成功/失败门槛

### `references/experiment_record_template.md`

包含：

- 阶段记录模板
- run 记录模板
- failure log 模板
- 结果表模板

### `references/autodl_runbook.md`

包含：

- AutoDL 目录规范
- conda 环境规范
- 训练命令规范
- 导出/评估/RKNN 命令规范
- 常见故障记录格式

### `references/result_analysis.md`

包含：

- PyTorch/ONNX/RKNN 对比表
- per-class AP 分析
- INT8 drop 分析
- 速度和模型大小分析

### `references/paper_claims.md`

包含：

- 论文主张边界
- Related Work 写法
- Method 写法
- Experiment 写法
- Limitation 写法

## 7. 计划中的 scripts 内容

后续可以考虑提供脚本，但不是第一版必须项。

### `scripts/init_stage_record.py`

作用：

- 根据 run name 创建阶段记录 md。
- 自动填入日期、目录、模板标题。

### `scripts/collect_run_summary.py`

作用：

- 从 `runs/obb/<run>/results.csv`、评估日志、RKNN 日志中抽取关键指标。
- 输出 markdown 表格行。

### `scripts/update_experiment_table.py`

作用：

- 把抽取到的指标追加到统一实验表。
- 避免手动复制数值出错。

第一版 skill 可以先不包含脚本，只保留 references 模板；等项目实验记录稳定后再脚本化。

## 8. 本项目当前应纳入 skill 的特化规则

结合当前 Edge-symbol-obb 项目，skill 应内置以下偏好：

1. OBB 项目必须检查 angle branch、loss、ONNX 输出、RKNN 后处理四个接口。
2. 新 head 方案必须先 smoke test，再 e20，再 e100。
3. 对比对象必须区分：
   - 官方标准 YOLOv8-OBB
   - 当前 RV1106-M2 工程 baseline
   - 新增模块版本
4. 部署实验统一记录：
   - ONNX mAP
   - RKNN mAP
   - ONNX 到 RKNN drop
   - simulator latency
   - 目标硬件 latency 如果有
5. RKNN 环境版本必须写入阶段记录。
6. 论文出处必须分级：
   - 主出处
   - 补充出处
   - 背景出处
   - 不相关
7. 不允许把“借鉴背景论文”写成“直接创新出处”。

## 9. 下一阶段生成 Skill 时的 SKILL.md 草案要点

真正生成 `SKILL.md` 时，主体可以压缩为：

1. 什么时候使用本 skill。
2. 工作流总览：项目扫描 -> 论文调研 -> 论文解读 -> 创新映射 -> 实验计划 -> AutoDL 执行 -> 结果分析 -> 论文主张。
3. 每一步需要读取哪个 reference。
4. 强制规则：
   - 先看 docs 和现有记录。
   - 近三年顶会优先。
   - 每个创新点必须有论文出处。
   - 每阶段必须留痕。
   - AutoDL 默认平台。
   - baseline 不能随意选。
5. 输出产物规则：
   - 不只给建议，要生成对应文档或表格。
   - 不确定的结论必须标注为待验证。

`SKILL.md` 不应放入所有模板全文，否则会过长；模板全文应放入 `references/`。

## 10. 当前结论

本 skill 应定位为“科研项目全过程管理 skill”，而不是单纯“论文阅读 skill”。

它需要覆盖：

- 创新点准备
- 近三年顶会论文筛选
- 论文中文解读
- 创新点与论文相关性审计
- 实验计划制定
- 阶段留痕
- AutoDL 平台执行规范
- 结果分析
- 论文主张边界

下一阶段可以基于本文档生成正式 skill 目录。
