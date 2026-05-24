# Paper Reading Template

Use this reference when creating `docs/paper_<short_name>_cn.md`.

## Template

```markdown
# <Paper Short Name> 中文解读

## 1. 基本信息

| 项 | 内容 |
|---|---|
| 标题 | |
| 年份/会议 | |
| 论文地址 | |
| 代码地址 | |
| 项目相关性 | 强相关 / 中等相关 / 背景相关 / 不相关 |
| 对应模块 | head / neck / loss / augmentation / quantization / deployment |

## 2. 论文要解决的问题

- 原方法的问题：
- 为什么重要：
- 本项目是否存在同类问题：

## 3. 核心创新点

- 创新点 1：
- 创新点 2：
- 创新点 3：

用项目语言解释该方法改变了什么。

## 4. 方法原理

### 4.1 输入输出

### 4.2 结构或算法流程

### 4.3 损失、分配、后处理或校准

### 4.4 为什么理论上有效

## 5. 实验设置

| 项 | 内容 |
|---|---|
| 数据集 | |
| Baseline | |
| 指标 | |
| 训练设置 | |
| 消融设置 | |
| 速度/部署设置 | |

## 6. 论文结果

- 主表结论：
- 消融结论：
- 速度或部署结果：
- 失败或限制：

## 7. 对本项目的启发

- 可借鉴：
- 不适合照搬：
- 最小可实现版本：
- 需要新增实验：

## 8. 引用定位

该论文应作为：主出处 / 补充出处 / 背景出处。

论文写作中可以这样表述：

> ...
```

## Reading Rules

- Explain principles in Chinese unless the user requests English.
- Do not only translate the abstract.
- Separate paper claims from project claims.
- If the paper does not support the intended innovation, say so clearly.
- Always identify the exact experiment setting used by the paper when available.
