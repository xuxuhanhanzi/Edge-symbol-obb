# Experiment Planning

Use this reference before running experiments.

## Stage Plan Template

```markdown
# 实验计划：<stage_name>

## 1. 阶段目标

## 2. 假设

## 3. 主变量

## 4. 固定项

- 数据：
- imgsz：
- epochs：
- batch：
- optimizer：
- augmentation：
- seed：
- export setting：
- deployment setting：

## 5. 对比对象

- 官方 baseline：
- 工程 baseline：
- 新方案：

## 6. 指标

- PyTorch:
- ONNX:
- Deployment:
- Per-class:
- Speed/size:

## 7. 成功门槛

## 8. 失败处理

## 9. 预计产物

- 权重：
- 日志：
- 表格：
- 文档：
```

## Experiment Naming

Use:

```text
<platform>_<module>_<variant>_e<epochs>_b<batch>_<tag>
```

Examples:

```text
rv1106_qg_sincos_e100_b512_selected
official_yolov8n_obb_e100_b512
rv1106_m2_e20_b512_compare
```

## Planning Rules

- Run smoke tests before quick training.
- Run quick training before long training.
- Do not launch a full experiment until export and evaluation interfaces are known.
- Keep a failed experiment in the record if it changes the next decision.
- If a deployment path exists, the deployment result is part of the experiment, not an optional afterthought.

## Common Stage Sequence

```text
Stage 0: environment and dataset check
Stage A: baseline reproduction
Stage B: single-module smoke and quick ablation
Stage C: selected module full training
Stage D: export and deployment validation
Stage E: official baseline comparison
Stage F: paper table and claim consolidation
```
