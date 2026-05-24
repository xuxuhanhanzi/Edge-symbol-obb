# Experiment Record Template

Use this reference when creating `docs/experiments/<date>_<stage>_<run>.md`.

## Template

```markdown
# 实验记录：<run_name>

## 1. 目标

## 2. 环境

- 平台：
- GPU：
- Python：
- PyTorch：
- CUDA：
- Conda 环境：
- 代码路径：
- 代码 commit / diff：
- 数据集路径：

## 3. 实验变量

- 主变量：
- 固定项：
- 对比对象：

## 4. 命令

```bash

```

## 5. 输出路径

- 权重：
- 日志：
- TensorBoard：
- ONNX：
- RKNN 或部署模型：
- 可视化：

## 6. 结果

| 指标 | 数值 |
|---|---:|

## 7. 失败与异常

- 现象：
- 报错：
- 判断：
- 处理：

## 8. 结论

## 9. 下一步
```

## Record Rules

- Copy exact commands from the terminal.
- Record absolute paths on AutoDL.
- Record failed commands and fixes.
- Record environment versions when dependency issues occur.
- Do not overwrite earlier conclusions after later experiments; append updated conclusions.

## Minimal Table

```markdown
| Run | Main variable | PyTorch mAP50 | ONNX mAP50 | Deployment mAP50 | Drop | Decision |
|---|---|---:|---:|---:|---:|---|
```
