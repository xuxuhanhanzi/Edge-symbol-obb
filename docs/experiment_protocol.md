# Experiment Protocol

所有正式实验都必须满足本文件的记录规则。缺少记录的结果只能作为探索记录。

## Minimum Record

每次训练、验证、导出和部署实验至少记录：

- Git commit
- Python, PyTorch, CUDA, Ultralytics version
- model yaml
- data yaml
- pretrained weight path
- command or script entry
- image size
- batch size
- epochs
- optimizer and learning rate
- augmentation settings
- result directory
- best weight path

## Baseline Rules

baseline 必须分成三类，不允许混用：

- Official YOLOv8n-OBB baseline: 官方架构和官方 head/loss
- Current lightweight baseline: 当前 RV1106 灰度轻量化代码
- Deployment baseline: ONNX/RKNN/INT8 路径下的真实部署结果

当前仓库已修改 OBB head/loss，因此官方 baseline 需要在干净上游代码或独立分支中复现。

## Required Metrics

- Params
- FLOPs
- mAP50
- mAP50-95
- OBB mAP
- angle loss or angle error
- model size
- ONNX export status
- RKNN conversion status
- INT8 mAP drop
- RV1106 latency/FPS when available

## Result Promotion Rule

一个历史结果进入论文表格前，必须能从以下文件追溯：

- `args.yaml`
- `results.csv`
- `weights/best.pt`
- model yaml snapshot
- data yaml snapshot
- export or deployment log
