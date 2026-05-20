# Baseline Report

本文件记录阶段 1 baseline 结果。阶段 0 先登记已有历史线索，不直接认定为正式 baseline。

## Current Repository Findings

- Git branch: `main`
- Git commit: `3cb1989 Initial commit`
- `ultralytics.__version__`: `8.2.82`
- Current OBB model code: modified
- Current OBB loss code: modified with `angle_loss`
- Current training path: grayscale single-channel

## Historical Candidate Runs

| Run | Model YAML | Data YAML | Notes | Status |
|---|---|---|---|---|
| `runs/obb/qr_obb_rv11067` | `ultralytics/cfg/models/v8/M2.yaml` | `cfg/datasets/My_project.yaml` | 100 epochs, 256 image size, batch 128, has angle loss columns | Historical candidate |
| `runs/obb/new_300w` | `ultralytics/cfg/models/v8/yolov8-obb.yaml` | `My_project.yaml` | Current `yolov8-obb.yaml` is modified, not official baseline | Historical candidate |
| `rknn_logs/rknn_eval_20260515_010236.txt` | exported from `qr_obb_rv11067` path | `/root/autodl-tmp/yolo_dataset_gray` | RKNN INT8 mAP@0.5 reported as 0.9133, avg inference 74.288 ms/image | Historical candidate |

## Formal Baseline Tasks

| Task | Output | Status |
|---|---|---|
| Validate official architecture reference can build | `docs/stage0_environment_check.md` | Done |
| Validate RV1106 M2 config can build | `docs/stage0_environment_check.md` | Done |
| Run YOLOv8n-OBB official FP32 baseline | `experiments/baseline/` | Pending |
| Run current RV1106 lightweight FP32 baseline | `experiments/rv1106/` | Pending |
| Export current RV1106 lightweight baseline to ONNX | export log | Pending |
| Convert/export RKNN INT8 baseline | RKNN log | Pending |

## Historical Metrics Extracted So Far

| Source | mAP50 | mAP50-95 | angle loss | RKNN mAP50 | Latency |
|---|---:|---:|---:|---:|---:|
| `runs/obb/qr_obb_rv11067/results.csv`, epoch 100 | 0.99059 | 0.95872 | 0.05690 val | | |
| `rknn_logs/rknn_eval_20260515_010236.txt` | | | | 0.9133 | 74.288 ms/image avg |

以上历史指标需要重新确认数据集、权重和导出文件后才能进入正式结果表。
