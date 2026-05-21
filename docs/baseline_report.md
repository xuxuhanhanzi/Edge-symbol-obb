# Baseline Report

This document records the formal baseline status for the RV1106 grayscale OBB project.

## Current Repository Findings

- Git branch: `main`
- Git commit at initial audit: `3cb1989 Initial commit`
- `ultralytics.__version__`: `8.2.82`
- Current task: grayscale single-channel OBB detection
- Current formal dataset entry: `datasets/industrial_symbol.yaml`
- Dataset label check: PASS after quarantining invalid samples

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
| Validate dataset labels | `docs/dataset_report.md` | Done |
| Run current RV1106 lightweight FP32 baseline | `runs/obb/rv1106_m2_baseline_e100_b256` | Done |
| Export current RV1106 lightweight baseline to ONNX | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx` | Done |
| Validate current RV1106 lightweight ONNX baseline | `runs/obb/val2` | Done |
| Convert/export RKNN INT8 baseline | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn` | Debug done |
| Validate RKNN INT8 baseline | `rknn_logs/rknn_eval_20260521_002912.txt` | Done |
| Re-evaluate RKNN INT8 with continuous AP | `rknn_logs/rknn_eval_20260521_004405.txt` | Done |

## Next RKNN Baseline Command

The RKNN conversion/evaluation entry has been parameterized in `scripts/deploy_rv1106/convert_eval_rknn.py`.

Run this command on AutoDL for the next formal baseline step:

```bash
python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --onnx runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --rknn runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --dataset-txt artifacts/local/dataset_rknn.txt \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_inference_results \
  --target-platform rv1106 \
  --imgsz 256 \
  --quant-images 500
```

For a quick pipeline test, add `--debug-images 50`.

On AutoDL simulator, RKNN Toolkit2 cannot run `load_rknn()` with `target=None`. To rerun simulator evaluation with the updated AP implementation, use the full build-and-eval command above without `--eval-only`.

On real RKNN hardware, rerun RKNN evaluation without rebuilding the already-exported model by setting a runtime target:

```bash
python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --eval-only \
  --rknn runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_inference_results \
  --runtime-target rv1106 \
  --target-platform rv1106 \
  --imgsz 256
```

## RKNN Debug Run

The first RKNN INT8 debug run completed successfully on AutoDL with `rknn_env`.

Environment checks:

```bash
conda activate rknn_env
python -c "from rknn.api import RKNN; print('rknn_ok')"
```

Result:

- `rknn_ok`
- RKNN Toolkit2 version: `1.6.0+81f21f4d`

Debug command:

```bash
export OMP_NUM_THREADS=1

python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --onnx runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --rknn runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --dataset-txt artifacts/local/dataset_rknn.txt \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_inference_results \
  --target-platform rv1106 \
  --imgsz 256 \
  --quant-images 500 \
  --debug-images 50
```

Debug output summary:

| Item | Value |
|---|---:|
| Log | `rknn_logs/rknn_eval_20260521_002153.txt` |
| RKNN model | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn` |
| Quantization | INT8 |
| Calibration images | 500 |
| Debug eval images | 50 |
| Runtime mode | RKNN simulator (`target=None`) |
| mAP@0.5 | 0.9133 |
| Avg inference time | 776.553 ms/image |
| Min inference time | 42.040 ms/image |
| Max inference time | 35523.643 ms/image |

Per-class AP on the 50-image debug subset:

| Class | AP@0.5 |
|---|---:|
| DM | 0.8663 |
| QR | 0.9603 |

Notes:

- This run proves the RKNN conversion and evaluation pipeline is functional.
- The debug subset contains GT only for `DM` and `QR`, so this is not a formal full-dataset mAP.
- The reported latency is from RKNN simulator, not physical RV1106 hardware. The first session/runtime overhead also makes the average unstable, as shown by the 35523.643 ms max time.
- The RKNN build warning about input/output dtype changing to `int8` is expected for an INT8 RKNN model, but deployment preprocessing and postprocessing must respect this dtype change.

## Formal RKNN INT8 Full Validation

The full validation split was evaluated with the current RKNN INT8 baseline.

Command:

```bash
export OMP_NUM_THREADS=1

python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --onnx runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --rknn runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --dataset-txt artifacts/local/dataset_rknn.txt \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_inference_results \
  --target-platform rv1106 \
  --imgsz 256 \
  --quant-images 500
```

Summary:

| Item | Value |
|---|---:|
| Log | `rknn_logs/rknn_eval_20260521_002912.txt` |
| RKNN model | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn` |
| Quantization | INT8 |
| Calibration images | 500 |
| Eval images | 2635 |
| Runtime mode | RKNN simulator (`target=None`) |
| CONF threshold | 0.25 |
| NMS threshold | 0.7 |
| TOPK | 300 |
| mAP@0.5 | 0.9134 |
| Avg inference time | 79.521 ms/image |
| Min inference time | 39.279 ms/image |
| Max inference time | 34568.725 ms/image |

Per-class AP@0.5:

| Class | AP@0.5 |
|---|---:|
| BARCODE | 0.9074 |
| DM | 0.9046 |
| HANXIN | 0.9091 |
| QR | 0.8895 |
| PDF | 0.9091 |
| AZTEC | 0.9091 |
| CODEONE | 0.9091 |
| DOT | 0.9091 |
| GM | 0.9091 |
| MAXI | 1.0000 |
| MPDF | 0.9091 |
| MQR | 0.9091 |
| RMQR | 0.9091 |
| ULTRA | 0.9091 |
| UPN | 0.9091 |

Interpretation:

- The RKNN INT8 full-validation pipeline is functional.
- The reported speed is simulator speed, not physical RV1106 board speed.
- The max latency includes simulator/session overhead and should not be used as deployment latency.
- The RKNN evaluator currently uses an 11-point AP calculation. This is useful for trend checking but is not strictly the same metric implementation as the Ultralytics ONNX/PyTorch mAP50.
- The apparent drop from ONNX mAP50 `0.9905` to RKNN simplified AP@0.5 `0.9134` is about `0.0771`, but this combines both quantization/runtime effects and metric implementation differences.

Follow-up:

- `scripts/deploy_rv1106/convert_eval_rknn.py` has been updated to use continuous precision-envelope AP integration.
- The previous `0.9134` result should be rerun with the full build-and-eval command on AutoDL simulator before making a final RKNN accuracy conclusion.
- `--eval-only` is only suitable for real RKNN hardware with `--runtime-target`; it is not suitable for AutoDL simulator mode.

## Formal RKNN INT8 Full Validation with Continuous AP

The RKNN INT8 full-validation run was repeated after updating the RKNN evaluator from 11-point AP to continuous precision-envelope AP integration.

Command:

```bash
export OMP_NUM_THREADS=1

python -B scripts/deploy_rv1106/convert_eval_rknn.py \
  --onnx runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --rknn runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn \
  --image-dir /root/autodl-tmp/yolo_dataset_gray/val/images \
  --label-dir /root/autodl-tmp/yolo_dataset_gray/val/labels \
  --dataset-txt artifacts/local/dataset_rknn.txt \
  --log-dir rknn_logs \
  --save-dir artifacts/local/rknn_inference_results \
  --target-platform rv1106 \
  --imgsz 256 \
  --quant-images 500
```

Summary:

| Item | Value |
|---|---:|
| Log | `rknn_logs/rknn_eval_20260521_004405.txt` |
| RKNN model | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best_int8_rv1106.rknn` |
| Quantization | INT8 |
| Calibration images | 500 |
| Eval images | 2635 |
| Runtime mode | RKNN simulator (`target=None`) |
| AP method | Continuous precision-envelope integration |
| CONF threshold | 0.25 |
| NMS threshold | 0.7 |
| TOPK | 300 |
| mAP@0.5 | 0.9846 |
| Avg inference time | 76.679 ms/image |
| Min inference time | 39.273 ms/image |
| Max inference time | 34098.166 ms/image |

Per-class AP@0.5:

| Class | AP@0.5 |
|---|---:|
| BARCODE | 0.9782 |
| DM | 0.9605 |
| HANXIN | 0.9907 |
| QR | 0.9375 |
| PDF | 0.9856 |
| AZTEC | 0.9939 |
| CODEONE | 0.9950 |
| DOT | 0.9776 |
| GM | 0.9950 |
| MAXI | 0.9950 |
| MPDF | 0.9905 |
| MQR | 0.9950 |
| RMQR | 0.9946 |
| ULTRA | 0.9854 |
| UPN | 0.9950 |

Interpretation:

- Compared with the ONNX FP32 mAP50 of `0.990494`, RKNN INT8 mAP@0.5 is `0.9846`.
- The absolute drop is about `0.0059`, which is well within the initial acceptable deployment baseline threshold.
- The previous `0.9134` result was mainly caused by the earlier 11-point AP implementation and should not be used as the formal accuracy result.
- The reported latency is still simulator latency, not physical RV1106 board latency.
- Physical RV1106 board latency/FPS testing is intentionally deferred because no endpoint device is currently available. The current simulator latency must not be used as a final deployment latency claim.
- The next phase is model improvement. Detailed plan: `docs/model_improvement_plan.md`.
- Stage completion report: `docs/project_completion_report.md`.
- Revised experiment plan: `docs/revised_experiment_plan.md`.
- Handoff prompt for future sessions: `docs/session_handoff_prompt.md`.

## Formal FP32 PyTorch Baseline

| Run | Model YAML | Data YAML | Epochs | Batch | Image Size | Params | FLOPs | mAP50 | mAP50-95 | Training Time | Weight |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `runs/obb/rv1106_m2_baseline_e100_b256` | `configs/rv1106/yolov8n_obb_rv1106_m2.yaml` | `datasets/industrial_symbol.yaml` | 100 | 512 | 256 | 2.14M | 9.0G | 0.991 | 0.960 | 1.154 h | `runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt` |

Notes:

- The run directory name contains `b256`, but the actual command used `--batch 512`.
- Peak reported GPU memory was about 30.3 GB on an NVIDIA GeForce RTX 5090.
- Final validation on `best.pt` reported `P=0.986`, `R=0.975`, `mAP50=0.991`, `mAP50-95=0.960`.

## Formal ONNX Baseline

Export command:

```bash
python -B scripts/export/export_gray_obb_onnx.py \
  --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt \
  --imgsz 256 \
  --opset 19
```

Validation command:

```bash
python -B scripts/eval/val_onnx_gray.py \
  --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --data datasets/industrial_symbol.yaml \
  --imgsz 256
```

| Model | Format | Precision | Image Size | ONNX Size | P | R | mAP50 | mAP50-95 | Save Dir |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `rv1106_m2_baseline_e100_b256` | ONNX | FP32 | 256 | 8.2 MB | 0.985 | 0.980 | 0.990 | 0.960 | `runs/obb/val2` |

ONNX validation result dictionary:

- `metrics/precision(B)`: 0.9846879004808976
- `metrics/recall(B)`: 0.9795905864006227
- `metrics/mAP50(B)`: 0.9904943591129942
- `metrics/mAP50-95(B)`: 0.9596001726240346
- `fitness`: 0.9626895912729306

ONNX speed on AutoDL RTX 5090:

- preprocess: 0.49 ms/image
- inference: 14.96 ms/image
- postprocess: 6.40 ms/image

Interpretation:

- ONNX accuracy is aligned with the PyTorch FP32 baseline.
- PyTorch mAP50-95 was 0.960 and ONNX mAP50-95 was 0.9596, so the export did not introduce a meaningful accuracy regression.
- The warning `libgomp: Invalid value for environment variable OMP_NUM_THREADS` is an environment variable issue, not an accuracy or export failure.

## Historical Metrics Extracted So Far

| Source | mAP50 | mAP50-95 | angle loss | RKNN mAP50 | Latency |
|---|---:|---:|---:|---:|---:|
| `runs/obb/qr_obb_rv11067/results.csv`, epoch 100 | 0.99059 | 0.95872 | 0.05690 val | | |
| `rknn_logs/rknn_eval_20260515_010236.txt` | | | | 0.9133 | 74.288 ms/image avg |
