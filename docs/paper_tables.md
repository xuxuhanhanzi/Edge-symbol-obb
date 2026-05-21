# Paper Tables

## Main Results

| Model | Params | FLOPs | mAP50 | mAP50-95 | Angle Error/Loss | FPS-GPU | FPS-RV1106 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n-OBB official | | | | | | | |
| YOLOv8s-OBB official | | | | | | | |
| RV1106-M2 lightweight baseline | 2.14M | 9.0G | 0.991 | 0.960 | val angle loss TBD | | |
| + QG-OBB Head | | | | | | | |
| + SOF-FPN | | | | | | | |
| Full Model | | | | | | | |

## Ablation

| QG-OBB | SOF-FPN | GIS-Aug | mAP50 | mAP50-95 | Angle Error/Loss | INT8 Drop |
|---|---|---|---:|---:|---:|---:|
| No | No | No | | | | |
| Yes | No | No | | | | |
| No | Yes | No | | | | |
| Yes | Yes | No | | | | |
| Yes | Yes | Yes | | | | |

## Deployment

| Model | Format | Precision | Size | mAP50 | mAP50-95 | mAP50 Drop | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| RV1106-M2 lightweight baseline | PyTorch | FP32 | 4.6 MB | 0.991 | 0.960 | 0.000 | |
| RV1106-M2 lightweight baseline | ONNX | FP32 | 8.2 MB | 0.990 | 0.960 | 0.001 | 14.96 ms/image |
| RV1106-M2 lightweight baseline debug subset | RKNN | INT8 | | 0.913 | | 0.077 | 776.55 ms/image simulator |
| RV1106-M2 lightweight baseline full val | RKNN | INT8 | | 0.985 | | 0.006 | 76.68 ms/image simulator |
| Full Model | PyTorch | FP32 | | | | | |
| Full Model | RKNN | INT8 | | | | | |

## Baseline Run Notes

- Current formal FP32 baseline: `runs/obb/rv1106_m2_baseline_e100_b256`.
- Actual training batch was `512`, although the run name contains `b256`.
- Training setup: `epochs=100`, `imgsz=256`, `workers=16`, `amp=False`, `optimizer=SGD`.
- Hardware: NVIDIA GeForce RTX 5090, peak reported memory about 30.3 GB.
- Current ONNX FP32 baseline: `runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx`.
- ONNX validation: `P=0.985`, `R=0.980`, `mAP50=0.990`, `mAP50-95=0.960`, save dir `runs/obb/val2`.
- RKNN debug run: `rknn_logs/rknn_eval_20260521_002153.txt`, 50 images, simulator runtime, `mAP@0.5=0.9133`.
- RKNN debug latency is not a final device latency because it used simulator mode and includes unstable first-run overhead.
- RKNN full validation run: `rknn_logs/rknn_eval_20260521_002912.txt`, 2635 images, simulator runtime, simplified 11-point `mAP@0.5=0.9134`.
- RKNN full validation mAP is not directly identical to Ultralytics COCO-style mAP50 because the RKNN script currently uses a simplified AP implementation.
- RKNN continuous-AP validation run: `rknn_logs/rknn_eval_20260521_004405.txt`, 2635 images, simulator runtime, continuous precision-envelope `mAP@0.5=0.9846`.
- The continuous-AP RKNN INT8 drop versus ONNX FP32 mAP50 is about `0.0059`; this is acceptable for the first deployment baseline.
- Physical RV1106 board latency/FPS testing is deferred because no endpoint device is currently available. Simulator latency must not be used as the final deployment latency.
