# RV1106 Deployment Scripts

This folder contains RKNN conversion and deployment evaluation scripts.

## Main Entry

- `convert_eval_rknn.py`: convert the formal ONNX baseline to RKNN and evaluate RKNN mAP@0.5.
- `eval_rknn_basic.py`: legacy early debug script; prefer `convert_eval_rknn.py` for formal experiments.

## Formal Baseline Command

Run this on the AutoDL server after ONNX validation has passed:

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

Use `--build-only` to only export the RKNN model and skip evaluation.

Use `--debug-images 50` for a quick pipeline check before running the full validation split.

On AutoDL simulator, RKNN Toolkit2 cannot run `load_rknn()` with `target=None`. To rerun simulator evaluation, use the formal baseline command above without `--eval-only`; this rebuilds from ONNX and then evaluates with the current AP implementation.

On real RKNN hardware, an already-exported RKNN model can be loaded and evaluated without rebuilding by setting a runtime target:

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

## Baseline Acceptance Rule

- ONNX FP32 baseline: `mAP50=0.990494`, `mAP50-95=0.959600`.
- RKNN INT8 `mAP50` drop <= 0.03 is acceptable for the first deployment baseline.
- RKNN INT8 `mAP50` drop > 0.05 requires debugging quantization data, preprocessing, output decoding, and postprocessing thresholds.
- The RKNN evaluator now reports AP@0.5 using continuous precision-envelope integration instead of 11-point AP.
