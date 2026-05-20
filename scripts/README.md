# Scripts

本目录保存项目级脚本入口。脚本应尽量只做一件事，并把模型、数据、输出路径写清楚。

当前阶段 0 可用入口：

```text
python -B scripts\train\smoke_obb.py
```

## Layout

| Directory | Purpose |
|---|---|
| `scripts/data/` | 数据转换、标签检查、数据集统计 |
| `scripts/train/` | 训练入口和训练 smoke check |
| `scripts/eval/` | PyTorch/ONNX 评估、输出对比 |
| `scripts/export/` | ONNX/RKNN 前置导出 |
| `scripts/deploy_rv1106/` | RKNN 转换、RV1106 推理和部署评估 |

## Moved Legacy Entrypoints

| Old root file | New path |
|---|---|
| `model_run_train.py` | `scripts/train/train_rv1106_gray_obb.py` |
| `export_onnx.py` | `scripts/export/export_gray_obb_onnx.py` |
| `val_onnx_gray.py` | `scripts/eval/val_onnx_gray.py` |
| `eval_onnx.py` | `scripts/eval/eval_onnx_4head.py` |
| `compare_std_vs_4head.py` | `scripts/eval/compare_std_vs_4head.py` |
| `eval_rknn.py` | `scripts/deploy_rv1106/eval_rknn_basic.py` |
| `rknn_convert_eval.py` | `scripts/deploy_rv1106/convert_eval_rknn.py` |
