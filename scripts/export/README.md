# Export Scripts

计划放置 ONNX、RKNN 前置导出和导出检查脚本。

当前入口：

- `export_gray_obb_onnx.py`: 导出灰度 OBB 权重到 ONNX。

示例：

```text
python -B scripts/export/export_gray_obb_onnx.py --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt --imgsz 256 --opset 19
```
