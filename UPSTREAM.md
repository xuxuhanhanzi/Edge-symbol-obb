# Upstream and Current Snapshot

本文件冻结阶段 0 的工程基线，后续实验必须先引用这里的版本边界。

## Snapshot

- Date: 2026-05-20
- Repository: `https://github.com/xuxuhanhanzi/Edge-symbol-obb`
- Branch: `main`
- Local commit: `3cb1989 Initial commit`
- Local package version: `ultralytics.__version__ == 8.2.82`
- License inherited from upstream: AGPL-3.0

## Upstream Baseline Status

当前仓库不是干净的 Ultralytics 官方 YOLOv8-OBB baseline。它已经包含面向工业二维码/条形码和 RV1106 的实验性修改，因此后续论文或消融实验不能直接把当前 `ultralytics/cfg/models/v8/yolov8-obb.yaml` 称为官方 baseline。

阶段 0 的 baseline 定义如下：

1. **Official architecture reference**: `configs/baseline/yolov8n_obb_official_arch.yaml`
2. **Current RV1106 lightweight baseline**: `configs/rv1106/yolov8n_obb_rv1106_m2.yaml`
3. **Historical runs**: `runs/` 下已有结果只能作为历史实验线索，进入论文表格前必须重新确认数据集、代码版本、权重和导出链路。

## Known Local Modifications

以下改动已经存在于当前代码中：

- `ultralytics/cfg/models/v8/yolov8-obb.yaml`
  - 已改为灰度单通道 `ch: 1`
  - 已改为 NPU 通道模板
  - 使用 Ghost/C3Ghost/SPPF 等轻量化结构
  - `nc` 固定为 15
- `ultralytics/nn/modules/head.py`
  - `OBB` head 已被重写
  - `reg_max` 改为 8
  - bbox/class/angle 分支通道被压缩
  - 增加 RKNN/ONNX 四头导出路径 `onnx_4head`
- `ultralytics/utils/loss.py`
  - `v8OBBLoss` 已增加 `angle_loss`
  - 使用周期角度损失与长宽比权重
- `ultralytics/models/yolo/obb/train.py`
  - `OBBModel` 构造时强制 `ch=1`
- `scripts/` 实验入口
  - `scripts/train/train_rv1106_gray_obb.py`: 灰度训练入口与预处理 monkey patch
  - `scripts/export/export_gray_obb_onnx.py`: 灰度 ONNX 导出入口
  - `scripts/deploy_rv1106/convert_eval_rknn.py`: RKNN 转换与评估入口
  - `scripts/eval/eval_onnx_4head.py`, `scripts/deploy_rv1106/eval_rknn_basic.py`: 历史评估脚本

## Rule for Future Experiments

所有后续实验记录必须至少包含：

- Git commit
- model yaml
- data yaml
- training command or script
- image size, batch size, epochs
- FP32 validation result
- export status
- INT8/RKNN status when applicable
- weight path
- result directory

如果无法提供以上信息，该结果只能作为探索记录，不能作为正式 baseline 或论文证据。
