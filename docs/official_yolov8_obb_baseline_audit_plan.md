# 官方 YOLOv8-OBB Baseline 审计计划

## 1. 目标

建立论文中可解释的 YOLOv8n-OBB baseline，并将其与 RV1106-M2 工程轻量模型区分开。

核心原则：

```text
官方风格 baseline 应使用标准 YOLOv8n-OBB 结构：
Conv、C2f、SPPF、PAN-style head、OBB head。
```

它不应使用 RV1106-M2 工程模块，例如 `SCDown`、`C2fCIB`，也不应使用本仓库被改写过的 NPU/Ghost 模板。

## 2. 审计结果

第一次服务器审计输出过长，因为命令中使用了 `print(model)` 和 `print(head)`。虽然控制台被截断，但已经暴露了关键事实：

| 配置 | 结果 | 决策 |
|---|---|---|
| `ultralytics/cfg/models/v8/yolov8-obb.yaml` | 本地 NPU/Ghost 模板，包含 `GhostConv` 和 `C3Ghost`，参数量约 3,013,072 | 不能作为官方 YOLOv8-OBB baseline |
| `configs/baseline/yolov8n_obb_official_arch.yaml` | 官方风格 YOLOv8n-OBB 结构，3 通道输入，参数量约 3,488,104 | 保留为 3 通道结构参考 |
| `configs/baseline/yolov8n_obb_official_gray.yaml` | 官方风格 YOLOv8n-OBB 结构，显式 `ch: 1` 灰度输入 | 推荐作为当前灰度数据集的官方风格 baseline |

重要说明：

当前仓库已经修改过 OBB head、loss 和 export 逻辑。因此这里的 baseline 是“当前本地代码下的官方风格结构 baseline”，不是严格干净上游 Ultralytics baseline。

## 3. 更安全的审计命令

不要打印完整模型。使用摘要审计命令：

```bash
cd ~/ultralytics_yolov8-main/ultralytics_yolov8-main

python - <<'PY'
from ultralytics import YOLO

candidates = [
    "ultralytics/cfg/models/v8/yolov8-obb.yaml",
    "configs/baseline/yolov8n_obb_official_arch.yaml",
    "configs/baseline/yolov8n_obb_official_gray.yaml",
]

for cfg in candidates:
    print("\n===", cfg)
    try:
        m = YOLO(cfg).model
        head = m.model[-1]
        first = next(m.parameters())
        modules = [type(x).__name__ for x in m.model[:10]]
        print("params", sum(p.numel() for p in m.parameters()))
        print("first_weight_shape", tuple(first.shape))
        print("input_channels", tuple(first.shape)[1])
        print("head_type", type(head).__name__)
        print("head_ne", getattr(head, "ne", None))
        print("head_no", getattr(head, "no", None))
        print("first_10_modules", modules)
    except Exception as e:
        print("ERROR", type(e).__name__, e)
PY
```

判断规则：

- 如果模块列表中出现 `GhostConv` 或 `C3Ghost`，则该配置不是官方 baseline。
- 如果首层权重通道数为 1，且模块为 Conv/C2f/SPPF，则可作为灰度官方风格 baseline。
- 如果首层权重通道数为 3，则它是 3 通道结构参考，不应直接用于当前灰度对比，除非训练链路也明确保持 RGB。

## 4. 已知验证通道问题

第一次 `official_yolov8n_obb_gray_e20_b512` 尝试在第 1 个 epoch 后失败，错误为：

```text
RuntimeError: expected input[512, 3, 288, 288] to have 1 channels
```

原因：

- 训练 preprocess 已经正确将 3 通道图像转为 1 通道。
- 但验证阶段 `BaseValidator.__call__()` 没有把局部变量 `model` 保存到 `self.model`。
- 因此 `OBBValidator.preprocess()` 无法判断模型期望 1 通道输入，导致 validation batch 保持 3 通道。

修复：

- 在 `ultralytics/engine/validator.py` 中恢复 `self.model = model`。
- 将 `args.model` 传入 `install_grayscale_patches(args.model)`。
- 在训练脚本中使用模型 yaml 的 `ch` 字段作为通道数 fallback。

修复后，`official_yolov8n_obb_gray_e20_b512_fix1` 已成功跑完 20 epoch。

## 5. Quick Check 命令

```bash
conda activate base
cd ~/ultralytics_yolov8-main/ultralytics_yolov8-main

python -B scripts/train/train_rv1106_smoke.py \
  --model configs/baseline/yolov8n_obb_official_gray.yaml \
  --data datasets/industrial_symbol.yaml \
  --name official_yolov8n_obb_gray_e20_b512 \
  --epochs 20 \
  --batch 512 \
  --workers 16 \
  --device 0 \
  --imgsz 256
```

目的：

- 确认官方风格灰度 baseline 可以训练。
- 与 `rv1106_m2_e20_b512_compare` 和 `rv1106_qg_sincos_e20_b512` 做 quick 对照。
- 判断是否值得进入 full 100 epoch。

## 6. Full Run 命令

只有在 quick check 正常后才运行：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/baseline/yolov8n_obb_official_gray.yaml \
  --data datasets/industrial_symbol.yaml \
  --name official_yolov8n_obb_gray_e100_b512 \
  --epochs 100 \
  --batch 512 \
  --workers 16 \
  --device 0 \
  --imgsz 256
```

根据当前修正版实验方案，official gray e100 已暂缓，不再是立即优先任务。

## 7. ONNX 与 RKNN 验证

训练完成后复用同一导出和部署路径：

```bash
RUN=official_yolov8n_obb_gray_e100_b512

python -B scripts/export/export_gray_obb_onnx.py \
  --weights runs/obb/$RUN/weights/best.pt \
  --imgsz 256 \
  --opset 19 \
  --device 0

python -B scripts/eval/val_onnx_gray.py \
  --weights runs/obb/$RUN/weights/best.onnx \
  --data datasets/industrial_symbol.yaml \
  --imgsz 256 \
  --batch 1
```

然后切换到 `rknn232` 环境，使用 `scripts/deploy_rv1106/convert_eval_rknn.py` 进行 RKNN 转换和评估。

## 8. 当前对比表目标

| 模型 | 角色 | PyTorch mAP50 | PyTorch mAP50-95 | ONNX mAP50 | ONNX mAP50-95 | RKNN mAP50 | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| Official YOLOv8n-OBB gray | 论文 baseline | 待补 | 待补 | 待补 | 待补 | 可选 | 官方风格结构 |
| Official YOLOv8n-OBB gray + QG | 论文方法 | 待补 | 待补 | 待补 | 待补 | 可选 | 后续必要时实现 |
| RV1106-M2 scalar | 部署 baseline | 0.991 | 0.960 | 0.9905 | 0.9596 | 0.9843 | 已完成 |
| RV1106-M2 + QG | 部署方法 | 0.990 | 0.957 | 0.9895 | 0.9577 | 0.9854 | 已完成 |

## 9. 结论边界

- 如果官方风格 baseline 只用于 PyTorch/ONNX 对比，需要明确说明。
- 如果 RKNN 转换成功，可将其 RKNN 结果作为额外部署对比。
- 当前最强 QG 结论仍然是：

```text
QG 并未在 100 epoch 下提升 FP32 精度；
但它改善了 RV1106-M2 路线上的 RKNN INT8 精度保持率。
```

## 10. 当前状态

根据修正版实验方案，official gray baseline 已完成 e20 sanity check。正式 e100 暂缓。

下一步优先级转为：

1. 创新点闭环表。
2. 当前数据集审计。
3. hard-val 构建。
4. 外部数据集搜索。
