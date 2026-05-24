# 阶段三：Hard-Val 固定数据集构建记录

## 1. 阶段目标

阶段二已经从 `val` split 中导出困难候选：

```text
objects=2693
hard_candidates=1565
```

候选数量偏多，且主要由 `large_rotation` 主导，因此不能直接把所有候选作为 hard-val。本阶段目标是进行二次筛选，生成固定、可复现、可检查的数据集。

## 2. 关于 train 来源样本

可以分析 `train` split，帮助发现数据分布、训练难例和后续增强方向。但如果当前模型已经使用 full train 训练过，直接把 train 样本加入 hard-val 会产生训练集泄漏。

因此本阶段分两类输出：

- 正式 hard-val：默认只使用 `val/test` 来源样本。
- train 诊断集：显式使用 `--include-train`，只用于诊断和样本复核，不作为正式泛化指标。

## 3. 正式 Hard-Val 构建命令

```bash
conda activate base
cd ~/ultralytics_yolov8-main/ultralytics_yolov8-main

python -B scripts/data/build_hard_val_from_candidates.py \
  --candidates artifacts/local/dataset_audit/hard_candidates.csv \
  --data datasets/industrial_symbol.yaml \
  --out-root /root/autodl-tmp/yolo_dataset_gray_hard \
  --out-yaml datasets/industrial_symbol_hard.yaml \
  --manifest docs/hard_val_manifest.csv \
  --report docs/hard_val_build_report.md \
  --max-total 700
```

## 4. 可选：Train 诊断集构建命令

先挖掘 train 候选：

```bash
python -B scripts/data/analyze_obb_hard_cases.py \
  --data datasets/industrial_symbol.yaml \
  --split train \
  --out-dir artifacts/local/dataset_audit_train \
  --report docs/dataset_geometry_audit_train.md
```

再生成诊断集：

```bash
python -B scripts/data/build_hard_val_from_candidates.py \
  --candidates artifacts/local/dataset_audit/hard_candidates.csv artifacts/local/dataset_audit_train/hard_candidates.csv \
  --data datasets/industrial_symbol.yaml \
  --out-root /root/autodl-tmp/yolo_dataset_gray_hard_train_diagnostic \
  --out-yaml datasets/industrial_symbol_hard_train_diagnostic.yaml \
  --manifest docs/hard_val_train_diagnostic_manifest.csv \
  --report docs/hard_val_train_diagnostic_build_report.md \
  --max-total 1000 \
  --include-train
```

## 5. 构建后检查

正式 hard-val 构建完成后，必须先检查标签：

```bash
python -B scripts/data/check_labels.py \
  --data datasets/industrial_symbol_hard.yaml \
  --report docs/hard_val_check_report.md
```

然后再进行模型评估：

```bash
python - <<'PY'
from ultralytics import YOLO
from scripts.train.train_rv1106_smoke import install_grayscale_patches

install_grayscale_patches()

for run in [
    "rv1106_m2_e100_b512",
    "rv1106_qg_sincos_e100_b512_selected",
    "official_yolov8n_obb_gray_e20_b512_fix1",
]:
    print("\\n===", run)
    model = YOLO(f"runs/obb/{run}/weights/best.pt")
    model.val(
        task="obb",
        data="datasets/industrial_symbol_hard.yaml",
        imgsz=256,
        batch=128,
        device=0,
        plots=True,
    )
PY
```

## 6. 记录要求

本阶段完成后，需要保存：

- `docs/hard_val_build_report.md`
- `docs/hard_val_manifest.csv`
- `docs/hard_val_check_report.md`
- 三个模型在 hard-val 上的 PyTorch 验证输出

正式论文或阶段总结中，只能使用无 train 泄漏的 hard-val 结果。
