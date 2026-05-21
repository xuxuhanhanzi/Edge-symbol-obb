# Session Handoff Prompt

下面这段内容用于在更换 AI 会话后快速交接项目。新会话中的 AI 应先完整阅读这份 prompt，再继续协助项目开发。

---

你现在需要接手一个名为 `Edge-symbol-obb` 的工业符号旋转框检测项目。项目基于 Ultralytics YOLOv8 OBB，目标是在灰度单通道工业符号数据集上训练轻量 OBB 模型，并最终面向 Rockchip RV1106 NPU 做 ONNX / RKNN / INT8 部署。

本地项目路径：

```text
C:/Users/27475/Desktop/ultralytics_yolov8-main/ultralytics_yolov8-main
```

重要约束：

- 不要批量删除文件或目录。
- 禁止使用 `del /s`, `rd /s`, `rmdir /s`, `Remove-Item -Recurse`, `rm -rf`。
- 需要删除文件时只能一次删除一个明确路径的文件。
- 修改文件优先使用补丁方式，避免无关重构。
- 用户主要在 Windows 本机维护代码，在 AutoDL 服务器上训练和运行 RKNN。
- 当前没有真实 RV1106 端侧设备，因此实机 FPS / latency 测试暂缓。
- RKNN simulator latency 只能作为调试参考，不能作为论文最终部署速度。

## 1. 项目当前状态

项目已经完成 baseline 阶段，进入模型改进阶段。

已完成完整链路：

```text
dataset check
-> PyTorch FP32 training
-> ONNX FP32 export
-> ONNX FP32 validation
-> RKNN INT8 conversion
-> RKNN INT8 simulator validation
```

当前正式 baseline：

```text
Model YAML: configs/rv1106/yolov8n_obb_rv1106_m2.yaml
Data YAML: datasets/industrial_symbol.yaml
Task: grayscale single-channel OBB detection
Input size: 256
Classes: 15
Run: runs/obb/rv1106_m2_baseline_e100_b256
```

注意：run name 里有 `b256`，但实际训练 batch 是 `512`。

## 2. 关键文档

请优先阅读这些文件：

```text
docs/project_completion_report.md
docs/revised_experiment_plan.md
docs/baseline_report.md
docs/model_improvement_plan.md
docs/paper_tables.md
```

这些文档分别记录：

- 当前项目完成情况
- 修正后的实验计划
- baseline 全部实验数据
- 模型改进阶段计划
- 论文表格草稿

原始用户计划文件在：

```text
C:/Users/27475/Desktop/Edge-symbol-obb_实验计划.md
```

该文件内容有编码显示问题，但核心方向已经在 `docs/revised_experiment_plan.md` 中重新整理。

## 3. 已完成代码工作

### 数据检查

已实现：

```text
scripts/data/check_labels.py
scripts/data/quarantine_bad_samples.py
```

服务器上数据检查初始 FAIL，坏样本隔离后 PASS。

已处理异常：

| Type | Count |
|---|---:|
| checkpoint image | 6 |
| orphan label | 3 |
| zero-area label | 74 |
| total actions | 83 |

最终状态：

```text
status=PASS
```

### 训练入口

已实现：

```text
scripts/train/train_rv1106_smoke.py
```

正式 baseline 训练命令：

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --name rv1106_m2_baseline_e100_b256 \
  --epochs 100 \
  --batch 512 \
  --workers 16
```

### ONNX 导出与验证

已实现：

```text
scripts/export/export_gray_obb_onnx.py
scripts/eval/val_onnx_gray.py
```

`val_onnx_gray.py` 已经解决两个关键问题：

- 灰度单通道输入适配。
- OBB ONNX 4-head 输出 decode 后再进入 Ultralytics NMS。

### RKNN 转换与验证

已实现：

```text
scripts/deploy_rv1106/convert_eval_rknn.py
```

该脚本已经参数化，并支持：

- ONNX 输入路径
- RKNN 输出路径
- 数据路径
- calibration 图片数量
- debug images
- build-only
- eval-only with real runtime target
- continuous precision-envelope AP

注意：

- AutoDL simulator 不能使用 `--eval-only` 加载已有 `.rknn` 推理。
- RKNN Toolkit2 限制：`load_rknn()` 不支持 `target=None` simulator 推理。
- 在 AutoDL simulator 上复评必须重新走 `load_onnx -> build -> init_runtime(target=None)`。
- `--eval-only` 只适用于真实 RKNN 设备，并且需要 `--runtime-target rv1106`。

## 4. 当前实验数据

### PyTorch FP32 baseline

训练环境：

```text
AutoDL
GPU: NVIDIA GeForce RTX 5090
Ultralytics: 8.2.82
torch: 2.7.0+cu128
epochs: 100
batch: 512
imgsz: 256
workers: 16
amp: False
optimizer: SGD
```

模型规模：

```text
Params: 2.14M
FLOPs: 9.0G
best.pt size: 4.6 MB
```

结果：

| Metric | Value |
|---|---:|
| Precision | 0.986 |
| Recall | 0.975 |
| mAP50 | 0.991 |
| mAP50-95 | 0.960 |
| Training time | 1.154 h |

权重：

```text
runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt
```

### ONNX FP32 baseline

导出命令：

```bash
python -B scripts/export/export_gray_obb_onnx.py \
  --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt \
  --imgsz 256 \
  --opset 19
```

验证命令：

```bash
python -B scripts/eval/val_onnx_gray.py \
  --weights runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx \
  --data datasets/industrial_symbol.yaml \
  --imgsz 256
```

ONNX 结果：

| Metric | Value |
|---|---:|
| ONNX size | 8.2 MB |
| Precision | 0.9846879004808976 |
| Recall | 0.9795905864006227 |
| mAP50 | 0.9904943591129942 |
| mAP50-95 | 0.9596001726240346 |
| Fitness | 0.9626895912729306 |

结论：

```text
ONNX export passed. Accuracy is aligned with PyTorch FP32.
```

### RKNN INT8 baseline

RKNN 环境：

```bash
conda activate rknn_env
python -c "from rknn.api import RKNN; print('rknn_ok')"
```

结果：

```text
rknn_ok
RKNN Toolkit2 version: 1.6.0+81f21f4d
```

正式 RKNN continuous AP 复评命令：

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

正式结果：

| Metric | Value |
|---|---:|
| Log | `rknn_logs/rknn_eval_20260521_004405.txt` |
| Eval images | 2635 |
| Calibration images | 500 |
| Runtime | simulator |
| AP method | continuous precision-envelope integration |
| RKNN INT8 AP@0.5 | 0.9846 |
| Avg simulator inference | 76.679 ms/image |

和 ONNX 对比：

```text
ONNX mAP50: 0.990494
RKNN INT8 AP@0.5: 0.9846
Drop: about 0.0059
```

结论：

```text
RKNN INT8 accuracy passed. The INT8 drop is acceptable.
```

注意：

```text
76.679 ms/image 是 RKNN simulator latency，不是 RV1106 实机速度。
```

## 5. 暂缓事项

暂缓真实 RV1106 板端速度测试。

原因：

```text
当前用户手边没有端侧 RV1106 设备。
```

要求：

- 不要把 simulator latency 写成最终部署速度。
- 论文或报告中只能写 simulator runtime 或 deferred。
- 后续有设备后再用 `--eval-only --runtime-target rv1106` 或板端 runtime 脚本补测。

## 6. 下一阶段任务

下一阶段进入模型改进阶段。

优先级：

```text
1. QG-OBB Head
2. SOF-FPN
3. GIS-Aug
4. Full Model
```

当前最应该执行的任务是：

```text
创建 docs/qg_obb_head_design.md
审计当前 OBB head / loss / ONNX export / RKNN decode 接口
设计最小 sin-cos QG-OBB Head
```

不要立刻直接改模型代码。先做设计审计。

## 7. QG-OBB Head 计划

目标：

```text
提升 angle branch 的周期稳定性和 INT8 量化稳定性。
```

第一步审计文件：

```text
ultralytics/nn/modules/head.py
相关 OBB loss 文件
scripts/eval/val_onnx_gray.py
scripts/deploy_rv1106/convert_eval_rknn.py
configs/rv1106/yolov8n_obb_rv1106_m2.yaml
```

需要记录：

- 当前 OBB head 输入输出 shape。
- 当前 angle 输出范围。
- 当前 angle decode 方式。
- 当前 ONNX 4-head 输出格式。
- 当前 RKNN 后处理使用 angle 的方式。
- 如果改成 sin-cos angle branch，需要同步修改哪些文件。

最小设计方向：

```text
angle scalar -> sin(theta), cos(theta)
decode -> atan2(sin, cos)
box branch unchanged
cls branch unchanged
loss adds sin-cos alignment or unit-cycle constraint
ONNX/RKNN decode must be updated
```

实验顺序：

```text
shape test
-> 1 epoch smoke train
-> 20 epoch quick train
-> 100 epoch full train
-> ONNX export
-> ONNX validation
-> RKNN INT8 conversion
-> RKNN INT8 validation
```

建议 run names：

```text
rv1106_qg_sincos_smoke
rv1106_qg_sincos_e20_b512
rv1106_qg_sincos_e100_b512
```

通过标准：

| Metric | Requirement |
|---|---:|
| PyTorch mAP50 | >= baseline - 0.003 |
| PyTorch mAP50-95 | >= baseline - 0.005 |
| ONNX export | PASS |
| ONNX mAP drop | <= 0.005 |
| RKNN INT8 AP@0.5 drop vs ONNX | <= 0.03 |
| Params/FLOPs | no obvious increase |

## 8. SOF-FPN 计划

在 QG-OBB Head 完成后进行。

目标：

```text
提升多尺度符号、方向敏感目标和低质量图像的特征表达。
```

优先尝试：

```text
P3 / P4 lightweight orientation-preserving branch
depthwise conv
lightweight fusion gate
```

暂不优先引入 FFT 或复杂频域算子，因为 RKNN 部署风险较高。

## 9. GIS-Aug 计划

在结构稳定后进行。

候选增强：

```text
random rotation
perspective transform
motion blur
low light
reflection / glare
partial occlusion
small object scaling
large aspect-ratio variation
```

先做可控增强，不要优先做 diffusion synthetic data。

## 10. 对新 AI 的工作要求

接手后请先做：

1. 阅读 `docs/project_completion_report.md`。
2. 阅读 `docs/revised_experiment_plan.md`。
3. 阅读 `docs/model_improvement_plan.md`。
4. 不要重复 baseline 阶段已经完成的实验。
5. 不要把 RKNN simulator latency 当成真实 RV1106 latency。
6. 下一步从 `docs/qg_obb_head_design.md` 开始。

如果用户要求继续推进代码，请先完成 QG-OBB Head 的设计审计文档，再进入代码实现。

---

推荐下一句回复用户：

```text
我会先创建 docs/qg_obb_head_design.md，并审计当前 OBB head、loss、ONNX 输出和 RKNN decode 的接口，再给出最小 sin-cos QG-OBB Head 的实现方案。
```
