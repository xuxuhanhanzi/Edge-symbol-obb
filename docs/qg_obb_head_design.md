# QG-OBB Head Design Audit

## 1. Purpose

This document audits the current OBB head, loss, ONNX export, ONNX validation and RKNN postprocess interfaces before implementing the next model-improvement stage.

The immediate goal is a minimal QG-OBB Head experiment for the RV1106 grayscale OBB baseline. The first implementation should focus on a low-risk sin-cos angle branch, not a full Gaussian or Cholesky-style head.

Current fixed baseline:

| Item | Value |
|---|---|
| Model YAML | `configs/rv1106/yolov8n_obb_rv1106_m2.yaml` |
| Data YAML | `datasets/industrial_symbol.yaml` |
| Task | grayscale single-channel OBB detection |
| Input | `1 x 256 x 256` |
| Classes | 15 |
| PyTorch FP32 | mAP50 `0.991`, mAP50-95 `0.960` |
| ONNX FP32 | mAP50 `0.990494`, mAP50-95 `0.959600` |
| RKNN INT8 | continuous AP@0.5 `0.9846` |

The current baseline must remain reproducible. QG-OBB changes should be introduced through a new model config and compatibility code paths.

## 2. Files Audited

| File | Role |
|---|---|
| `configs/rv1106/yolov8n_obb_rv1106_m2.yaml` | Current lightweight RV1106 OBB baseline |
| `ultralytics/nn/modules/head.py` | Current modified `OBB` head and `onnx_4head` export branch |
| `ultralytics/utils/loss.py` | Current `v8OBBLoss`, rotated assigner, bbox decode and angle loss |
| `ultralytics/utils/tal.py` | `dist2rbox` decode and rotated assigner angle similarity |
| `ultralytics/utils/ops.py` | OBB polygon conversion and post-NMS box regularization |
| `ultralytics/engine/exporter.py` | Forces OBB ONNX export into `onnx_4head` format |
| `scripts/export/export_gray_obb_onnx.py` | Grayscale ONNX export entry |
| `scripts/eval/val_onnx_gray.py` | Grayscale ONNX validation and 4-head decode |
| `scripts/deploy_rv1106/convert_eval_rknn.py` | RKNN conversion, DFL decode, OBB postprocess and AP calculation |

## 3. Current Baseline Interface

### 3.1 Model Config

`configs/rv1106/yolov8n_obb_rv1106_m2.yaml` defines:

```text
nc: 15
ch: 1
head last layer: [[15, 18, 21], 1, OBB, [15, 1]]
```

The final `OBB` arguments are `[nc, ne]`, so the current angle branch has `ne=1`.

For `imgsz=256`, the expected detection scales are:

| Level | Stride | Grid | Anchors |
|---|---:|---:|---:|
| P3 | 8 | `32 x 32` | 1024 |
| P4 | 16 | `16 x 16` | 256 |
| P5 | 32 | `8 x 8` | 64 |
| Total | | | 1344 |

The neck/head channels entering `OBB` are expected to follow the RV1106 channel template:

| Level | Channels |
|---|---:|
| P3 | 80 |
| P4 | 144 |
| P5 | 208 |

### 3.2 Current OBB Head

The current `OBB` head in `ultralytics/nn/modules/head.py` is already a local modified implementation:

- `reg_max = 8`
- `self.no = nc + reg_max * 4 = 15 + 32 = 47`
- bbox branch output per level: `32` channels
- class branch output per level: `15` channels
- angle branch output: `ne=1` channel
- `onnx_4head` per-level output adds an objectness-like channel, so each feature map has `32 + 15 + 1 = 48` channels

Training output:

```text
feats: list of 3 tensors
  P3: [B, 47, 32, 32]
  P4: [B, 47, 16, 16]
  P5: [B, 47, 8, 8]
angle:
  [B, 1, 1344]
```

Normal inference output before Ultralytics NMS is effectively:

```text
[B, 4 + nc + 1, 1344] = [B, 20, 1344]
```

where the last channel is the scalar angle.

ONNX 4-head export output:

```text
out_p3: [B, 48, 32, 32]
out_p4: [B, 48, 16, 16]
out_p5: [B, 48, 8, 8]
angle:  [B, 1, 1344]
```

The class branch is sigmoid-activated inside the `onnx_4head` branch. The extra objectness-like channel is `clamp(sum(cls), 0, 1)` and is present for RKNN-style postprocess compatibility, but current ONNX validator ignores it when rebuilding Ultralytics-format predictions.

### 3.3 Current Angle Semantics

The current angle branch emits a raw scalar. There is no sigmoid, tanh, clamp or explicit range mapping in the active `OBB.forward()` path.

The scalar is used directly as radians:

```text
theta = angle_raw
```

`dist2rbox()` then uses:

```text
cos(theta), sin(theta)
```

Dataset polygons are converted to `xywhr` using `cv2.minAreaRect`; the utility docstring says the converted rotation is in radians from `0` to `pi/2`. Later display/prediction code regularizes predicted boxes back into a canonical OBB representation after NMS.

Important implication:

- Training and inference currently tolerate raw predicted angles outside `[0, pi/2]` because `sin/cos` are periodic.
- The custom angle loss explicitly treats angle as periodic.
- ONNX and RKNN postprocess assume the angle output is already the final scalar theta.

### 3.4 Current Loss Interface

`v8OBBLoss.__call__()` receives:

```text
preds -> (feats, pred_angle)
pred_angle before permute: [B, 1, 1344]
pred_angle after permute:  [B, 1344, 1]
```

`bbox_decode()` converts DFL distances and scalar angle into:

```text
pred_bboxes: [B, 1344, 5] = xywhr
```

The current angle loss:

```text
delta = pred_theta - target_theta
delta_wrapped = delta - round(delta / pi) * pi
angle_loss = sin(2 * delta_wrapped)^2
```

This is a pi-periodic loss. It matches the OBB equivalence that a box angle offset by pi represents the same physical orientation.

The rotated assigner also includes angle similarity:

```text
angle_sim = abs(cos(gt_angle - pred_angle))
score = probiou * angle_sim
```

So angle quality affects positive assignment, not only the final loss.

### 3.5 Current ONNX Validation Decode

`scripts/eval/val_onnx_gray.py` identifies current 4-head ONNX output as:

```text
3 feature maps with ndim == 4
1 angle tensor with ndim == 3
```

It infers `reg_max`, performs DFL softmax/projection on CPU/GPU through PyTorch, rebuilds anchors, then calls:

```text
dist2rbox(dist, angle, anchors, dim=1)
```

Finally it returns:

```text
torch.cat((dbox, cls, angle), 1)
```

This means the ONNX validator must still receive a scalar angle after any QG decode.

### 3.6 Current RKNN Decode

`scripts/deploy_rv1106/convert_eval_rknn.py` is hardcoded to the current baseline:

```text
NUM_CLASSES = 15
REG_MAX = 8
feature output channels accepted: 47 or 48
angle output length: 1344
angle offsets:
  32 grid -> [0, 1024)
  16 grid -> [1024, 1280)
  8 grid  -> [1280, 1344)
```

RKNN postprocess does:

```text
box_feat -> DFL decode -> l,t,r,b
angle = angle_slice[idx]
cos(angle), sin(angle)
center decode
OBB NMS
continuous AP@0.5
```

It currently assumes the angle tensor is one scalar per anchor.

## 4. QG-OBB Minimal Sin-Cos Design

### 4.1 Recommendation: Double-Angle Sin-Cos

The project plan says to try:

```text
angle scalar -> sin(theta), cos(theta)
decode -> atan2(sin, cos)
```

For OBB, the current loss and assigner already treat orientation as pi-periodic. A safer minimal design is therefore:

```text
angle scalar theta -> sin(2 * theta), cos(2 * theta)
decode theta -> 0.5 * atan2(sin2, cos2)
```

Reason:

- `theta` and `theta + pi` should be equivalent for an oriented rectangle.
- `sin(theta), cos(theta)` is 2pi-periodic.
- `sin(2theta), cos(2theta)` is pi-periodic and matches the current `sin(2 * delta)^2` loss.

In this document, "sin-cos QG-OBB" refers to the double-angle vector:

```text
angle_vec = [sin2theta, cos2theta]
```

If a strict single-angle experiment is needed later, it can be added as a separate ablation. It should not be the first implementation.

### 4.2 Output Contract

The QG minimal head should preserve all downstream contracts by decoding the vector back to a scalar theta before existing OBB decode and NMS.

Training output:

```text
feats: unchanged list of 3 tensors
  P3: [B, 47, 32, 32]
  P4: [B, 47, 16, 16]
  P5: [B, 47, 8, 8]
angle_vec:
  [B, 2, 1344]
```

Decoded scalar used inside loss:

```text
angle_theta:
  [B, 1344, 1]
```

ONNX 4-head output:

```text
out_p3:    [B, 48, 32, 32]
out_p4:    [B, 48, 16, 16]
out_p5:    [B, 48, 8, 8]
angle_vec: [B, 2, 1344]
```

ONNX/RKNN postprocess must decode `angle_vec` to:

```text
angle_theta: [B, 1, 1344]
```

before calling the existing `dist2rbox` or equivalent CPU postprocess.

### 4.3 Decode Function

Use a small shared logic pattern in each place that needs scalar theta:

```text
if angle channels == 1:
    theta = angle
elif angle channels == 2:
    vec = angle / max(norm(angle), eps)
    sin2, cos2 = vec[..., 0:1], vec[..., 1:2]
    theta = 0.5 * atan2(sin2, cos2)
else:
    error
```

Keep the channel order explicit:

```text
channel 0: sin(2 * theta)
channel 1: cos(2 * theta)
```

For PyTorch tensors, use `torch.atan2`.

For RKNN CPU postprocess, use `math.atan2` or `np.arctan2`.

The ONNX graph used for RKNN should output the raw vector. This keeps `atan2` out of the exported model and avoids RKNN operator support risk.

## 5. Required Code Changes

### 5.1 Model Config

Add a new config instead of modifying the baseline:

```text
configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml
```

Start from `yolov8n_obb_rv1106_m2.yaml` and change only the final OBB arg:

```text
- [[15, 18, 21], 1, OBB, [15, 2]]
```

This keeps architecture, channels, DFL, class branch and box branch identical.

### 5.2 `ultralytics/nn/modules/head.py`

Update the existing `OBB` class with a compatibility path:

- Preserve current behavior when `self.ne == 1`.
- Enable sin-cos behavior when `self.ne == 2`.
- Add a helper such as `_decode_angle(angle)` that returns scalar theta.
- In normal training, return raw `angle_vec`; the loss will decode it.
- In normal inference, set `self.angle` to decoded scalar theta before `super().forward(x)`.
- In `onnx_4head`, return raw `angle_vec` as the fourth ONNX output, not decoded theta.

Do not change:

- `reg_max = 8`
- bbox branch channel count
- class branch channel count
- per-level ONNX feature map outputs

### 5.3 `ultralytics/utils/loss.py`

Update `v8OBBLoss` to accept both scalar and vector angle predictions.

Required behavior:

- If `pred_angle` has one channel, run the current baseline path unchanged.
- If `pred_angle` has two channels, decode `theta` for `bbox_decode`, assigner and existing angle loss.
- Add a vector alignment term for sin-cos predictions.

Suggested vector target:

```text
target_vec = [sin(2 * target_theta), cos(2 * target_theta)]
```

Suggested vector loss:

```text
pred_unit = normalize(pred_vec)
loss_vec = 1 - dot(pred_unit, target_vec)
loss_unit = (norm(pred_vec) - 1)^2
loss_angle = current_periodic_theta_loss + alpha * loss_vec + beta * loss_unit
```

Initial coefficients:

```text
alpha = 0.25
beta = 0.05
```

These should be treated as first-pass values for smoke/quick experiments, not final tuned hyperparameters.

Keep the externally reported loss names unchanged for the first experiment:

```text
box_loss, cls_loss, dfl_loss, angle_loss
```

The vector alignment and unit-cycle penalties can be included inside `angle_loss`. If diagnosis becomes difficult, split them into separate logged losses later.

### 5.4 `scripts/eval/val_onnx_gray.py`

Update 4-head decode:

- Accept angle tensor with shape `[B, 1, 1344]` or `[B, 2, 1344]`.
- Decode two-channel angle to scalar theta before `dist2rbox`.
- Keep the returned validator tensor shape unchanged:

```text
[B, 4 + nc + 1, 1344]
```

This keeps Ultralytics OBB NMS and metrics unchanged.

### 5.5 `scripts/deploy_rv1106/convert_eval_rknn.py`

Update RKNN postprocess:

- Accept angle output length `1344` for scalar baseline.
- Accept angle output length `2688` or shape `[1, 2, 1344]` for QG sin-cos.
- Decode per-anchor theta on CPU.
- Keep the final detection format unchanged:

```text
[cls_id, cx, cy, w, h, angle, score]
```

Avoid adding `atan2` into the RKNN graph. The RKNN model should output raw sin-cos channels and let CPU postprocess decode them.

### 5.6 Export Entry

`scripts/export/export_gray_obb_onnx.py` should not need structural changes. The exporter already forces OBB ONNX export into `onnx_4head` through `ultralytics/engine/exporter.py`.

Still, after QG changes, export must be checked because the fourth ONNX output changes from:

```text
[B, 1, 1344]
```

to:

```text
[B, 2, 1344]
```

## 6. Experiment Plan

### 6.1 Shape Test

Before training, run a local shape check:

```text
model: configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml
input: [1, 1, 256, 256]
expected training angle output: [1, 2, 1344]
expected ONNX 4-head angle output: [1, 2, 1344]
```

The shape test should explicitly verify:

- scalar baseline model still builds
- QG model builds
- training forward returns `(feats, angle_vec)`
- decoded theta has shape `[B, 1344, 1]` in loss
- ONNX validation decode emits `[B, 20, 1344]`

### 6.2 Smoke Train

Use a separate run name:

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --name rv1106_qg_sincos_smoke \
  --epochs 1 \
  --batch 128 \
  --workers 8
```

Smoke pass criteria:

- no shape errors
- no NaN loss
- `angle_loss` is finite
- validation path completes

### 6.3 Quick Train

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --name rv1106_qg_sincos_e20_b512 \
  --epochs 20 \
  --batch 512 \
  --workers 16
```

Quick pass criteria:

| Metric | Requirement |
|---|---:|
| mAP50 | no severe collapse |
| mAP50-95 | near baseline trend |
| angle_loss | finite and decreasing/stable |

### 6.4 Full Train

```bash
python -B scripts/train/train_rv1106_smoke.py \
  --model configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml \
  --name rv1106_qg_sincos_e100_b512 \
  --epochs 100 \
  --batch 512 \
  --workers 16
```

Full pass criteria:

| Metric | Requirement |
|---|---:|
| PyTorch mAP50 | `>= baseline - 0.003` |
| PyTorch mAP50-95 | `>= baseline - 0.005` |
| Params/FLOPs | no obvious increase |

### 6.5 Export and Deployment Validation

Every QG candidate that survives quick/full training must repeat:

```text
PyTorch val
-> ONNX export
-> ONNX val
-> RKNN INT8 conversion
-> RKNN INT8 val
```

Acceptance gates:

| Gate | Requirement |
|---|---:|
| ONNX export | PASS |
| ONNX mAP drop vs PyTorch | `<= 0.005` |
| RKNN INT8 AP@0.5 drop vs ONNX | `<= 0.03` |

Simulator latency remains a pipeline-debug number only. It must not be used as final RV1106 device latency.

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Angle vector norm collapses near zero | `atan2` unstable, noisy gradients | normalize with epsilon, add unit-cycle penalty |
| Single-angle sin-cos conflicts with OBB pi-periodicity | duplicate orientations learned inconsistently | use double-angle `[sin(2theta), cos(2theta)]` first |
| Loss balance hurts assignment early | mAP collapses in smoke/quick train | keep vector terms small and preserve existing theta loss |
| ONNX validator expects scalar angle | validation shape error | decode vector to scalar before `dist2rbox` |
| RKNN output order/shape differs after conversion | postprocess reads wrong channels | add explicit shape detection and debug print for outputs |
| QG code accidentally changes baseline behavior | baseline no longer reproducible | branch on `self.ne`, keep `ne=1` path unchanged |

## 8. Implementation Checklist

Do not start full training until all items below pass.

1. Add `configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml`.
2. Update `OBB` head with `ne=1` compatibility and `ne=2` vector decode.
3. Update `v8OBBLoss` to decode two-channel angle vectors and add vector alignment.
4. Update ONNX validation decode for `[B, 2, 1344]` angle output.
5. Update RKNN postprocess for scalar and two-channel angle outputs.
6. Run scalar baseline build smoke to ensure compatibility.
7. Run QG shape test.
8. Run QG 1-epoch smoke train.
9. Export QG ONNX and validate output shapes.
10. Only then proceed to 20-epoch quick training.

## 9. Current Design Decision

The next implementation should use the existing `OBB` class with an `ne == 2` compatibility branch, rather than introducing a new head class immediately.

Reason:

- the YAML already supports passing `ne` as the second `OBB` argument;
- adding a new head class would require parser/import plumbing and increases first-pass risk;
- branching inside `OBB` preserves the current baseline path when `ne == 1`;
- the experiment remains easy to ablate by changing only the final YAML line.

If the minimal sin-cos experiment passes all gates and becomes the long-term direction, it can later be refactored into a named `QGOBB` head for clarity.
