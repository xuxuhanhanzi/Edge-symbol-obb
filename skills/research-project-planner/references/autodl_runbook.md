# AutoDL Runbook

Use this reference for AutoDL-based experiments.

## Required Environment Record

Run before important experiments:

```bash
pwd
python - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

Record:

- workspace path
- dataset path
- conda env
- GPU
- Python
- PyTorch
- CUDA
- commit or changed-file list

## Standard Training Flow

```text
smoke train -> quick train -> full train -> PyTorch val
```

## Standard Deployment Flow

```text
PyTorch val -> ONNX export -> ONNX val -> target build -> target eval
```

For RKNN-style deployment, record:

```bash
python - <<'PY'
import numpy, onnx, cv2, importlib.metadata as md
from rknn.api import RKNN
print("numpy", numpy.__version__)
print("onnx", onnx.__version__)
print("cv2", cv2.__version__)
print("rknn-toolkit2", md.version("rknn-toolkit2"))
rknn = RKNN(verbose=False)
print("RKNN init ok")
rknn.release()
PY
```

## Path Checklist

Record:

| Item | Example |
|---|---|
| Code root | `/root/ultralytics_yolov8-main/ultralytics_yolov8-main` |
| Dataset | `/root/autodl-tmp/yolo_dataset_gray` |
| Run output | `runs/obb/<run_name>` |
| Exported ONNX | `runs/obb/<run_name>/weights/best.onnx` |
| Deployment model | `runs/obb/<run_name>/weights/*.rknn` |
| Deployment logs | `rknn_logs/*.txt` |
| Debug artifacts | `artifacts/local/<name>` |

## Common Failure Record

When a run fails, record:

- exact command
- exact traceback or stuck progress
- environment versions
- whether build-only works
- whether eval-only works
- whether failure is model graph, dependency, memory, or postprocess
- fix attempt and result
