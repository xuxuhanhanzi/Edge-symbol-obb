# Export grayscale OBB weights to ONNX.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
import ultralytics.utils.checks as checks


# ============================================================
# 单通道灰度模型导出辅助
# ============================================================

# 关闭 AMP 检查，避免官方默认 3 通道测试影响导出
checks.check_amp = lambda *args, **kwargs: True

# 修复 AutoBackend warmup 默认 3 通道问题
original_autobackend_warmup = AutoBackend.warmup


def custom_autobackend_warmup(self, imgsz=(1, 1, 256, 256)):
    try:
        expected_c = next(self.model.parameters()).shape[1]
        if isinstance(imgsz, tuple) and len(imgsz) == 4:
            imgsz = (imgsz[0], expected_c, imgsz[2], imgsz[3])
    except Exception:
        pass

    return original_autobackend_warmup(self, imgsz=imgsz)


AutoBackend.warmup = custom_autobackend_warmup


if __name__ == "__main__":
    # 改成你的 best.pt 实际路径
    model_path = "runs/obb/qr_obb_rv11067/weights/best.pt"

    model = YOLO(model_path)

    result = model.export(
        format="onnx",
        imgsz=256,
        batch=1,
        opset=19,
        simplify=False,
        dynamic=False,
        half=False,
        device=0,
    )

    print("ONNX export success:")
    print(result)
