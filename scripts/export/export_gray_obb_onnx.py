"""Export grayscale OBB weights to ONNX with command-line arguments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
import ultralytics.utils.checks as checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="runs/obb/rv1106_m2_baseline_e100_b256/weights/best.pt",
        help="PyTorch .pt weights to export.",
    )
    parser.add_argument("--imgsz", type=int, default=256, help="Export image size.")
    parser.add_argument("--batch", type=int, default=1, help="Static export batch size.")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset version.")
    parser.add_argument("--device", default=0, help="CUDA device id or cpu.")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX.")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic axes.")
    parser.add_argument("--simplify", action="store_true", help="Run ONNX simplifier.")
    return parser.parse_args()


def install_grayscale_export_patch() -> None:
    """Patch AutoBackend warmup to respect one-channel grayscale models."""
    checks.check_amp = lambda *args, **kwargs: True

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


def main() -> int:
    args = parse_args()
    install_grayscale_export_patch()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"export_weights={weights}")
    print(f"export_imgsz={args.imgsz}")
    print(f"export_opset={args.opset}")

    model = YOLO(str(weights))
    result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device,
    )

    print(f"onnx_export={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
