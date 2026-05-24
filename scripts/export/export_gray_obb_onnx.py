"""Export grayscale OBB weights to ONNX with command-line arguments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn

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


def split_qg_angle_head_for_rknn(model) -> bool:
    """Convert legacy QG angle Conv2d(out=2) heads to two Conv2d(out=1) heads for RKNN export."""
    head = model.model[-1]
    if getattr(head, "ne", None) != 2 or hasattr(head, "cv4_sin") or not hasattr(head, "cv4"):
        return False

    trunks = nn.ModuleList()
    sin_heads = nn.ModuleList()
    cos_heads = nn.ModuleList()

    for seq in head.cv4:
        layers = list(seq.children())
        if not layers or not isinstance(layers[-1], nn.Conv2d) or layers[-1].out_channels != 2:
            return False

        last = layers[-1]
        sin = nn.Conv2d(
            last.in_channels,
            1,
            last.kernel_size,
            last.stride,
            last.padding,
            last.dilation,
            last.groups,
            bias=last.bias is not None,
            padding_mode=last.padding_mode,
            device=last.weight.device,
            dtype=last.weight.dtype,
        )
        cos = nn.Conv2d(
            last.in_channels,
            1,
            last.kernel_size,
            last.stride,
            last.padding,
            last.dilation,
            last.groups,
            bias=last.bias is not None,
            padding_mode=last.padding_mode,
            device=last.weight.device,
            dtype=last.weight.dtype,
        )
        sin.weight.data.copy_(last.weight.data[0:1])
        cos.weight.data.copy_(last.weight.data[1:2])
        if last.bias is not None:
            sin.bias.data.copy_(last.bias.data[0:1])
            cos.bias.data.copy_(last.bias.data[1:2])

        trunks.append(nn.Sequential(*layers[:-1]))
        sin_heads.append(sin)
        cos_heads.append(cos)

    head.cv4 = trunks
    head.cv4_sin = sin_heads
    head.cv4_cos = cos_heads
    return True


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
    if split_qg_angle_head_for_rknn(model.model):
        print("qg_angle_export=split_sin_cos_heads")
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
