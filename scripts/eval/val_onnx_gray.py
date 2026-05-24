"""Validate a grayscale OBB ONNX model with Ultralytics validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.tal import dist2rbox, make_anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="runs/obb/rv1106_m2_baseline_e100_b256/weights/best.onnx",
        help="ONNX model path.",
    )
    parser.add_argument("--data", default="datasets/industrial_symbol.yaml", help="Dataset yaml path.")
    parser.add_argument("--imgsz", type=int, default=256, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size for ONNX.")
    parser.add_argument("--device", default=0, help="CUDA device id or cpu.")
    parser.add_argument("--split", default="val", help="Dataset split to validate.")
    parser.add_argument("--workers", type=int, default=0, help="Validation dataloader workers.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--plots", action="store_true", help="Save validation plots.")
    return parser.parse_args()


def install_grayscale_val_patch() -> None:
    """Patch validation preprocessing and warmup for one-channel ONNX models."""
    original_val_preprocess = OBBValidator.preprocess
    original_autobackend_warmup = AutoBackend.warmup
    original_postprocess = OBBValidator.postprocess

    def custom_val_preprocess(self, batch):
        batch = original_val_preprocess(self, batch)
        if batch["img"].shape[1] == 3:
            batch["img"] = batch["img"].mean(dim=1, keepdim=True)
        return batch

    def custom_autobackend_warmup(self, imgsz=(1, 1, 256, 256)):
        if isinstance(imgsz, tuple) and len(imgsz) == 4:
            imgsz = (imgsz[0], 1, imgsz[2], imgsz[3])
        return original_autobackend_warmup(self, imgsz=imgsz)

    def custom_postprocess(self, preds):
        if is_obb_4head_output(preds, self.nc):
            preds = decode_obb_4head_output(preds, self.nc, self.device, self.args.imgsz)
        return original_postprocess(self, preds)

    OBBValidator.preprocess = custom_val_preprocess
    AutoBackend.warmup = custom_autobackend_warmup
    OBBValidator.postprocess = custom_postprocess


def is_obb_4head_output(preds, nc: int) -> bool:
    if not isinstance(preds, (list, tuple)) or len(preds) not in (4, 5):
        return False

    feature_maps = [p for p in preds if hasattr(p, "ndim") and p.ndim == 4]
    angle_maps = [p for p in preds if hasattr(p, "ndim") and p.ndim in (2, 3)]
    if len(feature_maps) != 3 or len(angle_maps) not in (1, 2):
        return False

    return all(p.shape[1] >= nc + 4 for p in feature_maps)


def decode_angle_output(angle: torch.Tensor, batch: int, anchors: int) -> torch.Tensor:
    """Decode scalar theta or QG [sin(2theta), cos(2theta)] ONNX angle output."""
    if angle.shape[-1] != anchors:
        angle = angle.reshape(batch, -1, anchors)
    if angle.shape[-1] != anchors:
        raise ValueError(f"Angle output anchors={angle.shape[-1]} do not match box anchors={anchors}")
    if angle.shape[1] == 1:
        return angle
    if angle.shape[1] == 2:
        return 0.5 * torch.atan2(angle[:, 0:1], angle[:, 1:2])
    raise ValueError(f"Unsupported angle output shape={tuple(angle.shape)}")


def decode_obb_4head_output(preds, nc: int, device: torch.device, imgsz: int) -> torch.Tensor:
    """Convert 4-head grayscale ONNX OBB output to Ultralytics NMS input."""
    tensors = [p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in preds]
    tensors = [p.to(device) for p in tensors]

    feature_maps = sorted([p for p in tensors if p.ndim == 4], key=lambda x: x.shape[-1], reverse=True)
    angle_tensors = [p for p in tensors if p.ndim in (2, 3)]

    channels = feature_maps[0].shape[1]
    reg_max = (channels - nc - 1) // 4 if channels >= nc + 5 else (channels - nc) // 4
    if reg_max <= 0:
        raise ValueError(f"Unable to infer reg_max from ONNX output channels={channels}, nc={nc}")

    raw = torch.cat([p[:, : reg_max * 4 + nc].reshape(p.shape[0], reg_max * 4 + nc, -1) for p in feature_maps], 2)
    box_raw, cls = raw.split((reg_max * 4, nc), 1)

    batch = box_raw.shape[0]
    dist = box_raw.view(batch, 4, reg_max, -1).softmax(2)
    proj = torch.arange(reg_max, dtype=dist.dtype, device=dist.device).view(1, 1, reg_max, 1)
    dist = (dist * proj).sum(2)

    strides = torch.tensor([imgsz / p.shape[-1] for p in feature_maps], dtype=dist.dtype, device=dist.device)
    anchors, stride_tensor = make_anchors(feature_maps, strides, 0.5)
    anchors = anchors.transpose(0, 1).unsqueeze(0)
    stride_tensor = stride_tensor.transpose(0, 1).unsqueeze(0)

    if len(angle_tensors) == 1:
        angle = decode_angle_output(angle_tensors[0], batch, dist.shape[-1])
    elif len(angle_tensors) == 2:
        angle = torch.cat([a.reshape(batch, 1, -1) for a in angle_tensors], 1)
        angle = decode_angle_output(angle, batch, dist.shape[-1])
    else:
        raise ValueError(f"Unsupported number of angle outputs: {len(angle_tensors)}")

    dbox = dist2rbox(dist, angle, anchors, dim=1) * stride_tensor
    cls = cls.clamp(0, 1) if cls.min() >= 0 and cls.max() <= 1 else cls.sigmoid()
    return torch.cat((dbox, cls, angle), 1)


def main() -> int:
    args = parse_args()
    install_grayscale_val_patch()

    weights = Path(args.weights)
    if not weights.exists():
        pt_hint = weights.with_suffix(".pt")
        raise FileNotFoundError(
            f"ONNX model not found: {weights}\n"
            f"Export it first from the matching PyTorch weights, for example:\n"
            f"python -B scripts/export/export_gray_obb_onnx.py --weights {pt_hint} --imgsz {args.imgsz}"
        )

    print(f"val_weights={weights}")
    print(f"val_data={args.data}")
    print(f"val_imgsz={args.imgsz}")

    model = YOLO(str(weights))
    metrics = model.val(
        task="obb",
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        plots=args.plots,
    )

    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
