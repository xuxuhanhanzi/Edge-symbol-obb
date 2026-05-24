"""One-epoch smoke training for the RV1106 grayscale OBB baseline.

This script checks that model build, dataloading, OBB loss, validation, and
run artifact creation work before launching a long baseline training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import yaml_load
import ultralytics.utils.checks as checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/rv1106/yolov8n_obb_rv1106_m2.yaml", help="Model yaml path.")
    parser.add_argument("--data", default="datasets/industrial_symbol.yaml", help="Dataset yaml path.")
    parser.add_argument("--name", default="rv1106_smoke", help="Run name under runs/obb.")
    parser.add_argument("--epochs", type=int, default=1, help="Smoke training epochs.")
    parser.add_argument("--imgsz", type=int, default=256, help="Input image size.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size. Use 192 or 256 on RTX 5090 if stable.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--device", default=0, help="CUDA device id or cpu.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP after the smoke path is known to work.")
    parser.add_argument("--cache", default=False, help="Ultralytics cache argument: False, ram, or disk.")
    parser.add_argument("--angle", type=float, default=5.0, help="OBB angle loss gain.")
    parser.add_argument("--qg-angle-align", type=float, default=0.25, help="QG unit-cycle alignment loss gain.")
    parser.add_argument("--qg-angle-unit", type=float, default=0.05, help="QG vector norm regularization loss gain.")
    return parser.parse_args()


def install_grayscale_patches(model_spec: str | Path | None = None) -> None:
    """Patch train/val preprocessing and warmup for one-channel grayscale models."""
    checks.check_amp = lambda *args, **kwargs: True

    original_train_preprocess = OBBTrainer.preprocess_batch
    original_val_preprocess = OBBValidator.preprocess
    original_autobackend_warmup = AutoBackend.warmup

    def expected_input_channels_from_yaml(spec):
        if not spec:
            return None
        path = Path(spec)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            return None
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            return None
        try:
            ch = yaml_load(path).get("ch", 3)
            return int(ch[0] if isinstance(ch, (list, tuple)) else ch)
        except Exception:
            return None

    configured_input_c = expected_input_channels_from_yaml(model_spec)

    def expected_input_channels(model, default=None):
        """Infer the model input channel count from the first parameter tensor."""
        for candidate in (getattr(model, "model", None), model):
            try:
                return next(candidate.parameters()).shape[1]
            except Exception:
                continue
        return default

    def custom_train_preprocess_batch(self, batch):
        batch = original_train_preprocess(self, batch)
        expected_c = expected_input_channels(
            getattr(self, "model", None), configured_input_c or batch["img"].shape[1]
        )
        if expected_c == 1 and batch["img"].shape[1] == 3:
            batch["img"] = batch["img"].mean(dim=1, keepdim=True)
        return batch

    def custom_val_preprocess(self, batch):
        batch = original_val_preprocess(self, batch)
        expected_c = expected_input_channels(
            getattr(self, "model", None), configured_input_c or batch["img"].shape[1]
        )
        if expected_c == 1 and batch["img"].shape[1] == 3:
            batch["img"] = batch["img"].mean(dim=1, keepdim=True)
        return batch

    def custom_autobackend_warmup(self, imgsz=(1, 1, 256, 256)):
        try:
            expected_c = expected_input_channels(self.model)
            if isinstance(imgsz, tuple) and len(imgsz) == 4:
                imgsz = (imgsz[0], expected_c, imgsz[2], imgsz[3])
        except Exception:
            pass
        return original_autobackend_warmup(self, imgsz=imgsz)

    OBBTrainer.preprocess_batch = custom_train_preprocess_batch
    OBBValidator.preprocess = custom_val_preprocess
    AutoBackend.warmup = custom_autobackend_warmup


def normalize_cache_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"false", "0", "none", "no"}:
        return False
    if text in {"true", "1", "yes"}:
        return True
    return text


def main() -> int:
    args = parse_args()
    install_grayscale_patches(args.model)

    model = YOLO(args.model)
    results = model.train(
        task="obb",
        data=args.data,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        cache=normalize_cache_arg(args.cache),
        angle=args.angle,
        qg_angle_align=args.qg_angle_align,
        qg_angle_unit=args.qg_angle_unit,
        save=True,
        save_period=-1,
        patience=0,
        val=True,
        plots=True,
        exist_ok=True,
        deterministic=False,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        cos_lr=True,
        warmup_epochs=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=180.0,
        perspective=0.001,
        scale=0.5,
        shear=0.0,
        erasing=0.0,
        mosaic=0.3,
        close_mosaic=0,
    )

    print(f"smoke_train_done save_dir={results.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
