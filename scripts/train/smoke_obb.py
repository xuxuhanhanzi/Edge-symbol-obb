"""Minimal OBB smoke checks for stage 0."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO, __version__


def main():
    print("import_ok")
    print(f"ultralytics_version={__version__}")

    checks = {
        "official_arch_reference": "configs/baseline/yolov8n_obb_official_arch.yaml",
        "rv1106_m2": "configs/rv1106/yolov8n_obb_rv1106_m2.yaml",
        "rv1106_qg_sincos": "configs/rv1106/yolov8n_obb_rv1106_qg_sincos.yaml",
    }
    for name, cfg in checks.items():
        model = YOLO(cfg)
        print(f"{name}: task={model.task}, model_build_ok")


if __name__ == "__main__":
    main()
