from ultralytics import YOLO
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.autobackend import AutoBackend

# 备份原函数
original_val_preprocess = OBBValidator.preprocess
original_autobackend_warmup = AutoBackend.warmup


def custom_val_preprocess(self, batch):
    batch = original_val_preprocess(self, batch)
    if batch["img"].shape[1] == 3:
        batch["img"] = batch["img"].mean(dim=1, keepdim=True)
    return batch


def custom_autobackend_warmup(self, imgsz=(1, 1, 256, 256)):
    if isinstance(imgsz, tuple) and len(imgsz) == 4:
        imgsz = (imgsz[0], 1, imgsz[2], imgsz[3])
    return original_autobackend_warmup(self, imgsz=imgsz)


OBBValidator.preprocess = custom_val_preprocess
AutoBackend.warmup = custom_autobackend_warmup


if __name__ == "__main__":
    model = YOLO("runs/obb/qr_obb_rv11065/weights/best.onnx")

    metrics = model.val(
        task="obb",
        data="/root/ultralytics_yolov8-main/ultralytics/cfg/datasets/My_project.yaml",
        imgsz=256,
        batch=1,
        device=0,
    )

    print(metrics)