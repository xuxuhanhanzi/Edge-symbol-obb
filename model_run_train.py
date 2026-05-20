
# model_run_train_complete.py
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.nn.autobackend import AutoBackend
import ultralytics.utils.checks as checks

# ===================================================================
# 🛡️ 单通道灰度支持：拦截官方预处理
# ===================================================================

# 0. 禁用 AMP 自动检查（训练可选开启）
# 官方 AMP 测试容易报错，先关闭，再在训练参数中安全开启
checks.check_amp = lambda *args, **kwargs: True

# 1. 备份官方函数
original_train_preprocess = OBBTrainer.preprocess_batch
original_val_preprocess = OBBValidator.preprocess
original_autobackend_warmup = AutoBackend.warmup

# 2. 训练预处理拦截：保证单通道
def custom_train_preprocess_batch(self, batch):
    batch = original_train_preprocess(self, batch)
    if batch["img"].shape[1] == 3:
        # 用平均值生成灰度图，而不是直接裁通道
        batch["img"] = batch["img"].mean(dim=1, keepdim=True)
    return batch

# 3. 验证预处理拦截
def custom_val_preprocess(self, batch):
    batch = original_val_preprocess(self, batch)
    if batch["img"].shape[1] == 3:
        batch["img"] = batch["img"].mean(dim=1, keepdim=True)
    return batch

# 4. 智能 Warmup 拦截：动态感知模型通道数
def custom_autobackend_warmup(self, imgsz=(1, 1, 256, 256)):
    try:
        expected_c = next(self.model.parameters()).shape[1]
        if isinstance(imgsz, tuple) and len(imgsz) == 4:
            imgsz = (imgsz[0], expected_c, imgsz[2], imgsz[3])
    except Exception:
        pass
    return original_autobackend_warmup(self, imgsz=imgsz)

# 5. 全局替换
OBBTrainer.preprocess_batch = custom_train_preprocess_batch
OBBValidator.preprocess = custom_val_preprocess
AutoBackend.warmup = custom_autobackend_warmup
# ===================================================================

if __name__ == '__main__':
    # 使用我们修改过的 YAML 配置（带高分辨率偏置）
    model = YOLO("ultralytics/cfg/models/v8/M6.yaml")  # 请确保路径正确


    results = model.train(
        task="obb",
        data="cfg/datasets/My_project.yaml",  # 指向你数据集 YAML
        name="qr_obb_rv1106",   # 日志名称



        # --- 训练参数 ---
        save_period=20,
        patience=50,
        epochs=100,             # 正式训练
        imgsz=256,              # 输入尺寸
        batch=128,               # 初次 sanity check 用小 batch
        cache=False,            # 初次训练禁用高速缓存
        workers=4,
        device=0,
        amp=False,              # 初次训练关闭 AMP
        deterministic=False,
        

        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        cos_lr=True,
        warmup_epochs=3.0,

        # --- NPU/RV1106 参数 ---
        # reg_max=8,             # 可根据硬件内存设置

        # --- 数据增强控制 ---
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
        mosaic=0.3,             # 先降低 mosaic 强度
        close_mosaic=20,
    )

    print("训练完成，结果保存在:", results.save_dir)
