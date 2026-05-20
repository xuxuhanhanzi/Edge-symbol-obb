import torch
from ultralytics import YOLO
import os
import math

# ================= 配置区域 =================
# 请确保这些路径正确
BASE_DIR = "/home/lab/桌面/YOLOV8/ultralytics_yolov8-main"
DATA_YAML = os.path.join(BASE_DIR, "ultralytics/cfg/datasets/qr_barcode_test.yaml")

# 运行目录 (runs)
RUNS_DIR = os.path.join(BASE_DIR, "runs/obb")
PROJECT_NAME = "yolov8-obb-width0.155"

# 1. 预训练/原始训练配置
PRETRAIN_WEIGHTS = os.path.join(RUNS_DIR, PROJECT_NAME, "weights/yolo_v8_obb_vanillanet_aug.pt")

# 2. 约束训练(稀疏训练)配置
SPARSE_PROJECT_NAME = "Constraint_Training"
SPARSE_INPUT_MODEL = os.path.join(RUNS_DIR, PROJECT_NAME, "weights/best.pt")
SPARSE_OUTPUT_DIR = os.path.join(RUNS_DIR, SPARSE_PROJECT_NAME)

# 3. 剪枝配置
PRUNED_MODEL_PATH = os.path.join(SPARSE_OUTPUT_DIR, "weights/last_pruned.pt")

# 4. 微调配置
FINETUNE_PROJECT_NAME = "Pruning_Finetune02041524"


# ================= 功能函数 =================

def step1_train():
    """
    Step 1: 正常训练 (Baseline)
    """
    print("\n========== Step 1: Normal Training ==========")
    model = YOLO(PRETRAIN_WEIGHTS)
    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=100,
        batch=32,
        project=RUNS_DIR,
        name=PROJECT_NAME,
        exist_ok=True
    )


def step2_Constraint_train(sr=0.0001):
    """
    Step 2: 约束训练 (Sparse Training)
    sr: 稀疏率，建议 0.0001 - 0.001
    """
    print(f"\n========== Step 2: Constraint Training (SR={sr}) ==========")

    if not os.path.exists(SPARSE_INPUT_MODEL):
        print(f"错误：找不到输入模型 {SPARSE_INPUT_MODEL}")
        return

    model = YOLO(SPARSE_INPUT_MODEL)

    # --- 定义稀疏训练回调函数 ---
    def on_train_batch_end(trainer):
        for module in trainer.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, 'weight'):
                if module.weight.grad is not None:
                    module.weight.grad.data.add_(sr * torch.sign(module.weight.data))

    model.add_callback("on_train_batch_end", on_train_batch_end)

    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=20, # 约束训练不需要太多轮次，因为我们只是为了让不重要的权重趋向0
        batch=32,
        amp=False, # 必须关闭 AMP
        save_period=1,
        project=RUNS_DIR,
        name=SPARSE_PROJECT_NAME,
        exist_ok=True
    )
    print(f"约束训练完成，模型保存在: {SPARSE_OUTPUT_DIR}")


def step3_pruning(ratio=0.5):
    """
    Step 3: 执行剪枝
    ratio: 保留通道的比例 (0.0 - 1.0)
    """
    print(f"\n========== Step 3: Pruning (Ratio={ratio}) ==========")
    from LL_pruning import do_pruning

    input_model = os.path.join(SPARSE_OUTPUT_DIR, "weights/last.pt")

    if not os.path.exists(input_model):
        print(f"错误：找不到约束训练模型 {input_model}")
        return

    # 调用剪枝函数，传入我们设定的比例
    do_pruning(input_model, PRUNED_MODEL_PATH, target_ratio=ratio)


def step4_finetune():
    """
    Step 4: 微调 (Finetune)
    """
    print("\n========== Step 4: Finetuning ==========")

    if not os.path.exists(PRUNED_MODEL_PATH):
        print(f"错误：找不到剪枝后的模型 {PRUNED_MODEL_PATH}")
        return

    model = YOLO(PRUNED_MODEL_PATH)

    # 微调正常开启 AMP，不需要稀疏回调
    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=20, # 微调建议增加轮次，比如 20-50
        batch=32,
        project=RUNS_DIR,
        name=FINETUNE_PROJECT_NAME,
        exist_ok=True
    )


if __name__ == "__main__":
    # --- 1. 正常训练 ---
    # step1_train()

    # --- 2. 约束训练 ---
    step2_Constraint_train(sr=0.002)

    # --- 3. 剪枝 ---
    # 这里设置你想要的比例，比如 0.8 (保留80%) 或 0.5 (保留50%)
    # 因为代码已经有了对齐保护，这里可以是任意数值
    step3_pruning(ratio=0.5)

    # --- 4. 微调 ---
    step4_finetune()

    print("\n所有流程执行完毕！")