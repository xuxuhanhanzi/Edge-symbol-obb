import os
import cv2
import random
import albumentations as A
import numpy as np
from pathlib import Path

from torch.utils.hipify.hipify_python import value

# ================= 配置参数 =================
# 请确保这些路径与你的实际路径一致
ORIG_IMG_DIR = "VOCdevkit/VOC2007/JPEGImages"  # 原图目录
ORIG_LABEL_DIR = "VOCdevkit/VOC2007/YOLOLabels"  # 原标注目录
AUG_IMG_DIR = "VOCdevkit/VOC2007/augmented/images"  # 增强图保存目录
AUG_LABEL_DIR = "VOCdevkit/VOC2007/augmented/labels"  # 增强标注保存目录

AUG_NUM_PER_IMG = 4  # 每张原图生成多少张增强图
RANDOM_SEED = 42
IMG_SIZE = (640, 640)  # (width, height)

# ================= 初始化目录 =================
Path(AUG_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(AUG_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# ================= 定义增强管道 =================
# 修复了参数名并调整了强度，适合YOLO训练
transform = A.Compose([
    # 1. 几何变换：缩放与裁剪
    A.RandomResizedCrop(
        size=IMG_SIZE,
        scale=(0.8, 1),
        ratio=(0.9, 1.1),
        p=0.2
    ),
    A.RandomScale(scale_limit=(-0.2, -0.5), p=0.6),
    A.PadIfNeeded(
            min_height=640,
            min_width=640,
            position='center',  # 缩小后的图片居中
            border_mode=0,      # 填充模式（0=黑色，可根据需求改）
        ),

    # 2. 几何变换：旋转 (修正参数名 border_value -> value)
    A.Rotate(
        limit=20,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7
    ),

    A.Affine(shear=10, p=0.5),

    # 3. 色彩变换
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.4),

    # 4. 噪声与模糊
    A.MotionBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(std_range=(0.2,0.3), p=0.3),

    # 5. 翻转
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),  # 垂直翻转对某些数据集(如行人/车)可能不合理，设低一点

], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_area=10,  # 增强后面积太小则丢弃(像素)
    min_visibility=0.3,  # 增强后保留面积比例过低则丢弃
    check_each_transform=True
))


# ================= 工具函数 =================

def validate_original_bbox(bbox):
    """
    校验原始bbox是否合法。
    YOLO格式: center_x, center_y, w, h (全部归一化到 0~1)
    修正：允许bbox接触边缘，只检查是否严重越界。
    """
    x, y, w, h = bbox
    # 宽高必须大于0且小于等于1
    if w <= 0 or h <= 0 or w > 1 or h > 1:
        return False
    # 中心点必须在图内
    if x < 0 or x > 1 or y < 0 or y > 1:
        return False
    return True


def clamp_bbox(bbox):
    """
    将bbox坐标强制限制在 [0.0, 1.0] 范围内，防止浮点误差导致越界。
    """
    x, y, w, h = bbox
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    # 再次检查：如果中心点修正后导致半宽半高超出边界，需要微调（可选，通常clamp够了）
    return [x, y, w, h]


# ================= 主程序 =================
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    img_files = [f for f in os.listdir(ORIG_IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    total_imgs = len(img_files)
    total_aug_imgs = 0
    skipped_files = 0

    print(f"开始处理，共发现 {total_imgs} 张原始图片...")

    for i, img_name in enumerate(img_files):
        # 进度打印
        if i % 100 == 0:
            print(f"处理进度: {i}/{total_imgs}")

        # 1. 读取路径
        img_path = os.path.join(ORIG_IMG_DIR, img_name)
        label_path = os.path.join(ORIG_LABEL_DIR, Path(img_name).stem + ".txt")

        # 2. 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 图片损坏: {img_name}")
            skipped_files += 1
            continue
        # Albumentations 使用 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 读取标注
        if not os.path.exists(label_path):
            print(f"[跳过] 缺少标注文件: {img_name}")
            skipped_files += 1
            continue

        bboxes = []
        class_labels = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        # 解析原始标注
        has_valid_label = False
        for line in lines:
            line = line.strip()
            if not line: continue
            try:
                parts = list(map(float, line.split()))
                cls_id = int(parts[0])
                bbox = parts[1:]  # x, y, w, h

                if validate_original_bbox(bbox):
                    bboxes.append(bbox)
                    class_labels.append(cls_id)
                    has_valid_label = True
            except Exception:
                continue

        # 如果原图没有有效标注（或者是背景图），视情况处理
        # 这里选择：如果没有标注，依然做增强（作为背景负样本），除非你只想保留有目标的图
        # 如果只想保留有目标的图，取消下面两行的注释：
        # if not has_valid_label:
        #     continue

        # 4. 生成增强样本
        for aug_idx in range(AUG_NUM_PER_IMG):
            try:
                # 执行增强
                if len(bboxes) > 0:
                    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                else:
                    # 处理无标注图片（纯背景）
                    augmented = transform(image=img, bboxes=[], class_labels=[])

                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_cls = augmented["class_labels"]

                # 5. 处理增强后的标注
                final_labels = []
                if len(aug_bboxes) > 0:
                    for cls, bbox in zip(aug_cls, aug_bboxes):
                        # 坐标截断 + 格式化
                        cx, cy, w, h = clamp_bbox(bbox)

                        # 过滤掉极其微小的框（宽高接近0）
                        if w > 0.001 and h > 0.001:
                            final_labels.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                # 如果增强后没有框（且原图有框），说明框被裁掉了，这种图通常也可以保留作为负样本
                # 如果你不想保留空图，可以加判断 if not final_labels: continue

                # 6. 保存文件
                # 构建文件名：原名_aug0.jpg
                save_name_stem = f"{Path(img_name).stem}_aug{aug_idx}"
                save_img_path = os.path.join(AUG_IMG_DIR, save_name_stem + ".jpg")
                save_label_path = os.path.join(AUG_LABEL_DIR, save_name_stem + ".txt")

                # 转回 BGR 保存
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, aug_img_bgr)

                with open(save_label_path, "w") as f:
                    f.write("\n".join(final_labels))

                total_aug_imgs += 1

            except Exception as e:
                print(f"[错误] 处理 {img_name} 第 {aug_idx} 次增强失败: {e}")
                continue

    print("\n=== 处理完成 ===")
    print(f"原图数量: {total_imgs}")
    print(f"跳过数量: {skipped_files}")
    print(f"生成增强图: {total_aug_imgs}")
    print(f"保存路径: {AUG_IMG_DIR}")