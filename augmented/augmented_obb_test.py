import os
import cv2
import random
import albumentations as A
import numpy as np
from pathlib import Path

# ================= 配置参数 =================
ORIG_IMG_DIR = "/home/lab/桌面/ultralytics_yolov8-main/VOCdevkit/VOC2007/test/images"  # OBB原图目录
ORIG_LABEL_DIR = "/home/lab/桌面/ultralytics_yolov8-main/VOCdevkit/VOC2007/test/labels"  # OBB原标注目录
AUG_IMG_DIR = "/home/lab/桌面/ultralytics_yolov8-main/VOCdevkit/VOC2007/test/augmented/images"  # 增强图保存目录
AUG_LABEL_DIR = "/home/lab/桌面/ultralytics_yolov8-main/VOCdevkit/VOC2007/test/augmented/labels"  # 增强OBB标注保存目录

AUG_NUM_PER_IMG = 4  # 每张原图生成多少张增强图
RANDOM_SEED = 42
IMG_SIZE = (640, 640)  # (width, height)

# ================= 初始化目录 =================
Path(AUG_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(AUG_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# ================= 定义增强管道（适配OBB关键点） =================
transform = A.Compose([
    A.RandomScale(scale_limit=(-0.1, -0.4), p=0.7),
    A.PadIfNeeded(
        min_height=640,
        min_width=640,
        position='center',
        border_mode=0,
    ),
    A.Rotate(
        limit=89,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.6
    ),
    A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.7, p=0.6),
    A.MotionBlur(blur_limit=(3, 5), p=0.8),
    A.GaussNoise(std_range=(0.2, 0.3), p=0.2),
    A.HorizontalFlip(p=0.6),
    A.VerticalFlip(p=0.6),
    # A.RandomResizedCrop(
        #     size=IMG_SIZE,
        #     scale=(0.8, 0.9),
        #     ratio=(1, 1.5),
        #     p=1
        # ),
    # A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0),
    # A.Affine(shear=10, p=1),
    # A.RandomGamma(gamma_limit=(90, 110), p=0),
], keypoint_params=A.KeypointParams(
    format="xy",  # 关键点格式为(x,y)
    remove_invisible=False,
    angle_in_degrees=True
), additional_targets={"class_labels": "labels"})


# ================= 工具函数 =================
def validate_original_obb(obb_coords):
    """校验原始OBB四顶点是否合法（归一化坐标0~1）"""
    if len(obb_coords) != 8:
        return False
    for c in obb_coords:
        if c < 0 or c > 1:
            return False
    return True


def normalize_obb_keypoints(keypoints, img_shape):
    """将像素坐标的OBB顶点归一化到0~1"""
    h, w = img_shape[:2]
    normalized = []
    for (x, y) in keypoints:
        x_norm = max(0.0, min(1.0, x / w))
        y_norm = max(0.0, min(1.0, y / h))
        normalized.extend([x_norm, y_norm])
    return normalized


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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # 3. 读取OBB标注（解析四顶点）
        if not os.path.exists(label_path):
            print(f"[跳过] 缺少标注文件: {img_name}")
            skipped_files += 1
            continue

        obb_keypoints = []  # 存储格式：[[(x1,y1), (x2,y2), (x3,y3), (x4,y4)], ...]
        class_labels = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            try:
                parts = list(map(float, line.split()))
                cls_id = int(parts[0])
                obb_coords = parts[1:]  # 8个归一化坐标

                if validate_original_obb(obb_coords):
                    # 将归一化坐标转回像素坐标（Albumentations需要像素坐标做变换）
                    keypoints = []
                    for j in range(0, 8, 2):
                        x_pix = obb_coords[j] * orig_w
                        y_pix = obb_coords[j + 1] * orig_h
                        keypoints.append((x_pix, y_pix))
                    obb_keypoints.append(keypoints)
                    class_labels.append(cls_id)
            except Exception as e:
                print(f"[警告] 解析标注失败: {line} | {e}")
                continue

        # 4. 生成增强样本
        for aug_idx in range(AUG_NUM_PER_IMG):
            try:
                # 展平关键点（Albumentations要求一维列表）
                flat_keypoints = [p for obb in obb_keypoints for p in obb] if obb_keypoints else []

                # 执行增强
                augmented = transform(
                    image=img,
                    keypoints=flat_keypoints,
                    class_labels=class_labels
                )

                aug_img = augmented["image"]
                aug_keypoints = augmented["keypoints"]
                aug_cls = augmented["class_labels"]
                aug_h, aug_w = aug_img.shape[:2]

                # 5. 处理增强后的OBB顶点
                final_labels = []
                num_obb = len(aug_cls)
                for i_obb in range(num_obb):
                    # 恢复单个OBB的4个顶点
                    start = i_obb * 4
                    end = start + 4
                    if end > len(aug_keypoints):
                        continue
                    obb_pts = aug_keypoints[start:end]

                    # 归一化顶点并保存
                    norm_obb = normalize_obb_keypoints(obb_pts, (aug_h, aug_w))
                    if all(0.0 <= c <= 1.0 for c in norm_obb):
                        obb_line = f"{int(aug_cls[i_obb])} {' '.join([f'{c:.6f}' for c in norm_obb])}"
                        final_labels.append(obb_line)

                # 6. 保存文件
                save_name_stem = f"{Path(img_name).stem}_aug{aug_idx}"
                save_img_path = os.path.join(AUG_IMG_DIR, save_name_stem + ".jpg")
                save_label_path = os.path.join(AUG_LABEL_DIR, save_name_stem + ".txt")

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
