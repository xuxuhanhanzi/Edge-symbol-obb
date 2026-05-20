import os
import math
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon
import onnxruntime as ort

# ---------- 配置 ----------
DEBUG_MODE = False
SAVE_INFERENCE = True
SAVE_DIR = "/root/ultralytics_yolov8-main/inference_results"

ONNX_PATH = "/root/ultralytics_yolov8-main/runs/obb/qr_obb_rv11065/weights/best_4head.onnx"
IMAGE_DIR = "/root/autodl-tmp/yolo_dataset_gray/val/images"
LABEL_DIR = "/root/autodl-tmp/yolo_dataset_gray/val/labels"

INPUT_SIZE = (256, 256)
BG_COLOR = 114
CONF_THRESH = 0.25
NMS_THRESH = 0.7
TOPK = 300

CLASSES = [
    "BARCODE", "DM", "HANXIN", "QR", "PDF",
    "AZTEC", "CODEONE", "DOT", "GM", "MAXI",
    "MPDF", "MQR", "RMQR", "ULTRA", "UPN"
]
NUM_CLASSES = len(CLASSES)
REG_MAX = 8


def load_labels(txt_path, img_w, img_h):
    objects = []
    if not os.path.exists(txt_path):
        return objects

    with open(txt_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 9:
                continue

            cls_id = int(parts[0])
            pts = np.array(parts[1:9], dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h

            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                objects.append([cls_id, poly])

    return objects


def letterbox(gray_img, size, color):
    h, w = gray_img.shape
    tw, th = size

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(gray_img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.ones((th, tw), dtype=np.uint8) * color
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    return canvas, scale, pad_x, pad_y


def rotate_rect(cx, cy, w, h, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    hw, hh = w / 2.0, h / 2.0

    signs_w = [1, -1, -1, 1]
    signs_h = [1, 1, -1, -1]

    pts = []
    for i in range(4):
        x = signs_w[i] * hw
        y = signs_h[i] * hh
        rx = x * ca - y * sa + cx
        ry = x * sa + y * ca + cy
        pts.append((rx, ry))

    return pts


def poly_iou(poly1, poly2):
    try:
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter

        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def obb_iou(b1, b2):
    poly1 = Polygon(rotate_rect(*b1[1:6]))
    poly2 = Polygon(rotate_rect(*b2[1:6]))
    return poly_iou(poly1, poly2)


def nms_obb(dets, thresh):
    if not dets:
        return []

    dets = sorted(dets, key=lambda x: x[6], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [
            d for d in dets
            if d[0] != best[0] or obb_iou(best, d) <= thresh
        ]

    return keep


def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def dfl_decode(box_feat):
    n = box_feat.shape[-1]
    xywh = np.zeros((4, n), dtype=np.float32)

    for k in range(4):
        dist = softmax(box_feat[0, k * REG_MAX:(k + 1) * REG_MAX, :], axis=0)
        coord = np.arange(REG_MAX, dtype=np.float32).reshape(REG_MAX, 1)
        xywh[k] = np.sum(dist * coord, axis=0)

    return xywh


def process(outputs, shape, scale, px, py, conf_thresh):
    feats = []
    angle_vec = None

    for out in outputs:
        if len(out.shape) == 4 and out.shape[1] in (47, 48):
            feats.append(out)
        elif len(out.shape) == 3 and out.shape[-1] == 1344:
            angle_vec = out.reshape(-1)

    if angle_vec is None:
        angle_vec = outputs[3].reshape(-1)

    if len(feats) != 3:
        feats = [outputs[0], outputs[1], outputs[2]]

    feats = sorted(feats, key=lambda x: x.shape[2], reverse=True)

    results = []

    angle_offsets = {
        32: (0, 1024),
        16: (1024, 1280),
        8: (1280, 1344),
    }

    for feat in feats:
        _, _, gh, gw = feat.shape
        stride = shape[0] // gw

        if gw not in angle_offsets:
            continue

        start, end = angle_offsets[gw]
        angle_slice = angle_vec[start:end]

        box_feat = feat[:, :32, :, :].reshape(1, 32, -1)

        # head.py 里 cls 已经 sigmoid，不能再 sigmoid
        cls_conf = feat[:, 32:32 + NUM_CLASSES, :, :]

        box_lt_rb = dfl_decode(box_feat)

        for gy in range(gh):
            for gx in range(gw):
                idx = gy * gw + gx

                scores = cls_conf[0, :, gy, gx]
                cls_id = int(np.argmax(scores))
                score = float(scores[cls_id])

                if score < conf_thresh:
                    continue

                l, t, r, b = box_lt_rb[:, idx]

                bw = (l + r) * stride
                bh = (t + b) * stride

                dx = (r - l) * 0.5
                dy = (b - t) * 0.5

                # 关键：angle_head 已经是最终 angle
                ang = float(angle_slice[idx])

                ca, sa = math.cos(ang), math.sin(ang)

                anchor_x = gx + 0.5
                anchor_y = gy + 0.5

                cx = (anchor_x + dx * ca - dy * sa) * stride
                cy = (anchor_y + dx * sa + dy * ca) * stride

                ocx = (cx - px) / scale
                ocy = (cy - py) / scale
                obw = bw / scale
                obh = bh / scale

                results.append([
                    cls_id,
                    float(ocx),
                    float(ocy),
                    float(obw),
                    float(obh),
                    float(ang),
                    score,
                ])

    results = sorted(results, key=lambda x: x[6], reverse=True)[:TOPK]
    return results


def compute_ap(prec, rec):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        ap += np.max(prec[mask]) / 11.0 if np.any(mask) else 0.0
    return ap


def draw_result(img, gts, dets):
    draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for g in gts:
        pts = np.array(g[1].exterior.coords)[:4].astype(np.int32)
        cv2.polylines(draw_img, [pts], True, (0, 255, 0), 2)

    for d in dets:
        pts = np.array(rotate_rect(*d[1:6])).astype(np.int32)
        cv2.polylines(draw_img, [pts], True, (0, 0, 255), 2)

        label = f"{CLASSES[d[0]]} {d[6]:.2f}"
        cv2.putText(
            draw_img,
            label,
            (int(d[1]), int(d[2])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return draw_img


def main():
    if not os.path.exists(ONNX_PATH):
        print(f"Model not found: {ONNX_PATH}")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"Image dir not found: {IMAGE_DIR}")
        return

    if SAVE_INFERENCE:
        os.makedirs(SAVE_DIR, exist_ok=True)

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print("ONNX input:")
    for i in session.get_inputs():
        print(" ", i.name, i.shape, i.type)

    print("ONNX outputs:")
    for o in session.get_outputs():
        print(" ", o.name, o.shape, o.type)

    imgs = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if DEBUG_MODE:
        imgs = imgs[:20]

    print(f"\n--> Starting inference: {len(imgs)} images")

    all_dets = {c: [] for c in range(NUM_CLASSES)}
    all_gts = {c: [] for c in range(NUM_CLASSES)}

    for img_id, fname in enumerate(tqdm(imgs)):
        img_path = os.path.join(IMAGE_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img_h, img_w = img.shape

        lb, sc, px, py = letterbox(img, INPUT_SIZE, BG_COLOR)
        inp = np.expand_dims(lb, axis=(0, 1)).astype(np.float32) / 255.0

        outputs = session.run(None, {input_name: inp})

        dets = process(outputs, INPUT_SIZE, sc, px, py, CONF_THRESH)
        dets = nms_obb(dets, NMS_THRESH)

        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(fname)[0] + ".txt"
        )
        gts = load_labels(label_path, img_w, img_h)

        if SAVE_INFERENCE:
            draw_img = draw_result(img, gts, dets)
            cv2.imwrite(os.path.join(SAVE_DIR, fname), draw_img)

        for d in dets:
            poly = Polygon(rotate_rect(*d[1:6]))
            if poly.is_valid and poly.area > 0:
                all_dets[d[0]].append([d[6], img_id, poly])

        for g in gts:
            all_gts[g[0]].append([img_id, g[1]])

    print("\n" + "=" * 30)
    print("Per-class AP @0.5 IOU:")
    print("=" * 30)

    aps = []

    for c in range(NUM_CLASSES):
        gt = all_gts[c]
        pred = all_dets[c]

        if not gt:
            print(f"  {CLASSES[c]:10s} : No GT")
            continue

        if not pred:
            print(f"  {CLASSES[c]:10s} : 0.0000")
            aps.append(0.0)
            continue

        pred = sorted(pred, key=lambda x: x[0], reverse=True)

        matched = [False] * len(gt)
        tp = np.zeros(len(pred), dtype=np.float32)
        fp = np.zeros(len(pred), dtype=np.float32)

        for i, p in enumerate(pred):
            best_iou = 0.0
            best_j = -1

            for j, g in enumerate(gt):
                if g[0] != p[1] or matched[j]:
                    continue

                iou = poly_iou(p[2], g[1])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.5:
                tp[i] = 1.0
                matched[best_j] = True
            else:
                fp[i] = 1.0

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)

        rec = tp_c / (len(gt) + 1e-8)
        prec = tp_c / (tp_c + fp_c + 1e-8)

        ap = compute_ap(prec, rec)
        aps.append(ap)

        print(f"  {CLASSES[c]:10s} : {ap:.4f}")

    final_map = np.mean(aps) if aps else 0.0
    print(f"\nFinal mAP@0.5: {final_map:.4f}")
    print(f"Saved inference images to: {SAVE_DIR}")


if __name__ == "__main__":
    main()