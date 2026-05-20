import cv2
import math
import numpy as np
import onnxruntime as ort

STD_ONNX = "runs/obb/qr_obb_rv11065/weights/best_std.onnx"
HEAD4_ONNX = "runs/obb/qr_obb_rv11065/weights/best_4head.onnx"

IMG_PATH = "/root/autodl-tmp/yolo_dataset_gray/val/images/BARCODE_00726.jpg"

INPUT_SIZE = 256
NUM_CLASSES = 15
REG_MAX = 8
CONF_THRESH = 0.25

CLASSES = [
    "BARCODE", "DM", "HANXIN", "QR", "PDF",
    "AZTEC", "CODEONE", "DOT", "GM", "MAXI",
    "MPDF", "MQR", "RMQR", "ULTRA", "UPN"
]


def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def letterbox(gray, size=256, color=114):
    h, w = gray.shape
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(gray, (nw, nh))
    canvas = np.ones((size, size), dtype=np.uint8) * color

    px = (size - nw) // 2
    py = (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized

    return canvas, scale, px, py


def dfl_decode(box_feat):
    n = box_feat.shape[-1]
    out = np.zeros((4, n), dtype=np.float32)

    for k in range(4):
        dist = softmax(box_feat[0, k * REG_MAX:(k + 1) * REG_MAX, :], axis=0)
        proj = np.arange(REG_MAX, dtype=np.float32).reshape(REG_MAX, 1)
        out[k] = np.sum(dist * proj, axis=0)

    return out


def decode_4head(outputs):
    feats = outputs[:3]
    angle_vec = outputs[3].reshape(-1)

    feats = sorted(feats, key=lambda z: z.shape[2], reverse=True)

    angle_offsets = {
        32: (0, 1024),
        16: (1024, 1280),
        8: (1280, 1344),
    }

    dets = []

    for feat in feats:
        _, _, gh, gw = feat.shape
        stride = INPUT_SIZE // gw

        start, end = angle_offsets[gw]
        angles = angle_vec[start:end]

        box_feat = feat[:, :32, :, :].reshape(1, 32, -1)
        cls_conf = feat[:, 32:32 + NUM_CLASSES, :, :]

        box_ltrb = dfl_decode(box_feat)

        for gy in range(gh):
            for gx in range(gw):
                idx = gy * gw + gx

                scores = cls_conf[0, :, gy, gx]
                cls_id = int(np.argmax(scores))
                score = float(scores[cls_id])

                if score < CONF_THRESH:
                    continue

                l, t, r, b = box_ltrb[:, idx]

                bw = (l + r) * stride
                bh = (t + b) * stride

                dx = (r - l) * 0.5
                dy = (b - t) * 0.5

                # 关键：四头 angle_head 已经是最终 angle，不要 sigmoid，不要再映射
                angle = float(angles[idx])

                ca, sa = math.cos(angle), math.sin(angle)

                anchor_x = gx + 0.5
                anchor_y = gy + 0.5

                # 官方 dist2rbox 对齐公式
                cx = (anchor_x + dx * ca - dy * sa) * stride
                cy = (anchor_y + dx * sa + dy * ca) * stride

                dets.append([
                    cls_id,
                    score,
                    float(cx),
                    float(cy),
                    float(bw),
                    float(bh),
                    angle,
                    gw,
                    gx,
                    gy,
                ])

    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    return dets[:20]


def decode_std(output):
    pred = output[0]

    cls_part = pred[4:4 + NUM_CLASSES, :]
    angle_part = pred[4 + NUM_CLASSES:, :]

    dets = []

    for i in range(pred.shape[1]):
        scores = cls_part[:, i]
        cls_id = int(np.argmax(scores))
        score = float(scores[cls_id])

        if score < CONF_THRESH:
            continue

        cx, cy, w, h = pred[:4, i]
        angle = float(angle_part[0, i])

        dets.append([
            cls_id,
            score,
            float(cx),
            float(cy),
            float(w),
            float(h),
            angle,
            i,
        ])

    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    return dets[:20]


def run_onnx(path, inp):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: inp})


def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Image not found: {IMG_PATH}"

    lb, _, _, _ = letterbox(img, INPUT_SIZE)
    inp = lb[None, None, :, :].astype(np.float32) / 255.0

    std_out = run_onnx(STD_ONNX, inp)
    head4_out = run_onnx(HEAD4_ONNX, inp)

    print("\n标准 ONNX outputs:")
    for o in std_out:
        print(o.shape)

    print("\n四头 ONNX outputs:")
    for o in head4_out:
        print(o.shape)

    std_dets = decode_std(std_out[0])
    h4_dets = decode_4head(head4_out)

    print("\n========== 标准 ONNX Top20 ==========")
    for d in std_dets:
        print(
            f"{CLASSES[d[0]]:8s} conf={d[1]:.4f} "
            f"cx={d[2]:.2f} cy={d[3]:.2f} w={d[4]:.2f} h={d[5]:.2f} "
            f"ang={d[6]:.4f} idx={d[7]}"
        )

    print("\n========== 四头 ONNX Top20 ==========")
    for d in h4_dets:
        print(
            f"{CLASSES[d[0]]:8s} conf={d[1]:.4f} "
            f"cx={d[2]:.2f} cy={d[3]:.2f} w={d[4]:.2f} h={d[5]:.2f} "
            f"ang={d[6]:.4f} grid={d[7]} xy=({d[8]},{d[9]})"
        )


if __name__ == "__main__":
    main()