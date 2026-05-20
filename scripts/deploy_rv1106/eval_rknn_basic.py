import os, sys, time, math
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon
from rknn.api import RKNN

# ---------- 配置 ----------
# 🚀 更改点 1: 将模型路径改为你的 ONNX 模型路径
ONNX_PATH = "/root/ultralytics_yolov8-main/test.onnx" 
IMAGE_DIR = "/root/autodl-tmp/yolo_dataset_gray/val/images"
LABEL_DIR = "/root/autodl-tmp/yolo_dataset_gray/val/labels"

INPUT_SIZE = (256, 256)
BG_COLOR = 114
CONF_THRESH = 0.5
NMS_THRESH = 0.4
CLASSES = ['BARCODE', 'DM', 'HANXIN', 'QR', 'PDF',
           'AZTEC', 'CODEONE', 'DOT', 'GM', 'MAXI',
           'MPDF', 'MQR', 'RMQR', 'ULTRA', 'UPN']
NUM_CLASSES = len(CLASSES)

# ---------- 工具函数 ----------
def load_labels(txt_path):
    objects = []
    if not os.path.exists(txt_path):
        return objects
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            cls_id = int(parts[0])
            xmin, ymin, xmax, ymax, angle = map(float, parts[1:6])
            objects.append([cls_id, xmin, ymin, xmax, ymax, angle])
    return objects

def letterbox(gray_img, size, color):
    h, w = gray_img.shape
    tw, th = size
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(gray_img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.ones((th, tw), dtype=np.uint8) * color
    pad_x = (tw - nw)//2
    pad_y = (th - nh)//2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, scale, pad_x, pad_y

def rotate_rect(xmin, ymin, xmax, ymax, angle):
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    w, h = xmax-xmin, ymax-ymin
    ca, sa = math.cos(angle), math.sin(angle)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot = np.array([[ca, -sa], [sa, ca]])
    rotated = np.dot(corners, rot)
    return [(cx+r[0], cy+r[1]) for r in rotated]

def obb_iou(b1, b2):
    poly1 = Polygon(rotate_rect(*b1[1:7]))
    poly2 = Polygon(rotate_rect(*b2[1:7]))
    if not poly1.is_valid or not poly2.is_valid: return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter/union if union>0 else 0.0

def nms_obb(dets, thresh):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x[6], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if d[0]!=best[0] or obb_iou(best, d)<=thresh]
    return keep

def sigmoid(x): return 1/(1+np.exp(-x))

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# 适配 reg_max=8 的 DFL 解码 (通道数降至 8)
def dfl_decode(feat):
    N = feat.shape[-1]
    xywh = np.zeros((4, N), dtype=np.float32)
    for k in range(4):
        dist = feat[0, k*8:(k+1)*8, :]
        dist = softmax(dist, axis=0)
        coord = np.arange(8, dtype=np.float32).reshape(8,1)
        xywh[k] = np.sum(dist * coord, axis=0)
    return xywh

# 适配总计 47 通道的特征图切分 (32 个 bbox 通道 + 15 个类别通道)
def process(outputs, shape, scale, px, py, th):
    h, w = shape
    feats = outputs[:3]                    
    angle_vec = outputs[3].reshape(-1)     
    strides = [8, 16, 32]
    results = []
    offset = 0
    for i, feat in enumerate(feats):
        stride = strides[i]
        gh, gw = feat.shape[2], feat.shape[3]
        N = gh * gw
        angle_slice = angle_vec[offset:offset+N]
        offset += N

        cls_conf = sigmoid(feat[:, 32:, :, :])          
        box_raw = feat[:, :32, :, :]                    
        box_raw_flat = box_raw.reshape(1, 32, -1)
        box_lt_rb = dfl_decode(box_raw_flat)            

        for h_ in range(gh):
            for w_ in range(gw):
                idx = h_*gw + w_
                scores = cls_conf[0, :, h_, w_]
                cls_id = int(np.argmax(scores))
                score = scores[cls_id]
                if score < th: continue

                l, t_, r, b = box_lt_rb[:, idx]
                bw, bh = l+r, t_+b
                dx = (r - l)*0.5
                dy = (b - t_)*0.5

                ang = (angle_slice[idx] - 0.25) * math.pi
                ca, sa = math.cos(ang), math.sin(ang)

                cx = (w_ + 0.5 + dx*ca - dy*sa) * stride
                cy = (h_ + 0.5 + dx*sa + dy*ca) * stride

                half_w = (abs(bw*ca) + abs(bh*sa)) * stride / 2
                half_h = (abs(bw*sa) + abs(bh*ca)) * stride / 2

                xmin = (cx - half_w - px) / scale
                ymin = (cy - half_h - py) / scale
                xmax = (cx + half_w - px) / scale
                ymax = (cy + half_h - py) / scale

                results.append([cls_id, xmin, ymin, xmax, ymax, ang, score])
    return results

# ---------- mAP 计算 ----------
def compute_ap(prec, rec):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        if not np.any(mask): p = 0.0
        else: p = np.max(prec[mask])
        ap += p/11.0
    return ap

def main():
    if not os.path.exists(ONNX_PATH):
        print(f"ONNX Model not found: {ONNX_PATH}")
        return

    # 设为 True 可以打印出构建模型的详细日志
    rknn = RKNN(verbose=True)
    
    # 🚀 更改点 2: 配置 RKNN
    # 你的输入是单通道灰度图，所以 mean 和 std 用单列表包围。
    # 【注意】: 这里的 0 和 255 必须与你 convert.py 里的配置完全一致！
    print("--> Configuring RKNN...")
    rknn.config(mean_values=[[0]], std_values=[[255]], target_platform='rk3588')

    # 🚀 更改点 3: 加载 ONNX 并在内存中构建
    print(f"--> Loading ONNX model: {ONNX_PATH}")
    ret = rknn.load_onnx(model=ONNX_PATH)
    if ret != 0:
        print("Load ONNX failed!")
        return

    print("--> Building RKNN model (Quantization=True for now)...")
    ret = rknn.build(do_quantization=False, dataset="/root/ultralytics_yolov8-main/dataset.txt")
    if ret != 0:
        print("Build RKNN failed!")
        return
    
    # 🚀 更改点 4: 初始化 PC 模拟器运行环境
    print("--> Init runtime environment...")
    ret = rknn.init_runtime(target=None)
    if ret != 0:
        print("Init runtime environment failed!")
        return

    imgs = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    imgs = imgs[:50]
    print(f"Total images: {len(imgs)}")


    all_dets = {c:[] for c in range(NUM_CLASSES)}
    all_gts = {c:[] for c in range(NUM_CLASSES)}
    img_id = 0

    for fname in tqdm(imgs):
        img = cv2.imread(os.path.join(IMAGE_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        lb, sc, px, py = letterbox(img, INPUT_SIZE, BG_COLOR)
        
        # 将 [256, 256] 扩展为 [1, 256, 256, 1] 以满足 NCHW/NHWC 的输入需求
        inputs = np.expand_dims(lb, axis=(0,-1))

        outputs = rknn.inference(inputs=[inputs])
        dets = process(outputs, INPUT_SIZE, sc, px, py, CONF_THRESH)
        dets = nms_obb(dets, NMS_THRESH)

        for d in dets:
            all_dets[d[0]].append([d[6], img_id] + d[1:6])

        for g in load_labels(os.path.join(LABEL_DIR, os.path.splitext(fname)[0]+'.txt')):
            all_gts[int(g[0])].append([img_id] + g[1:])

        img_id += 1

    aps = []
    print("\nPer-class AP @0.5 IOU:")
    for c in range(NUM_CLASSES):
        gt = all_gts[c]
        pred = all_dets[c]
        if len(gt)==0:
            print(f"  {CLASSES[c]:10s} : 0.0000 (no GT)")
            aps.append(0.0); continue
        if len(pred)==0:
            print(f"  {CLASSES[c]:10s} : 0.0000 (no det)")
            aps.append(0.0); continue

        pred = sorted(pred, key=lambda x:x[0], reverse=True)
        matched = [False]*len(gt)
        tp = np.zeros(len(pred)); fp = np.zeros(len(pred))
        for i, p in enumerate(pred):
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if g[0]!=p[1] or matched[j]: continue
                iou = obb_iou([c]+p[2:], [c]+g[1:])
                if iou>best_iou: best_iou, best_j = iou, j
            if best_iou>=0.5 and best_j>=0:
                tp[i]=1; matched[best_j]=True
            else:
                fp[i]=1
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        rec = tp_cum/len(gt)
        prec = tp_cum/(tp_cum+fp_cum+1e-8)
        ap = compute_ap(prec, rec)
        aps.append(ap)
        print(f"  {CLASSES[c]:10s} : {ap:.4f}")

    print(f"\nmAP@{CONF_THRESH}: {np.mean(aps):.4f}")
    rknn.release()

if __name__ == "__main__":
    main()
