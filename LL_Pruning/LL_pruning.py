from torch import nn
from ultralytics import YOLO
import torch
import math
# 导入基础模块
from ultralytics.nn.modules import (Bottleneck, Conv, C2f, SPPF, Detect,
                                    vanillanetBlock, C2f_SE)
# 导入卷积模块
from ultralytics.nn.modules.conv import SE_Bottleneck, SEAttention
# 导入 OBB 头
from ultralytics.nn.modules.head import OBB
import os


class PRUNE():
    def __init__(self) -> None:
        pass

    def prune_conv(self, conv1_module, conv2_module, ratio=0.5):
        target_bn = None
        cls_name = conv1_module.__class__.__name__

        # 1. 提取 BN 用于判断通道重要性
        if cls_name == 'vanillanetBlock':
            if hasattr(conv1_module, 'deploy') and conv1_module.deploy: return
            target_bn = conv1_module.conv1[1]
        elif cls_name == 'Conv':
            target_bn = conv1_module.bn
        elif cls_name in ['SE_Bottleneck', 'C2f', 'C2f_SE', 'SPPF']:
            target_bn = conv1_module.cv1.bn

        if target_bn is None: return

        # 2. 计算阈值与对齐 (RV1106 NPU 优化核心)
        gamma = target_bn.weight.data.detach().abs()
        n_total = len(gamma)

        # 基础目标通道数
        n_target = int(n_total * ratio)

        # 【NPU 优化】强制对齐到 divisor 的倍数 (通常为 8 或 16)
        # 这解决了奇数通道导致的 split 报错，同时提升 NPU 内存访问效率
        divisor = 8
        n_aligned = round(n_target / divisor) * divisor

        # 边界保护：不能小于 divisor，且不能超过总数
        if n_aligned < divisor:
            n_aligned = divisor
        if n_aligned > n_total:
            n_aligned = (n_total // divisor) * divisor
            if n_aligned == 0: n_aligned = divisor  # 防止总数本来就小于divisor的情况

        n = n_aligned

        # 获取保留通道的索引
        topk_values, topk_indices = torch.topk(gamma, n)
        keep_idxs = topk_indices

        # 排序索引，保持原有的通道顺序（虽然对于BN来说顺序不重要，但保持习惯）
        keep_idxs, _ = torch.sort(keep_idxs)

        print(f"✂️ 剪枝 {cls_name}: {n_total} -> {n} ({n / n_total:.2%}) [对齐到 {divisor}]")

        # 3. 剪枝操作实施
        def apply_prune(bn, conv, idxs, num):
            bn.weight.data = bn.weight.data[idxs]
            bn.bias.data = bn.bias.data[idxs]
            bn.running_var.data = bn.running_var.data[idxs]
            bn.running_mean.data = bn.running_mean.data[idxs]
            bn.num_features = num
            conv.weight.data = conv.weight.data[idxs]
            conv.out_channels = num
            if conv.bias is not None:
                conv.bias.data = conv.bias.data[idxs]

        def prune_se_module(se_module, new_channels):
            # SE 模块内部通常有一个 reduction ratio，重新计算中间层
            new_reduced = max(4, new_channels // 16)

            # 对齐 SE 中间层也建议对齐，这里简单保证偶数
            if new_reduced % 2 != 0: new_reduced += 1

            old_w1 = se_module.l1.weight.data
            old_w2 = se_module.l2.weight.data
            old_b2 = se_module.l2.bias.data if se_module.l2.bias is not None else None

            # 重建 L1
            new_l1 = nn.Linear(new_channels, new_reduced, bias=False)
            out_idx = min(new_reduced, old_w1.shape[0])
            in_idx = min(new_channels, old_w1.shape[1])
            new_l1.weight.data[:out_idx, :in_idx] = old_w1[:out_idx, :in_idx]
            se_module.l1 = new_l1

            # 重建 L2
            new_l2 = nn.Linear(new_reduced, new_channels, bias=True if old_b2 is not None else False)
            out_idx_2 = min(new_channels, old_w2.shape[0])
            in_idx_2 = min(new_reduced, old_w2.shape[1])
            new_l2.weight.data[:out_idx_2, :in_idx_2] = old_w2[:out_idx_2, :in_idx_2]
            if old_b2 is not None:
                new_l2.bias.data[:out_idx_2] = old_b2[:out_idx_2]
            se_module.l2 = new_l2

        upstream_out_channels = None

        if cls_name == 'Conv':
            apply_prune(conv1_module.bn, conv1_module.conv, keep_idxs, n)
            upstream_out_channels = n

        elif cls_name == 'vanillanetBlock':
            apply_prune(conv1_module.conv1[1], conv1_module.conv1[0], keep_idxs, n)
            conv2 = conv1_module.conv2[0]
            conv2.in_channels = n
            if conv2.weight.shape[1] > n:
                conv2.weight.data = conv2.weight.data[:, :n]
            upstream_out_channels = conv2.out_channels

        elif cls_name == 'SE_Bottleneck':
            apply_prune(conv1_module.cv1.bn, conv1_module.cv1.conv, keep_idxs, n)
            if hasattr(conv1_module, 'cv2'):
                conv1_module.cv2.conv.in_channels = n
                conv1_module.cv2.conv.weight.data = conv1_module.cv2.conv.weight.data[:, :n]
            upstream_out_channels = conv1_module.cv2.conv.out_channels

        elif cls_name in ['C2f', 'C2f_SE']:
            apply_prune(conv1_module.cv1.bn, conv1_module.cv1.conv, keep_idxs, n)
            # n 必定是偶数且是对齐的，所以整除 2 安全
            c_hidden = n // 2

            for m in conv1_module.m:
                m_cls = m.__class__.__name__
                if m_cls == 'SE_Bottleneck':
                    m.cv1.conv.in_channels = c_hidden
                    m.cv1.conv.weight.data = m.cv1.conv.weight.data[:, :c_hidden]
                    m.cv2.conv.out_channels = c_hidden
                    m.cv2.conv.weight.data = m.cv2.conv.weight.data[:c_hidden]
                    if m.cv2.conv.bias is not None:
                        m.cv2.conv.bias.data = m.cv2.conv.bias.data[:c_hidden]
                    if hasattr(m.cv2, 'bn'):
                        bn = m.cv2.bn
                        bn.num_features = c_hidden
                        bn.weight.data = bn.weight.data[:c_hidden]
                        bn.bias.data = bn.bias.data[:c_hidden]
                        bn.running_mean.data = bn.running_mean.data[:c_hidden]
                        bn.running_var.data = bn.running_var.data[:c_hidden]
                    if hasattr(m, 'se'):
                        prune_se_module(m.se, c_hidden)

            num_bottlenecks = len(conv1_module.m)
            # 重新计算 cv2 输入通道
            new_cv2_in = c_hidden * (2 + num_bottlenecks)
            conv1_module.cv2.conv.in_channels = new_cv2_in
            conv1_module.cv2.conv.weight.data = conv1_module.cv2.conv.weight.data[:, :new_cv2_in]
            upstream_out_channels = conv1_module.cv2.conv.out_channels

        elif cls_name == 'SPPF':
            apply_prune(conv1_module.cv1.bn, conv1_module.cv1.conv, keep_idxs, n)
            new_cv2_in = n * 4
            conv1_module.cv2.conv.in_channels = new_cv2_in
            conv1_module.cv2.conv.weight.data = conv1_module.cv2.conv.weight.data[:, :new_cv2_in]
            upstream_out_channels = conv1_module.cv2.conv.out_channels

        # 4. 更新下游模块 (Safe Update)
        if not isinstance(conv2_module, list):
            conv2_module = [conv2_module]

        for item in conv2_module:
            self.safe_update(item, upstream_out_channels)

    def safe_update(self, module, target_in):
        if module is None or target_in is None: return

        cls_name = module.__class__.__name__

        if cls_name in ['Conv', 'Conv2d']:
            target_conv = module.conv if hasattr(module, 'conv') else module

            if hasattr(target_conv, 'in_channels') and target_conv.in_channels != target_in:
                target_conv.in_channels = target_in
                if target_conv.weight.shape[1] > target_in:
                    target_conv.weight.data = target_conv.weight.data[:, :target_in]
                # 处理 group conv (如 DWConv)，通常 groups == in_channels
                if target_conv.groups > 1:
                    # 如果是 depthwise，groups 必须等于 input channels
                    if target_conv.groups == target_conv.weight.shape[1]:
                        target_conv.groups = target_in

        elif cls_name == 'vanillanetBlock':
            self.safe_update(module.conv1[0], target_in)

        elif cls_name in ['C2f', 'C2f_SE']:
            self.safe_update(module.cv1, target_in)

        elif isinstance(module, (list, nn.ModuleList)):
            for sub in module:
                self.safe_update(sub, target_in)

    def prune(self, m1, m2, ratio):
        cls_name = m1.__class__.__name__
        if cls_name in ['C2f', 'C2f_SE', 'SPPF']:
            m1 = m1.cv2

        if not isinstance(m2, list):
            m2 = [m2]

        targets = []
        for item in m2:
            if item is None: continue
            iname = item.__class__.__name__
            if iname in ['C2f', 'C2f_SE', 'SPPF', 'SE_Bottleneck']:
                targets.append(item.cv1)
            else:
                targets.append(item)

        self.prune_conv(m1, targets, ratio)


def do_pruning(modelpath, savepath, target_ratio=0.5):
    """
    执行剪枝的主函数
    :param target_ratio: 目标保留比例 (0.0 - 1.0)
    """
    pruning = PRUNE()
    print(f"\n加载模型: {modelpath}")
    yolo = YOLO(modelpath)

    print(f"执行局部比例剪枝 (目标保留比例: {target_ratio})")

    seq = yolo.model.model

    print("\n剪枝 Backbone...")
    for i in range(4):
        if i < len(seq) - 1:
            if seq[i].__class__.__name__ in ['Conv', 'vanillanetBlock']:
                pruning.prune_conv(seq[i], [seq[i + 1]], ratio=target_ratio)
        else:
            if seq[i].__class__.__name__ in ['Conv', 'vanillanetBlock']:
                pruning.prune_conv(seq[i], [], ratio=target_ratio)

    print("\n剪枝 Neck & 模块宽度...")
    for m in yolo.model.modules():
        if m.__class__.__name__ in ['C2f', 'C2f_SE', 'SPPF']:
            pruning.prune_conv(m, [], ratio=target_ratio)

    print("\n剪枝检测头 (合并处理层间连接)...")
    detect = seq[-1]
    if detect.__class__.__name__ in ['Detect', 'OBB']:
        head_indices = detect.f
        last_inputs = [seq[i] for i in head_indices]
        valid_pairs = []
        has_cv4 = hasattr(detect, 'cv4')
        head_branches = [None] * 3
        head_cv2 = detect.cv2
        head_cv3 = detect.cv3
        head_cv4 = detect.cv4 if has_cv4 else [None] * len(head_cv2)

        for i, (li, cl, cv2, cv3, cv4) in enumerate(zip(last_inputs, head_branches, head_cv2, head_cv3, head_cv4)):
            if li is None: continue
            dsts = []
            if cl is not None: dsts.append(cl)
            if cv2 is not None: dsts.append(cv2[0])
            if cv3 is not None: dsts.append(cv3[0])
            if cv4 is not None: dsts.append(cv4[0])

            # 自动检测并添加下采样层 (解决维度不匹配的关键)
            try:
                curr_idx = -1
                for idx, m in enumerate(seq):
                    if m is li:
                        curr_idx = idx
                        break
                if curr_idx != -1 and (curr_idx + 1) < len(seq):
                    next_layer = seq[curr_idx + 1]
                    if isinstance(next_layer, (nn.Conv2d, Conv)):
                        conv_layer = next_layer.conv if hasattr(next_layer, 'conv') else next_layer
                        # 检查是否为下采样层 (stride=2)
                        if conv_layer.stride == (2, 2) or conv_layer.stride == 2:
                            dsts.append(next_layer)
            except Exception as e:
                print(f"Warning: 无法自动匹配下采样层: {e}")

            valid_pairs.append((li, dsts))

        # 执行 Head 输入的剪枝
        for inputs in valid_pairs:
            pruning.prune(inputs[0], inputs[1], ratio=target_ratio)

        # 执行 Head 内部的剪枝
        for i in range(len(detect.cv2)):
            pruning.prune(detect.cv2[i][0], detect.cv2[i][1], ratio=target_ratio)
            pruning.prune(detect.cv2[i][1], detect.cv2[i][2], ratio=target_ratio)
            pruning.prune(detect.cv3[i][0], detect.cv3[i][1], ratio=target_ratio)
            pruning.prune(detect.cv3[i][1], detect.cv3[i][2], ratio=target_ratio)
            if has_cv4:
                pruning.prune(detect.cv4[i][0], detect.cv4[i][1], ratio=target_ratio)
                pruning.prune(detect.cv4[i][1], detect.cv4[i][2], ratio=target_ratio)

    print(f"\n保存模型至: {savepath}")
    for p in yolo.model.parameters(): p.requires_grad = True
    yolo.model.half()  # 保存为 FP16 减小体积
    yolo.ckpt['model'] = yolo.model
    yolo.ckpt['optimizer'] = None
    yolo.ckpt['ema'] = None
    yolo.ckpt['updates'] = 0

    torch.save(yolo.ckpt, savepath)
    fs_orig = os.path.getsize(modelpath) / 1024 / 1024
    fs_new = os.path.getsize(savepath) / 1024 / 1024
    print(f"📊 模型文件大小: {fs_orig:.2f} MB -> {fs_new:.2f} MB (压缩率: {(1 - fs_new / fs_orig) * 100:.2f}%)")


if __name__ == "__main__":
    # 可以在这里单独测试剪枝脚本
    MODEL = "/home/lab/桌面/YOLOV8/ultralytics_yolov8-main/runs/obb/Constraint_Training/weights/last.pt"
    SAVE = "/home/lab/桌面/YOLOV8/ultralytics_yolov8-main/runs/obb/Constraint_Training/weights/last_pruned.pt"
    do_pruning(MODEL, SAVE, target_ratio=0.8)