# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


# class v8OBBLoss(v8DetectionLoss):
#     """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

#     def __init__(self, model):
#         """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
#         super().__init__(model)
#         self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
#         self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

#     def preprocess(self, targets, batch_size, scale_tensor):
#         """Preprocesses the target counts and matches with the input batch size to output a tensor."""
#         if targets.shape[0] == 0:
#             out = torch.zeros(batch_size, 0, 6, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             counts = counts.to(dtype=torch.int32)
#             out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
#             for j in range(batch_size):
#                 matches = i == j
#                 n = matches.sum()
#                 if n:
#                     bboxes = targets[matches, 2:]
#                     bboxes[..., :4].mul_(scale_tensor)
#                     out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
#         return out

#     def __call__(self, preds, batch):
#         """Calculate and return the loss for the YOLO model."""
#         loss = torch.zeros(3, device=self.device)  # box, cls, dfl
#         feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
#         batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc), 1
#         )

#         # b, grids, ..
#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()
#         pred_angle = pred_angle.permute(0, 2, 1).contiguous()

#         dtype = pred_scores.dtype
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # targets
#         try:
#             batch_idx = batch["batch_idx"].view(-1, 1)
#             targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
#             rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
#             targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
#             targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#             gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
#             # print(f"gt_bboxes shape: {gt_bboxes.shape}")
#             mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
#         except RuntimeError as e:
#             raise TypeError(
#                 "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
#                 "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
#                 "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
#                 "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
#                 "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
#             ) from e

#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

#         bboxes_for_assigner = pred_bboxes.clone().detach()
#         # Only the first four elements need to be scaled
#         bboxes_for_assigner[..., :4] *= stride_tensor
#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             pred_scores.detach().sigmoid(),
#             bboxes_for_assigner.type(gt_bboxes.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes,
#             mask_gt,
#         )
#         # print(f"target_bboxes shape: {target_bboxes.shape}")
#         target_scores_sum = max(target_scores.sum(), 1)

#         # Cls loss
#         # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
#         loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         # Bbox loss
#         # Bbox loss
#         if fg_mask.sum():

#             # -----------------------------------------------------
#             # 1. 原始的 IoU 和 DFL 损失计算 (必须在角度损失之前进行 RBox 坐标归一化)
#             # -----------------------------------------------------

#             # 必须在调用 self.bbox_loss 之前进行坐标归一化
#             target_bboxes[..., :4] /= stride_tensor

#             # 🚨 修复参数缺失：调用 self.bbox_loss 必须传入所有 7 个参数
#             loss_iou, loss_dfl = self.bbox_loss(
#                 pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
#             )

#             # 将 IoU loss 和 DFL loss 赋值给 loss[0] 和 loss[2]
#             loss[0], loss[2] = loss_iou, loss_dfl

#             # -----------------------------------------------------
#             # 2. 动态加权角度损失 (Weighted Angle Loss)
#             # -----------------------------------------------------

#             # 注意：target_bboxes 此时已被修改 (归一化了前4个参数)，但角度参数 target_bboxes[..., 4:5] 仍是正确的。

#             # 提取前景目标 RBox 的角度和宽高 (fg_mask 已经将维度扁平化)
#             # 角度
#             target_angles = target_bboxes[fg_mask][..., 4:5]
#             pred_angles = pred_angle[fg_mask]

#             # 权重 (target_scores.sum(-1)[fg_mask].unsqueeze(-1) 即为 IoU 损失的权重)
#             weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

#             # 提取归一化后的宽高 (用于计算宽高比)
#             gt_w = target_bboxes[fg_mask][..., 2:3]
#             gt_h = target_bboxes[fg_mask][..., 3:4]

#             # 计算 Aspect Ratio (AR = min/max)
#             ar = torch.min(gt_w, gt_h) / torch.max(gt_w, gt_h)

#             # 动态角度增益 (需要确保 self.hyp 中设置了这些值)
#             angle_gain_max = getattr(self.hyp, 'angle_gain', 0.5)
#             ar_threshold = getattr(self.hyp, 'ar_thresh', 0.8)

#             # 动态权重计算：当 AR 接近 1 时，增益变大
#             dynamic_mask = (ar >= ar_threshold).float()

#             # Dynamic_weight: 只有接近正方形的目标 (mask=1) 才会得到 angle_gain_max 的权重
#             # 增加一个小的基线权重 (0.005) 以确保长条形也能得到微弱监督
#             dynamic_weight = dynamic_mask * angle_gain_max + (1 - dynamic_mask) * 0.005

#             # 计算 90度周期角度损失 (不使用原始 IoU 权重进行归一化)
#             # 🚨 确保 circular_smooth_l1_loss_90deg 函数已经添加到 loss.py 文件中！
#             loss_angle_elements = circular_smooth_l1_loss_90deg(
#                 pred_angles, target_angles, torch.ones_like(weight)
#             )

#             # 最终的角度损失 = 元素级损失 * 动态增益 * 原始 IoU 权重 (target_scores 权重)
#             weighted_loss_angle = (loss_angle_elements * dynamic_weight * weight).sum() / target_scores_sum

#             # 3. 将最终的角度损失加到 box_loss 中
#             # loss[0] = IoU Loss + DFL Loss (如果 DFL 在 IoU 损失中) + Weighted Angle Loss
#             loss[0] += weighted_loss_angle

#         else:
#             loss[0] += (pred_angle * 0).sum()  # 确保在没有目标时张量保持非空

#         # 以下代码在 if/else 块之外
#         loss[0] *= self.hyp.box  # box gain (这是 IoU + Angle Loss 的总权重)
#         loss[1] *= self.hyp.cls  # cls gain
#         loss[2] *= self.hyp.dfl  # dfl gain

#         return loss.sum() * batch_size, loss.detach()

#     def bbox_decode(self, anchor_points, pred_dist, pred_angle):
#         """
#         Decode predicted object bounding box coordinates from anchor points and distribution.

#         Args:
#             anchor_points (torch.Tensor): Anchor points, (h*w, 2).
#             pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
#             pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

#         Returns:
#             (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
#         """
#         if self.use_dfl:
#             b, a, c = pred_dist.shape  # batch, anchors, channels
#             pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#         return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


# def circular_smooth_l1_loss(pred_angle, gt_angle, weight, sigma=1.0, beta = 1.0):
#     """
#     Calculates the Circular Smooth L1 Loss for periodic angle regression.
#     The loss handles the 180-degree periodicity of the angle.
#     """
#     # 确保角度在 [-pi, pi] 范围内 (取决于模型的编码)
#     diff = torch.abs(pred_angle - gt_angle)

#     # 将角度差映射到 [0, pi]
#     # OBB 角度通常是 90度周期，但我们使用一个更通用的 180 度处理来确保鲁棒性
#     # 假设角度单位是弧度 (rad)。如果是度数，需要调整常数。
#     # 这里使用 180 度周期 (pi rad)
#     diff = torch.min(diff, torch.pi / 2 - diff)

#     # Smooth L1 Loss
#     # |x| < sigma^2: 0.5 * x^2 * sigma^-2
#     # |x| >= sigma^2: |x| - 0.5 * sigma^2

#     # 假设使用标准的 Smooth L1 Loss (beta=1.0/sigma)
#     loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none', beta=beta)

#     # return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none') * weight
#     return loss * weight


# def circular_smooth_l1_loss_90deg(pred_angle, gt_angle, weight, beta=1.0):
#     """
#     计算处理 90 度周期性的角度 Smooth L1 损失。
#     这解决了 OBB 中 w/h 互换导致的 90 度周期性歧义。
#     (假设角度单位是弧度)
#     """

#     # 1. 计算角度差异
#     diff = torch.abs(pred_angle - gt_angle)

#     # 2. 90度周期性修正 (假设角度单位是弧度)
#     # 将角度差映射到 [0, pi/4] 范围内，处理 90 度周期性
#     diff = torch.min(diff, torch.pi / 2 - diff)

#     # 3. 计算 Smooth L1 Loss
#     # F.smooth_l1_loss(input, target, reduction='none', beta=beta)
#     # target 是 0，所以简化为对 diff 应用 Smooth L1
#     loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none', beta=beta)

#     return loss * weight


# class E2EDetectLoss:
#     """Criterion class for computing training losses."""

#     def __init__(self, model):
#         """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
#         self.one2many = v8DetectionLoss(model, tal_topk=10)
#         self.one2one = v8DetectionLoss(model, tal_topk=1)

#     def __call__(self, preds, batch):
#         """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
#         preds = preds[1] if isinstance(preds, tuple) else preds
#         one2many = preds["one2many"]
#         loss_one2many = self.one2many(one2many, batch)
#         one2one = preds["one2one"]
#         loss_one2one = self.one2one(one2one, batch)
#         return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
        self.angle_gain = float(getattr(self.hyp, "angle", 5.0))
        self.qg_angle_align = float(getattr(self.hyp, "qg_angle_align", 0.25))
        self.qg_angle_unit = float(getattr(self.hyp, "qg_angle_unit", 0.05))

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        # 【修改点 1】：初始化 loss 维度为 4 (box, cls, dfl, angle)
        loss = torch.zeros(4, device=self.device)  
        
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()
        pred_angle_raw = pred_angle
        pred_angle = self.decode_angle(pred_angle_raw)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset.\n"
                "See https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            # 1. 坐标归一化
            target_bboxes[..., :4] /= stride_tensor
            
            # 2. 计算原始的 IoU 和 DFL 损失
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            loss[0], loss[2] = loss_iou, loss_dfl

            # 3. 【修改点 2】：调用最新的平滑加权角度损失
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )
            loss[3] += self.calculate_angle_vector_loss(
                pred_angle_raw, target_bboxes, fg_mask, weight, target_scores_sum
            )

        else:
            loss[0] += (pred_angle_raw * 0).sum()  # 确保在没有目标时张量保持非空

        # 乘以各项的超参数增益权重
        loss[0] *= self.hyp.box  # box gain 
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        loss[3] *= self.angle_gain  # angle gain

        return loss.sum() * batch_size, loss.detach()

    def decode_angle(self, pred_angle):
        """Decode scalar theta or QG unit-cycle angle vectors to scalar theta."""
        if pred_angle.shape[-1] == 1:
            return pred_angle
        if pred_angle.shape[-1] == 2:
            return 0.5 * torch.atan2(pred_angle[..., 0:1], pred_angle[..., 1:2])
        raise RuntimeError(f"Unsupported OBB angle channels: {pred_angle.shape[-1]}")

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_vector_loss(self, pred_angle_raw, target_bboxes, fg_mask, weight, target_scores_sum):
        """Unit-cycle alignment loss for QG angle vectors."""
        if pred_angle_raw.shape[-1] != 2:
            return (pred_angle_raw * 0).sum()

        pred_vec = pred_angle_raw[fg_mask]
        target_theta = target_bboxes[..., 4][fg_mask]
        target_vec = torch.stack(
            (torch.sin(2 * target_theta), torch.cos(2 * target_theta)),
            dim=-1,
        ).to(dtype=pred_vec.dtype)

        pred_norm = pred_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pred_unit = pred_vec / pred_norm
        align_loss = 1.0 - (pred_unit * target_vec).sum(dim=-1)
        unit_loss = (pred_norm.squeeze(-1) - 1.0).pow(2)
        qg_loss = (self.qg_angle_align * align_loss + self.qg_angle_unit * unit_loss) * weight
        return qg_loss.sum() / target_scores_sum

    # 【修改点 3】：引入教科书级别的周期性角度损失逻辑
    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=5):
        """Calculate oriented angle loss.
        使用正弦平方函数处理 180 度周期性，梯度极其平滑，极利于 INT8 量化。
        """
        import math
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        # 1. 动态 AR 高斯权重 (控制长宽比的敏感度)
        log_ar = torch.log(w_gt / h_gt)
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        # 2. 完美的 180 度周期性折叠机制 (防止极值惩罚与负数 Bug)
        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        
        # 3. 平滑的正弦惩罚
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        # 4. 叠加动态 AR 权重与类别置信度权重
        # 💡提示: 如果你的 15 个类别里，正方形目标的角度标注全都是乱标的，
        # 请把下面这行改为：ang_loss = (1.0 - scale_weight[fg_mask]) * ang_loss
        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
