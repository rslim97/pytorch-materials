import numpy as np
import torch
import torch.nn as nn
import json
import tempfile
from dataset import Dataset
from torch.utils.data import DataLoader
import cv2

from loss import FocalLoss


""" 
Legacy code following from CornerNet
Resulted from completing the square of IoU calculation,
to get the positive roots of the equations.
https://zhuanlan.zhihu.com/p/482584449
"""


def gaussian_radius(det_size, min_overlap=0.7):
    box_h, box_w = det_size
    a1 = 1
    b1 = box_h + box_w
    c1 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (box_h + box_w)
    c2 = (1 - min_overlap) * box_w * box_h
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_h + box_w)
    c3 = (min_overlap - 1) * box_w * box_h
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


# Map the original bounding box to the feature map
def generate_txtytwth(bbox, w, h, s):
    """
    Center point coordinates: (px_grid, py_grid)
    Offset: (tx, ty)
    Widh, Height: (tw, th)
    2D Gaussian function variance: sigma_w, sigma_h
    """
    xmin, ymin, xmax, ymax = bbox
    # xyxy format to cxcywh
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    box_w_s = box_w / s  # float
    box_h_s = box_h / s

    r = gaussian_radius([box_w_s, box_h_s])
    sigma_w = sigma_h = r / 3

    if box_w < 1e-28 or box_h < 1e-28:
        return False

    c_x_s = c_x / s
    c_y_s = c_y / s
    px_grid = int(c_x_s)  # take floor or int
    py_grid = int(c_y_s)

    tx = c_x_s - px_grid  # float
    ty = c_y_s - py_grid
    tw = np.log(box_w_s)  # float
    th = np.log(box_h_s)

    return px_grid, py_grid, tx, ty, tw, th, sigma_w, sigma_h


# Create a Gaussian heatmap and generate usable annotation information
def gt_creator(input_size, stride, num_classes, label_lists=[]):
    batch_size = len(label_lists)
    w = input_size
    h = input_size

    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, num_classes + 4 + 1])
    for batch_idx in range(batch_size):
        for gt_label in label_lists[batch_idx]:
            gt_cls = gt_label[-1]
            bbox = gt_label[:-1]
            result = generate_txtytwth(bbox, w, h, s)
            if result:
                px_grid, py_grid, tx, ty, tw, th, sigma_w, sigma_h = result
                # print(result)
                gt_tensor[batch_idx, py_grid, px_grid, int(gt_cls)] = 1.0
                gt_tensor[
                    batch_idx, py_grid, px_grid, num_classes : num_classes + 4
                ] = np.array([tx, ty, tw, th])
                gt_tensor[batch_idx, py_grid, px_grid, num_classes + 4] = 1.0

                # Create a Gaussian heatmap
                for i in range(
                    px_grid - 3 * int(sigma_w), px_grid + 3 * int(sigma_w) + 1
                ):
                    for j in range(
                        py_grid - 3 * int(sigma_h), py_grid + 3 * int(sigma_h) + 1
                    ):
                        if i < ws and j < hs:
                            g = np.exp(
                                -((i - px_grid) ** 2 + (j - py_grid) ** 2)
                                / (2 * sigma_h**2)
                            )
                            prev_g = gt_tensor[batch_idx, j, i, int(gt_cls)]
                            gt_tensor[batch_idx, j, i, int(gt_cls)] = max(g, prev_g)

    gt_tensor = gt_tensor.reshape(batch_size, -1, num_classes + 4 + 1)

    return gt_tensor


def get_loss(pred_cls, pred_txty, pred_twth, label, num_classes):
    cls_loss_func = FocalLoss()  # scalar tensor
    txty_loss_func = nn.BCEWithLogitsLoss(reduction="none")  # (N, hs * ws, 2)
    twth_loss_func = nn.SmoothL1Loss(reduction="none")  # (N, hs * ws, 2)

    gt_cls = label[:, :, :num_classes].float()
    gt_txtytwth = label[:, :, num_classes:-1].float()
    gt_box_conf = label[:, :, -1]

    N = pred_cls.shape[0]
    cls_loss = torch.sum(cls_loss_func(pred_cls, gt_cls)) / N
    # print('txty_loss_func(pred_txty, gt_txtytwth[:, :, :2]).shape', txty_loss_func(pred_txty, gt_txtytwth[:, :, :2]).shape)  
    # print('twth_loss_func(pred_twth, gt_txtytwth[:, :, 2:]).shape', twth_loss_func(pred_twth, gt_txtytwth[:, :, 2:]).shape)
    # print('cls_loss_func(pred_cls, gt_cls)', cls_loss_func(pred_cls, gt_cls))
    txty_loss = torch.sum(
        torch.sum(txty_loss_func(pred_txty, gt_txtytwth[:, :, :2]), dim=2) * gt_box_conf
    )
    twth_loss = torch.sum(
        torch.sum(twth_loss_func(pred_twth, gt_txtytwth[:, :, 2:]), dim=2) * gt_box_conf
    )

    total_loss = cls_loss + txty_loss + twth_loss
    return total_loss


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1].clone().detach().requires_grad_(True))
    return torch.stack(imgs, 0), targets
