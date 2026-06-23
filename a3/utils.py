import numpy as np
import torch
import torch.nn as nn
import json
import tempfile
from dataset import Dataset
from torch.utils.data import DataLoader
import cv2

from loss import FocalLoss


# Legacy code following from CornerNet
# Resulted from completing the square of IoU calculation,
# to get the positive roots of the equations.
def gaussian_radius(det_size, min_overlap=0.7):
    box_h, box_h = det_size
    a3 = 1
    b3 = box_h + box_h
    c3 = box_h * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2 * a3

    a2 = 4
    b2 = 2 * (box_h + box_h)
    c2 = (1 - min_overlap) * box_h * box_h
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2 * a2

    a1 = 4 * min_overlap
    b1 = -2 * min_overlap * (box_h + box_h)
    c1 = (min_overlap - 1) * box_h * box_h
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2 * a1

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

    px_s = c_x / s
    py_s = c_y / s
    px_grid = int(px_s)  # take floor or int
    py_grid = int(py_s)

    tx = px_s - px_grid  # float
    ty = py_s - py_grid
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
    cls_loss_func = FocalLoss()
    txty_loss_func = nn.BCEWithLogitsLoss(reduction="none")
    twth_loss_func = nn.SmoothL1Loss(reduction="none")

    gt_cls = label[:, :, :num_classes].float()
    gt_txtytwth = label[:, :, num_classes:-1].float()
    gt_box_conf = label[:, :, -1]

    N = pred_cls.shape[0]
    cls_loss = torch.sum(cls_loss_func(pred_cls, gt_cls)) / N
    txty_loss = torch.sum(
        torch.sum(txty_loss_func(pred_txty, gt_txtytwth[:, :, :2]), dim=2) * gt_box_conf
    )
    twth_loss = torch.sum(
        torch.sum(twth_loss_func(pred_twth, gt_txtytwth[:, :, 2:]), dim=2) * gt_box_conf
    )

    total_loss = cls_loss + txty_loss + twth_loss
    return total_loss
