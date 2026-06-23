import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        pred = torch.sigmoid(inputs)
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        loss = 0
        neg_weights = torch.pow(1 - targets, 4)
        pos_loss = pos_inds * torch.pow(1 - pred, 2) * torch.log(pred)
        neg_loss = neg_inds * neg_weights * torch.pow(pred, 2) * torch.log(1 - pred)

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()  # Reduction
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss -= neg_loss
        else:
            loss -= pos_loss + neg_loss
        return loss
