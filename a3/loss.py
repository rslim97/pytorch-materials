import torch
import torch.nn as nn


# class FocalLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, inputs, targets):
#         pred = torch.sigmoid(inputs)
#         pos_inds = targets.eq(1).float()
#         neg_inds = targets.lt(1).float()
#         loss = 0
#         neg_weights = torch.pow(1 - targets, 4)
#         pos_loss = pos_inds * torch.pow(1 - pred, 2) * torch.log(pred)
#         neg_loss = neg_inds * neg_weights * torch.pow(pred, 2) * torch.log(1 - pred)

#         num_pos = pos_inds.float().sum()
#         pos_loss = pos_loss.sum()  # Reduction
#         neg_loss = neg_loss.sum()

#         if num_pos == 0:
#             loss -= neg_loss
#         else:
#             loss -= pos_loss + neg_loss
#         return loss


# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**2 * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**4 * (inputs)**2 * torch.log(1.0 - inputs + 1e-14)
        return center_loss + other_loss