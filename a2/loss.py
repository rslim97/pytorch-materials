import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    """

    def __init__(self, weight=None, gamma=2.0, reduction="none"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # input: (N, C, H, W)
        # target: (N, C, H, W)
        log_prob = F.log_softmax(input, dim=1)
        prob = torch.exp(log_prob)
        # print('log_prob', log_prob.shape)
        # print('prob.shape', prob.shape)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )
