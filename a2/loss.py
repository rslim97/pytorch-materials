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
    

# class FocalLoss(nn.Module):
#     """
#     https://discuss.pytorch.org/t/using-focal-loss-for-multilabel-classification-problem/206839/4
#     """
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         else:
#             eps = 1e-7
#             inputs = torch.clamp(inputs, min=eps, max=1.0 - eps)
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
