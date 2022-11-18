from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):

    def __init__(self, T):
        super(DistillationLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss

class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        loss = 2 - 2 * (x * y).sum(dim=-1)

        return loss
