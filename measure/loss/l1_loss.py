import torch.nn as nn
from torch import Tensor
import torch


class L1Loss(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self._eps = eps

    def __call__(self,
                 pred: Tensor,
                 threshMap: Tensor,
                 threshMask: Tensor):
        totalElement: Tensor = threshMask.sum()
        if totalElement.item() == 0:
            return totalElement
        loss: Tensor = (torch.abs(pred[:, 0].float() - threshMap) * threshMask).sum() / totalElement
        return loss
