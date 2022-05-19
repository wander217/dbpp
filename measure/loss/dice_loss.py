import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self._eps: float = eps

    def __call__(self,
                 pred: Tensor,
                 probMap: Tensor,
                 probMask: Tensor):
        pred = pred[:, 0, :, :].float()
        probMap = probMap[:, 0, :, :].float()
        intersection: Tensor = (pred.float() * probMap * probMask).sum()
        uninon: Tensor = (probMap * probMask).sum() + (pred * probMask).sum() + self._eps
        loss: Tensor = 1. - 2. * intersection / uninon
        assert loss <= 1., loss
        return loss
