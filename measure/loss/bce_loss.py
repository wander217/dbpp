import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch


class BceLoss(nn.Module):
    def __init__(self, ratio: float, eps: float):
        super().__init__()
        self._ratio: float = ratio
        self._eps: float = eps

    def __call__(self,
                 pred: Tensor,
                 probMap: Tensor,
                 probMask: Tensor) -> Tensor:
        pos: Tensor = (probMap * probMask).byte()
        neg: Tensor = ((1 - probMap) * probMask).byte()
        loss: Tensor = F.binary_cross_entropy(pred.float(), probMap, reduction='none')[:, 0, :, :]

        posNum: int = int(pos.float().sum())
        posLoss: Tensor = loss * pos.float()
        negNum: int = min(int(neg.sum()), int(posNum * self._ratio))
        negLoss: Tensor = loss * neg.float()
        negLoss, _ = torch.topk(negLoss.view(-1), negNum)
        loss: Tensor = (posLoss.sum() + negLoss.sum()) / (posNum + negNum + self._eps)
        return loss
