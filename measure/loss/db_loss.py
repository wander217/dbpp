import torch.nn as nn
from typing import Dict, Tuple
from measure.loss.bce_loss import BceLoss
from measure.loss.l1_loss import L1Loss
from measure.loss.dice_loss import DiceLoss
from collections import OrderedDict
from torch import Tensor


class DBLoss(nn.Module):
    def __init__(self,
                 threshScale: int,
                 threshLoss: Dict,
                 probScale: int,
                 probLoss: Dict,
                 binaryScale: int,
                 binaryLoss: Dict):
        super().__init__()
        self._probScale: int = probScale
        self._probLoss: BceLoss = BceLoss(**probLoss)

        self._threshScale: int = threshScale
        self._threshLoss: L1Loss = L1Loss(**threshLoss)

        self._binaryScale = binaryScale
        self._binaryLoss = DiceLoss(**binaryLoss)

    def __call__(self, pred: OrderedDict, batch: OrderedDict) -> Tuple:
        probDist: Tensor = self._probLoss(pred['probMap'],
                                          batch['probMap'],
                                          batch['probMask'])
        loss: Tensor = probDist
        lossDict: OrderedDict = OrderedDict(probLoss=probDist)
        if 'threshMap' in pred:
            threshDist: Tensor = self._threshLoss(pred['threshMap'],
                                                  batch['threshMap'],
                                                  batch['threshMask'])
            binaryDist: Tensor = self._binaryLoss(pred['binaryMap'],
                                                  batch['probMap'],
                                                  batch['probMask'])
            lossDict.update(threshLoss=threshDist,
                            binaryLoss=binaryDist)
            loss = self._binaryScale * binaryDist + \
                   self._threshScale * threshDist + \
                   self._probScale * probDist
        return loss, lossDict
