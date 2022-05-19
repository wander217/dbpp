import torch.nn as nn
from measure import DBLoss
from structure import DBModel
from typing import Dict, OrderedDict, Tuple


class LossModel(nn.Module):
    def __init__(self, model: Dict, loss: Dict, device):
        super().__init__()
        self._device = device
        self._model: DBModel = DBModel(**model)
        self._model = self._model.to(self._device)
        self._loss: DBLoss = DBLoss(**loss)
        self._loss = self._loss.to(self._device)

    def forward(self, batch: OrderedDict, training: bool = True):
        if training:
            for key, value in batch.items():
                if (value is not None) and hasattr(value, 'to'):
                    batch[key] = value.to(self._device)
            pred: OrderedDict = self._model(batch['img'])
            loss, metric = self._loss(pred, batch)
            return pred, loss, metric
        batch['img'] = batch['img'].to(self._device)
        return self._model(batch['img'])
