import os.path
import os
import torch
from collections import OrderedDict
import torch.nn as nn
from torch import optim
from typing import Any, Tuple


class DetCheckpoint:
    def __init__(self, workspace: str, resume: str):
        self._workspace: str = workspace
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self._resume: str = resume.strip()

    def saveCheckpoint(self,
                       epoch: int,
                       model: nn.Module,
                       optim: optim.Optimizer,
                       scheduler=None):
        lastPath: str = os.path.join(self._workspace, "last.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch
        }, lastPath)

    def saveModel(self, model: nn.Module, epoch: int) -> Any:
        path: str = os.path.join(self._workspace, "checkpoint_{}.pth".format(epoch))
        torch.save({"model": model.state_dict()}, path)

    def load(self, device=torch.device('cpu')):
        if isinstance(self._resume, str) and bool(self._resume):
            data: OrderedDict = torch.load(self._resume, map_location=device)
            model: OrderedDict = data.get('model')
            optim: OrderedDict = data.get('optimizer')
            epoch: int = data.get('epoch')
            scheduler: OrderedDict = data.get('scheduler')
            return model, optim, epoch, scheduler

    def loadPath(self, path: str, device=torch.device('cpu')) -> OrderedDict:
        data: OrderedDict = torch.load(path, map_location=device)
        assert 'model' in data
        model: OrderedDict = data.get('model')
        return model
