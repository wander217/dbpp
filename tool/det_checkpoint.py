import os.path
import os
import torch
from collections import OrderedDict
import torch.nn as nn
from torch import optim
from typing import Any, Tuple


class DetCheckpoint:
    def __init__(self, workspace: str, resume: str):
        if not os.path.isdir("workspace"):
            os.mkdir("workspace")
        if not os.path.isdir(os.path.join("workspace", workspace)):
            os.mkdir(os.path.join("workspace", workspace))
        self._workspace: str = os.path.join("workspace", workspace)
        self._resume: str = resume.strip()

    def saveCheckpoint(self,
                       epoch: int,
                       model: nn.Module,
                       optim: optim.Optimizer):
        lastPath: str = os.path.join(self._workspace, "last.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch
        }, lastPath)

    def saveModel(self, model: nn.Module, step: int) -> Any:
        path: str = os.path.join(self._workspace, "checkpoint_{}.pth".format(step))
        torch.save({"model": model.state_dict()}, path)

    def load(self, device=torch.device('cpu')):
        if isinstance(self._resume, str) and bool(self._resume):
            data: OrderedDict = torch.load(self._resume, map_location=device)
            model: OrderedDict = data.get('model')
            optim: OrderedDict = data.get('optimizer')
            epoch: int = data.get('epoch')
            step: int = data.get('step')
            return model, optim, epoch, step

    def loadPath(self, path: str, device=torch.device('cpu')) -> OrderedDict:
        data: OrderedDict = torch.load(path, map_location=device)
        assert 'model' in data
        model: OrderedDict = data.get('model')
        return model
