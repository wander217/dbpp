from collections import OrderedDict
from torch import nn, Tensor
from structure.backbone import DBEfficientNet
from structure.neck import DBNeck
from structure.head import DBHead
from typing import List
import torch
import time
import yaml
import math


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        init_range = 1 / math.sqrt(module.out_features)
        nn.init.uniform_(module.weight, -init_range, init_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = DBEfficientNet(**kwargs['backbone'])
        self.neck = DBNeck(**kwargs['neck'])
        self.head = DBHead(**kwargs['head'])
        self.apply(weight_init)

    def forward(self, x: Tensor, training: bool = True) -> OrderedDict:
        brs: List = self.backbone(x)
        nrs: Tensor = self.neck(brs)
        hrs: OrderedDict = self.head(nrs, training)
        return hrs


# test
if __name__ == "__main__":
    file_config: str = r'D:\db_pp\config\dbpp_eb0.yaml'
    with open(file_config) as stream:
        data = yaml.safe_load(stream)
    model = DBModel(**data['lossModel']['model'])
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:', total_params)
    print('train params:', train_params)
    a = torch.rand((1, 3, 640, 640), dtype=torch.float)
    start = time.time()
    b = model(a)
    print('run:', time.time() - start)
