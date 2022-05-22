import torch
from torch import nn, Tensor
from structure.neck.db_attention import AdaptiveScaleFusion
from typing import List
from structure.neck.hour_glass import HourGlass


class DBNeck(nn.Module):
    def __init__(self, data_point: tuple, exp: int, bias: bool = False):
        super().__init__()

        self._ins: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(data_point[i], exp, kernel_size=1, bias=bias)
            for i in range(len(data_point))
        ])
        self._hour_glass: nn.Module = nn.Sequential(*[
            HourGlass(exp, exp) for _ in range(2)
        ])
        expOutput = exp // 4
        self.adaptiveScale: nn.Module = AdaptiveScaleFusion(exp, expOutput)

    def forward(self, feature: List) -> Tensor:
        '''

        :param feature: 4 feature with diffirent size: 1/32, 1/16, 1/8, 1/4
        :return: primitive probmap
        '''
        # input processing
        for i in range(len(self._ins)):
            feature[i] = self._ins[i](feature[i])
        # up sampling processing
        feature = self._hour_glass(feature)
        # Concatenate
        return feature[0]
