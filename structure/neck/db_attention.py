import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, inChannel: int, expChannel: int, outChannel: int = 4):
        super().__init__()
        self.conv: nn.Module = nn.Conv2d(in_channels=inChannel,
                                         out_channels=expChannel,
                                         kernel_size=3,
                                         padding=1,
                                         bias=False)
        self.channelWise: nn.Module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=expChannel,
                      out_channels=expChannel // 4,
                      kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=expChannel // 4,
                      out_channels=expChannel,
                      kernel_size=1,
                      bias=False),
            nn.Sigmoid()
        )
        self.spatialWise: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=1,
                      bias=False),
            nn.Sigmoid()
        )
        self.attnWise: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=expChannel,
                      out_channels=outChannel,
                      kernel_size=1,
                      bias=False),
            nn.Sigmoid()
        )
        self._outChannel: int = outChannel
        self._initWeight()

    def _initWeight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.ones_(module.bias)

    def forward(self, input: Tensor, features: List):
        # (N,exp,h/4,w/4)
        output: Tensor = self.conv(input)
        # (N,exp,h/4, w/4)
        global_x: Tensor = self.channelWise(output) + output
        # (N,1,h/4, w/4)
        x: Tensor = torch.mean(global_x, dim=1, keepdim=True)
        # (N,exp,h/4, w/4)
        global_x = global_x + self.spatialWise(x)
        # (N,4,h/4,w/4)
        score: Tensor = self.attnWise(global_x)
        attnFeatures: List = []
        for i in range(self._outChannel):
            attnFeatures.append(score[:, i:i + 1] * features[i])
        return torch.cat(attnFeatures, dim=1)
