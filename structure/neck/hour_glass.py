from torch import Tensor
import torch.nn as nn
from typing import List


class HourGlass(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self._up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel))
        self._up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel))
        self._up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel))

        self._down3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self._down4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self._down5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, feature: List):
        f2, f3, f4, f5 = feature

        f5_up: Tensor = self._up5(f5) + f4
        f4_up: Tensor = self._up4(f5_up) + f3
        f3_up: Tensor = self._up3(f4_up) + f2

        f3_down: Tensor = self._down3(f3_up) + f4_up
        f4_down: Tensor = self._down4(f3_down) + f5_up
        f5_down: Tensor = self._down5(f4_down)

        f2 = f2 + f3_up
        f3 = f3 + f3_down
        f4 = f4 + f4_down
        f5 = f5 + f5_down

        return f2, f3, f4, f5
