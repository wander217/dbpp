import math
import copy
import torch
import time
from typing import List
from torch import nn, Tensor
from structure.backbone.element import MBConv, BNeckConfig, ConvNormActivation

# Setting from paper
NETWORK_SETTING: dict = {
    'b0': (1.0, 1.0, 0.2),
    'b1': (1.0, 1.1, 0.2),
    'b2': (1.1, 1.2, 0.3),
    'b3': (1.2, 1.4, 0.3),
}

DATAPOINT: List = [2, 3, 5, 8]


class DBEfficientNet(nn.Module):
    def __init__(self, netID: str, depthProb: float, useSE: bool):
        super().__init__()
        # Lấy setting từ trên
        wFactor, dFactor, _ = NETWORK_SETTING[netID]
        self.data_point: list = DATAPOINT
        self.config = [  # resolution output b3
            BNeckConfig(1, 3, 1, 32, 16, 1, wFactor, dFactor),  # (1): w/2 * h/2
            BNeckConfig(6, 3, 2, 16, 24, 2, wFactor, dFactor),  # (2): w/4 *h/4 -> Chọn
            BNeckConfig(6, 5, 2, 24, 40, 2, wFactor, dFactor),  # (3): w/8 *h/8 -> Chọn
            BNeckConfig(6, 3, 2, 40, 80, 3, wFactor, dFactor),  # (4): w/16 * h/16
            BNeckConfig(6, 5, 1, 80, 112, 3, wFactor, dFactor),  # (5): w/16 * h/16 -> Chọn
            BNeckConfig(6, 5, 2, 112, 192, 4, wFactor, dFactor),  # (6): w/32 * h/32
            BNeckConfig(6, 3, 1, 192, 320, 1, wFactor, dFactor),  # (7): w/32 * h/32
        ]
        firstOutChannel: int = self.config[0].inChannel
        self.layers: nn.ModuleList = nn.ModuleList([
            ConvNormActivation(in_channels=3,
                               out_channels=firstOutChannel,
                               kernel_size=3,
                               stride=2,
                               norm_layer=nn.BatchNorm2d,
                               activation_layer=nn.SiLU)
        ])  # (0): w/2 * h/2
        stageBlockId: int = 0
        totalStageBlock: int = sum([setting.totalLayer for setting in self.config])
        for i, setting in enumerate(self.config):
            stage: List = []
            for _ in range(setting.totalLayer):
                blockSetting = copy.copy(setting)
                if stage:
                    blockSetting.inChannel = blockSetting.outChannel
                    blockSetting.stride = 1
                newStochastic: float = depthProb * float(stageBlockId) / totalStageBlock
                stage.append(MBConv(blockSetting, newStochastic, nn.BatchNorm2d, useSE=useSE))
                stageBlockId += 1
            self.layers.append(nn.Sequential(*stage))
        # Xây dựng phần cuối
        lastInChannel: int = self.config[-1].outChannel
        lastOutChannel: int = 4 * lastInChannel
        self.layers.append(
            ConvNormActivation(
                in_channels=lastInChannel,
                out_channels=lastOutChannel,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU
            )
        )  # (8): w/32 * h/32 ->Chọn

        # Cài đặt giá trị ban đầu
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init_range = 1 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
                nn.init.zeros_(module.bias)

    def forward(self, input: Tensor) -> List:
        features: List = []
        for i in range(len(self.layers)):
            input = self.layers[i](input)
            if i in self.data_point:
                features.append(input)
        return features


# test
if __name__ == "__main__":
    model = DBEfficientNet(netID="b0", depthProb=0.2, useSE=False)
    a = torch.rand((1, 3, 320, 320), dtype=torch.float)
    start = time.time()
    b = model(a)
    for item in b:
        print(item.size())
    # print(b.size())
