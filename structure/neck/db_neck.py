import torch
from torch import nn, Tensor
from structure.neck.db_attention import AdaptiveScaleFusion
from typing import List


class DBNeck(nn.Module):
    def __init__(self, dataPoint: tuple, exp: int, bias: bool = False):
        super().__init__()

        assert len(dataPoint) >= 4, len(dataPoint)
        self.in5: nn.Module = nn.Conv2d(dataPoint[-1], exp, kernel_size=1, bias=bias)
        self.in4: nn.Module = nn.Conv2d(dataPoint[-2], exp, kernel_size=1, bias=bias)
        self.in3: nn.Module = nn.Conv2d(dataPoint[-3], exp, kernel_size=1, bias=bias)
        self.in2: nn.Module = nn.Conv2d(dataPoint[-4], exp, kernel_size=1, bias=bias)

        # Upsampling layer
        self.up5: nn.Module = nn.Upsample(scale_factor=2)
        self.up4: nn.Module = nn.Upsample(scale_factor=2)
        self.up3: nn.Module = nn.Upsample(scale_factor=2)

        expOutput: int = exp // 4
        self.out5: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest')
        )
        self.out4: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        self.out3: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.out2: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias)
        )
        self.adaptiveScale: nn.Module = AdaptiveScaleFusion(exp, expOutput)
        # Cài đặt giá trị ban đầu
        self.in5.apply(self.weight_init)
        self.in4.apply(self.weight_init)
        self.in3.apply(self.weight_init)
        self.in2.apply(self.weight_init)

        self.out5.apply(self.weight_init)
        self.out4.apply(self.weight_init)
        self.out3.apply(self.weight_init)
        self.out2.apply(self.weight_init)

    def weight_init(self, module):
        class_name = module.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.kaiming_normal_(module.weight.data)
        elif class_name.find("BatchNorm") != -1:
            module.weight.data.fill_(1.)
            module.bias.data.fill_(1e-4)

    def forward(self, input: List) -> Tensor:
        '''

        :param input: 4 feature with diffirent size: 1/32, 1/16, 1/8, 1/4
        :return: primitive probmap
        '''
        assert len(input) == 4, len(input)

        # input processing
        fin5: Tensor = self.in5(input[3])
        fin4: Tensor = self.in4(input[2])
        fin3: Tensor = self.in3(input[1])
        fin2: Tensor = self.in2(input[0])

        # upsampling processing
        fup4: Tensor = self.up5(fin5) + fin4
        fup3: Tensor = self.up4(fup4) + fin3
        fup2: Tensor = self.up3(fup3) + fin2

        # Output processing
        fout5: Tensor = self.out5(fin5)
        fout4: Tensor = self.out4(fup4)
        fout3: Tensor = self.out3(fup3)
        fout2: Tensor = self.out2(fup2)

        # Concatenate
        fusion: Tensor = torch.cat([fout5, fout4, fout3, fout2], 1)
        adaptiveFusion: Tensor = self.adaptiveScale(fusion, [fout5, fout4, fout3, fout2])
        return adaptiveFusion
