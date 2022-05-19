from torch import nn, Tensor
from collections import OrderedDict
import torch


class DBHead(nn.Module):
    def __init__(self, k: int, exp: int, adaptive: bool, bias: bool = False):
        super().__init__()
        self.k: int = k
        self.adaptive: bool = adaptive

        exp_output: int = exp // 4
        self.prob: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2, bias=bias),
            nn.Sigmoid()
        )
        self.prob.apply(self.weight_init)
        if self.adaptive:
            self.thresh: nn.Module = nn.Sequential(
                nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
                nn.BatchNorm2d(exp_output),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2, bias=bias),
                nn.BatchNorm2d(exp_output),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2, bias=bias),
                nn.Sigmoid()
            )
            self.thresh.apply(self.weight_init)

    def weight_init(self, module):
        class_name = module.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.kaiming_normal_(module.weight.data)
        elif class_name.find("BatchNorm") != -1:
            module.weight.data.fill_(1.)
            module.bias.data.fill_(1e-4)

    def binarization(self, probMap: Tensor, threshMap: Tensor):
        return torch.reciprocal(1. + torch.exp(-self.k * (probMap - threshMap)))

    def forward(self, x: Tensor, training: bool = True) -> OrderedDict:
        result: OrderedDict = OrderedDict()
        # calculate probability map
        probMap: Tensor = self.prob(x)
        result.update(probMap=probMap)
        if self.adaptive and training:
            # calculate map
            threshMap: Tensor = self.thresh(x)
            binaryMap: Tensor = self.binarization(probMap, threshMap)
            result.update(binaryMap=binaryMap, threshMap=threshMap)
        return result
