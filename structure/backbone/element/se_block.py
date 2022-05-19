from torch import nn, Tensor


class SEBlock(nn.Module):
    def __init__(self, inChannel: int, reduction: int = 4, bias: bool = False):
        super().__init__()
        expChannel: int = max(inChannel // reduction, 1)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inChannel, expChannel, kernel_size=1, bias=bias),
            nn.SiLU(inplace=True),
            nn.Conv2d(expChannel, inChannel, kernel_size=1, bias=bias),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, input: Tensor) -> Tensor:
        scale = self.block(input)
        return scale * input
