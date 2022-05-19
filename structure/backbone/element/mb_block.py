from typing import Callable, List
from torch import nn, Tensor
from structure.backbone.element.se_block import SEBlock
from structure.backbone.element.bneck_config import BNeckConfig, adjust_size
from torchvision.ops.misc import ConvNormActivation
from torchvision.ops import StochasticDepth


class MBConv(nn.Module):
    def __init__(self,
                 config: BNeckConfig,
                 depthProb: float,
                 normLayer: Callable[..., nn.Module],
                 seLayer: Callable[..., nn.Module] = SEBlock,
                 useSE: bool = True) -> None:
        super().__init__()
        # assert config.stride in (1, 2), config.stride
        self._isResidual = (config.stride == 1) and (config.inChannel == config.outChannel)
        blocks: List = []
        expChannel: int = adjust_size(config.inChannel, config.expandRatio)
        if expChannel != config.inChannel:
            blocks.append(
                ConvNormActivation(
                    in_channels=config.inChannel,
                    out_channels=expChannel,
                    kernel_size=1,
                    norm_layer=normLayer,
                    activation_layer=nn.SiLU
                )
            )
        # deepthwise
        blocks.append(
            ConvNormActivation(
                in_channels=expChannel,
                out_channels=expChannel,
                kernel_size=config.kernelSize,
                stride=config.stride,
                groups=expChannel,
                norm_layer=normLayer,
                activation_layer=nn.SiLU
            )
        )
        # SE
        if useSE:
            blocks.append(seLayer(inChannel=expChannel))
        # project
        blocks.append(
            ConvNormActivation(
                in_channels=expChannel,
                out_channels=config.outChannel,
                kernel_size=1,
                norm_layer=normLayer,
                activation_layer=nn.SiLU
            )
        )
        self.layers = nn.Sequential(*blocks)
        self.stochasticDepth = StochasticDepth(depthProb, "row")
        self.output_channel = config.outChannel

    def forward(self, input: Tensor) -> Tensor:
        output = self.layers(input)
        if self._isResidual:
            output = self.stochasticDepth(output)
            output += input
        return output
