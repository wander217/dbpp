import math


def adjust_depth(totalLayer: int, dFactor: float):
    return int(math.ceil(totalLayer * dFactor))


def adjust_size(channel: int, expandRatio: float, divisor: int = 8):
    newChannel = max(divisor, int(channel * expandRatio + divisor/2) // divisor * divisor)
    if newChannel < 0.9 * channel:
        newChannel += divisor
    return newChannel


class BNeckConfig:
    def __init__(self,
                 expandRatio: float,
                 kernelSize: int,
                 stride: int,
                 inChannel: int,
                 outChannel: int,
                 totalLayer: int,
                 wFactor: float,
                 dFactor: float) -> None:
        self.expandRatio: float = expandRatio
        self.kernelSize: int = kernelSize
        self.stride: int = stride
        self.wFactor: float = wFactor
        self.dFactor: float = dFactor
        self.outChannel: int = adjust_size(outChannel, wFactor)
        self.inChannel: int = adjust_size(inChannel, wFactor)
        self.totalLayer: int = adjust_depth(totalLayer, dFactor)
