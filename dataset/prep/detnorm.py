import numpy as np
from typing import List, OrderedDict


class DetNorm:
    def __init__(self, mean: List):
        self.mean: np.ndarray = np.array(mean)

    def __call__(self, data: OrderedDict, isVisual: bool = False):
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: OrderedDict):
        print(data.keys())
        print(data['img'].shape)

    def _build(self, data: OrderedDict) -> OrderedDict:
        assert 'img' in data
        image: np.ndarray = data['img'].astype(np.float64)
        image = (image - self.mean) / 255.
        data['img'] = np.transpose(image, (2, 0, 1))
        return data
