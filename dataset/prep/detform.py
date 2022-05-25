from collections import OrderedDict
import numpy as np
from typing import Dict, List


class DetForm:
    def __init__(self, shrinkRatio: float):
        self._shrinkRatio = shrinkRatio

    def __call__(self, data: Dict, isVisual: bool = False) -> OrderedDict:
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: Dict):
        print(data.keys())

    def _build(self, data: Dict) -> OrderedDict:
        '''
        :param data: a dict contain : anno, img, train, shape, tar
        :return: a ordered idict contain: img, polygon, shape, ignore, train
        '''
        polygon: List = []
        ignore: List = []
        anno: np.ndarray = data['target']
        img: np.ndarray = data['img']
        train: bool = data['train']
        orgShape: np.ndarray = np.array(data['orgShape'])

        for tar in anno:
            polygon.append(np.array(tar['polygon']).reshape((-1, 2)))
            ignore.append(tar['ignore'])

        return OrderedDict(
            img=img,
            polygon=polygon,
            orgShape=orgShape,
            ignore=np.array(ignore, dtype=np.uint8),
            train=train
        )
