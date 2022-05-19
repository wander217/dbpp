from typing import List, OrderedDict


class DetFilter:
    def __init__(self, key: List):
        self.key: List = key

    def __call__(self, data: OrderedDict, isVisual: bool = False) -> OrderedDict:
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: OrderedDict):
        print(data.keys())

    def _build(self, data: OrderedDict) -> OrderedDict:
        for item in self.key:
            if item in data:
                del data[item]
        return data
