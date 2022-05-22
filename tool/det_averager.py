class DetAverager:
    def __init__(self):
        self._value: float = 0
        self._counter: int = 0

    def update(self, value: float, counter: int = 1):
        self._value += value
        self._counter += counter

    def calc(self) -> float:
        if self._counter == 0:
            return 0
        return self._value / self._counter

    def reset(self):
        self._value = 0
        self._counter = 0