import time
import numpy as np
import torch
from measure import DetAcc

tmp = time.time()
detAcc = DetAcc(0.5, 0.5)
targets = {
    "polygon": torch.Tensor([[
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(2, 2), (3, 2), (3, 3), (2, 3)]
    ]]),
    "ignore": torch.Tensor([[False, False]])
}
preds = [[np.array([(0.1, 0.1), (1, 0), (1, 1), (0, 1)])]]
detAcc(preds, targets)
print(detAcc.gather())
print(time.time() - tmp)
