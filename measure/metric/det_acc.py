from typing import List, Dict
import numpy as np
from torch import Tensor
from shapely.geometry import Polygon
from tool.det_averager import DetAverager


class DetAcc:
    def __init__(self, ignoreThresh: float, accThresh: float, scoreThresh: float):
        self._ignoreThresh: float = ignoreThresh
        self._accThresh: float = accThresh
        self._scoreThresh: float = scoreThresh
        self._result: List = []

    def __call__(self, predBoxes: np.ndarray, scores: np.ndarray, batch: Dict):
        targetBoxes: Tensor = batch['bbox']
        ignores: Tensor = batch['ignore']
        for targetBox, ignore, predBox, score in zip(targetBoxes, ignores, predBoxes, scores):
            target: List = [dict(polygon=targetBox[i], ignore=ignore[i]) for i in range(len(targetBox))]
            pred: List = [dict(polygon=predBox[i].tolist()) for i in range(predBox.shape[0]) if
                          score[i] > self._scoreThresh]
            self._result.append(self._evaluate(pred, target))

    def _union(self, polygon1, polygon2) -> float:
        return Polygon(polygon1).union(Polygon(polygon2)).area

    def _intersection(self, polygon1, polygon2) -> float:
        return Polygon(polygon1).intersection(Polygon(polygon2)).area

    def _iou(self, polygon1, polygon2) -> float:
        upper = self._intersection(polygon1, polygon2)
        lower = self._union(polygon1, polygon2)
        return upper / lower

    def _evaluate(self, pred: List, target: List) -> dict:
        targetPolygon: List = []
        targetIgnore: List = []
        predPolygon: List = []
        predIgnore: List = []
        totalMatch: int = 0

        for i in range(len(target)):
            polygon: Tensor = target[i]['polygon']
            ignore: Tensor = target[i]['ignore']
            polygonShape = Polygon(polygon)
            if not polygonShape.is_valid or not polygonShape.is_simple:
                continue
            targetPolygon.append(polygon)
            if ignore:
                targetIgnore.append(len(targetIgnore) - 1)

        # filtering predicted result
        for i in range(len(pred)):
            polygon: Tensor = pred[i]['polygon']
            polygonShape = Polygon(polygon)
            if not polygonShape.is_valid or not polygonShape.is_simple:
                continue
            predPolygon.append(polygon)
            if len(targetIgnore) > 0:
                for pos in targetIgnore:
                    ignoredPolygon = targetPolygon[targetIgnore[pos]]
                    intersection = self._intersection(polygon, ignoredPolygon)
                    area = Polygon(ignoredPolygon).area
                    percent = 0 if area == 0 else intersection / area
                    if percent > self._ignoreThresh:
                        predIgnore.append(len(predPolygon) - 1)
                        break

        if len(predPolygon) > 0 and len(targetPolygon) > 0:
            iouTable: np.ndarray = np.empty([len(targetPolygon), len(predPolygon)])
            predMask: np.ndarray = np.zeros((len(predPolygon),), dtype=np.uint8)
            targetMask: np.ndarray = np.zeros((len(targetPolygon),), dtype=np.uint8)
            for i in range(len(targetPolygon)):
                for j in range(len(predPolygon)):
                    iouTable[i][j] = self._iou(targetPolygon[i], predPolygon[j])
            for i in range(len(targetPolygon)):
                for j in range(len(predPolygon)):
                    if targetMask[i] == 0 and i not in targetIgnore \
                            and predMask[j] == 0 and j not in predIgnore:
                        if iouTable[i][j] > self._accThresh:
                            targetMask[i] = 1
                            predMask[j] = 1
                            totalMatch = totalMatch + 1
                            break
        totalTarget: int = len(targetPolygon) - len(targetIgnore)
        totalPred: int = len(predPolygon) - len(predIgnore)
        return dict(
            totalMatch=totalMatch,
            totalTarget=totalTarget,
            totalPred=totalPred
        )

    def gather(self):
        precision: DetAverager = DetAverager()
        recall: DetAverager = DetAverager()
        f1score: DetAverager = DetAverager()
        for element in self._result:
            totalMatch: int = element['totalMatch']
            totalTarget: int = element['totalTarget']
            totalPred: int = element['totalPred']
            p: float = 0 if totalPred == 0 else totalMatch / totalPred
            precision.update(p)
            r: float = 0 if totalTarget == 0 else totalMatch / totalTarget
            recall.update(r)
            f: float = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            f1score.update(f)
        self._result.clear()
        return dict(precision=precision.calc(),
                    recall=recall.calc(),
                    f1score=f1score.calc())
