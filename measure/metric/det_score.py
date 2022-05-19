from torch import Tensor
from typing import Dict, Tuple, List
import numpy as np
import cv2 as cv
import copy
from shapely.geometry import Polygon
import pyclipper


class DetScore:
    def __init__(self,
                 totalBox: int,
                 edgeThresh: float,
                 probThresh: float,
                 scoreThresh: float,
                 label: str,
                 resize:bool):
        self._totalBox: int = totalBox
        self._edgeThresh: float = edgeThresh
        self._probThresh: float = probThresh
        self._scoreThresh: float = scoreThresh
        self._label: str = label
        self._resize:bool = resize

    def __call__(self, pred: Dict, batch: Dict) -> Tuple:
        '''
            :param pred: result of pred
            :param batch: ground truth
            :return: fit boxes  and its score
        '''
        # thresholding probability map
        probMaps: Tensor = pred[self._label]
        segMaps: Tensor = probMaps > self._probThresh
        orgShapes: Tensor = batch['orgShape']
        newShapes: Tensor = batch['newShape']

        boxes: List = []
        scores: List = []
        batchSize: int = batch['img'].size(0)
        for i in range(batchSize):
            box, score = self._finding(probMaps[i], segMaps[i], orgShapes[i], newShapes[i])
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def _finding(self, probMap: Tensor, segMap: Tensor, orgSize: Tensor, newSize: Tensor) -> Tuple:
        '''
            :param probMap: probability map
            :param segMap: binary map
            :param orgSize: origin size of image: (width, height)
            :param newSize: new size of image: (width, height)
            :return: min bounding box and its score
        '''
        probMatrix: np.ndarray = probMap.cpu().detach().numpy()[0]
        segMatrix: np.ndarray = segMap.cpu().detach().numpy()[0]
        featShape: Tuple = newSize.cpu().detach().numpy()
        orgShape: np.ndarray = orgSize.cpu().detach().numpy()
        # find countour inside segment matrix
        contours, _ = cv.findContours(np.uint8(segMatrix * 255.),
                                      cv.RETR_LIST,
                                      cv.CHAIN_APPROX_SIMPLE)
        # only get bounding inside limit
        totalContour: int = min(len(contours), self._totalBox)
        boxes: np.ndarray = np.zeros((totalContour, 4, 2), dtype=np.int32)
        scores: np.ndarray = np.zeros((totalContour,), dtype=np.float64)
        for i in range(totalContour):
            eps: float = 0.002 * cv.arcLength(contours[i], True)
            approx: np.ndarray = cv.approxPolyDP(contours[i], eps, True)
            contour: np.ndarray = approx.reshape((-1, 2)).astype(np.int32)
            if contour.shape[0] < 4:
                continue
            polygon = Polygon(contour)
            if not polygon.is_valid and not polygon.is_simple:
                continue
            score: float = self._calcBoxScore(probMatrix, contour)
            if score < self._scoreThresh:
                continue
            contour = self._expand(contour).reshape((-1, 2))
            bbox, minEdge = self._minBoxSurrounding(contour)
            if minEdge < self._edgeThresh:
                continue
            if self._resize:
                bbox[:, 0] = np.clip(bbox[:, 0] / featShape[1] * orgShape[1], 0, orgShape[1] - 1)
                bbox[:, 1] = np.clip(bbox[:, 1] / featShape[0] * orgShape[0], 0, orgShape[0] - 1)
            boxes[i, :, :] = bbox.astype(np.int32)
            scores[i] = score
        return boxes, scores

    def _minBoxSurrounding(self, contour: np.ndarray) -> Tuple:
        '''
            :param contour: contour surrounding area
            :return:min bounding box and min value of height and width
        '''
        bbox: Tuple = cv.minAreaRect(contour)
        points: List = sorted(list(cv.boxPoints(bbox)), key=lambda x: x[0])
        # 1 4
        tmp1: List = sorted([points[0], points[1]], key=lambda x: x[1])
        # 2 3
        tmp2: List = sorted([points[2], points[3]], key=lambda x: x[1])
        ans: List = [tmp1[0], tmp2[0], tmp2[1], tmp1[1]]
        return np.asarray(ans), min(bbox[1])

    def _checkCollapse(self, p1: List, p2: List):
        return p1[0] == p2[0] and p1[1] == p2[1]

    def _calcBoxScore(self, probMap: np.ndarray, bbox: np.ndarray) -> float:
        '''
        calculate score of bounding box inside probability map
            :param probMap: probability map
            :param bbox: bounding box coordinate
            :return: score of b
            ounding box in probability map
        '''
        h, w = probMap.shape[:2]
        copiedBox: np.ndarray = copy.deepcopy(bbox)
        xMin: np.ndarray = np.clip(np.floor(copiedBox[:, 0].min()).astype(np.int32), 0, w - 1)
        xMax: np.ndarray = np.clip(np.ceil(copiedBox[:, 0].max()).astype(np.int32), 0, w - 1)
        yMin: np.ndarray = np.clip(np.floor(copiedBox[:, 1].min()).astype(np.int32), 0, h - 1)
        yMax: np.ndarray = np.clip(np.ceil(copiedBox[:, 1].max()).astype(np.int32), 0, h - 1)

        copiedBox[:, 0] = copiedBox[:, 0] - xMin
        copiedBox[:, 1] = copiedBox[:, 1] - yMin
        mask = np.zeros((yMax - yMin + 1, xMax - xMin + 1), dtype=np.uint8)
        cv.fillPoly(mask, [copiedBox.reshape((-1, 2)).astype(np.int32)], 1)
        return cv.mean(probMap[yMin:yMax + 1, xMin:xMax + 1], mask=mask)[0]

    def _expand(self, bbox: np.ndarray, ratio: float = 2) -> np.ndarray:
        '''
        expanding bounding box by ratio
            :param bbox: bounding box point
            :param ratio: expand ratio
            :return: expanded bounding box
        '''
        polygon = Polygon(bbox)
        dist: float = polygon.area * ratio / polygon.length
        expand = pyclipper.PyclipperOffset()
        expand.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expandedBox: np.ndarray = np.array(expand.Execute(dist)[0])
        return expandedBox
