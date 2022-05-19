import numpy as np
import pyclipper
from shapely.geometry import Polygon
import cv2 as cv
from typing import List, OrderedDict


class ThreshMaker:
    def __init__(self, expandRatio: float, minThresh: float, maxThresh: float):
        self._expandRatio: float = expandRatio
        self._minThresh: float = minThresh
        self._maxThresh: float = maxThresh

    def __call__(self, data: OrderedDict, isVisual: bool = False) -> OrderedDict:
        output = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, output: OrderedDict):
        print(output.keys())
        threshMap = np.uint8(output['threshMap'] * 255)
        threshMask = np.uint8(output['threshMask'] * 255)
        cv.imshow("thresh mask", threshMask)
        cv.imshow("thresh map", threshMap)

    def _build(self, data: OrderedDict) -> OrderedDict:
        img: np.ndarray = data['img']
        polygons: List = data['polygon']
        ignores: np.ndarray = data['ignore']

        threshMap = np.zeros(img.shape[:2], dtype=np.float64)
        threshMask = np.zeros(img.shape[:2], dtype=np.float64)

        for i in range(len(polygons)):
            if not ignores[i]:
                self._calc(polygons[i], threshMap, threshMask)
        threshMap = threshMap * (self._maxThresh - self._minThresh) + self._minThresh
        data.update(threshMap=threshMap, threshMask=threshMask)
        return data

    def _calc(self, polygon: np.ndarray, threshMap: np.ndarray, threshMask: np.ndarray) -> tuple:
        '''
            Hàm tính thresh_map và thresh_mask cho từng polygon
        '''
        polygonShape = Polygon(polygon)
        dist = polygonShape.area * (1 - np.power(self._expandRatio, 2)) / polygonShape.length
        subject: List = [tuple(point) for point in polygon]
        expand = pyclipper.PyclipperOffset()
        expand.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padPolygon: np.ndarray = np.array(expand.Execute(dist)[0])
        cv.fillPoly(threshMask, [padPolygon.astype(np.int32)], 1.)

        x_min: int = padPolygon[:, 0].min()
        x_max: int = padPolygon[:, 0].max()
        y_min: int = padPolygon[:, 1].min()
        y_max: int = padPolygon[:, 1].max()
        w: int = x_max - x_min + 1
        h: int = y_max - y_min + 1

        polygon[:, 0] = polygon[:, 0] - x_min
        polygon[:, 1] = polygon[:, 1] - y_min

        x_axis: np.ndarray = np.broadcast_to(np.linspace(0, w - 1, num=w).reshape((1, w)), (h, w))
        y_axis: np.ndarray = np.broadcast_to(np.linspace(0, h - 1, num=h).reshape((h, 1)), (h, w))
        distMap: np.ndarray = np.zeros((polygon.shape[0], h, w), dtype=np.float64)
        for i in range(polygon.shape[0]):
            i_n = (i + 1) % polygon.shape[0]
            absDist = self._calcDist(x_axis, y_axis, polygon[i], polygon[i_n])
            distMap[i] = np.clip(absDist / dist, 0., 1.)
        dist_map = np.min(distMap, axis=0)
        x_valid_min: int = min(max(x_min, 0), threshMap.shape[1] - 1)
        x_valid_max: int = min(max(x_max, 0), threshMap.shape[1] - 1)
        y_valid_min: int = min(max(y_min, 0), threshMap.shape[0] - 1)
        y_valid_max: int = min(max(y_max, 0), threshMap.shape[0] - 1)
        threshMap[y_valid_min:y_valid_max + 1, x_valid_min:x_valid_max + 1] = np.fmax(
            1 - dist_map[y_valid_min - y_min:y_valid_max - y_max + h,
                         x_valid_min - x_min:x_valid_max - x_max + w],
            threshMap[y_valid_min:y_valid_max + 1, x_valid_min:x_valid_max + 1])
        return threshMap, threshMask

    def _calcDist(self, xAxis: np.ndarray, yAxis: np.ndarray, p1: List, p2: List) -> np.ndarray:
        '''
            Tính khoảng cách từ mọi điểm trên trục x,y đến đoạn thẳng p1p2
        '''
        sq_dist_1: np.ndarray = np.square(xAxis - p1[0]) + np.square(yAxis - p1[1])
        sq_dist_2: np.ndarray = np.square(xAxis - p2[0]) + np.square(yAxis - p2[1])
        sq_dist_3: np.ndarray = np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])
        cosin: np.ndarray = (sq_dist_1 + sq_dist_2 - sq_dist_3) / (2. * np.sqrt(sq_dist_1 * sq_dist_2))
        sq_sin: np.ndarray = 1 - np.square(cosin)
        sq_sin = np.nan_to_num(sq_sin)
        dist: np.ndarray = np.sqrt(sq_dist_1 * sq_dist_2 * sq_sin / sq_dist_3)
        dist[cosin >= 0] = np.sqrt(np.fmin(sq_dist_1, sq_dist_2))[cosin >= 0]
        return dist
