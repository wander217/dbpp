import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
import pyclipper
from typing import List, OrderedDict
from collections import OrderedDict


class ProbMaker:
    def __init__(self, minTextSize: int, shrinkRatio: float):
        self._minTextSize = minTextSize
        self._shrinkRatio = shrinkRatio

    def __call__(self, data: OrderedDict, isVisual: bool = False) -> OrderedDict:
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: OrderedDict):
        print(data.keys())
        probMap = np.uint8(data['probMap'][0] * 255)
        probMask = np.uint8(data['probMask'] * 255)
        cv.imshow("probMap", probMap)
        cv.imshow("probMask", probMask)

    def _build(self, data: OrderedDict) -> OrderedDict:
        img: np.ndarray = data['img']
        polygons: List = data['polygon']
        ignores: np.ndarray = data['ignore']
        h, w, _ = img.shape

        h, w = img.shape[:2]
        if data['train']:
            polygons, ignores = self._valid(polygons, ignores)
        probMap: np.ndarray = np.zeros((1, h, w), dtype=np.float64)
        probMask: np.ndarray = np.ones((h, w), dtype=np.float64)

        for i in range(len(polygons)):
            polygon: np.ndarray = np.array(polygons[i])
            ph = min(np.linalg.norm(polygon[0] - polygon[3]),
                     np.linalg.norm(polygon[1] - polygon[2]))
            pw = min(np.linalg.norm(polygon[0] - polygon[1]),
                     np.linalg.norm(polygon[3] - polygon[2]))

            if ignores[i] or min(pw, ph) < self._minTextSize:
                cv.fillPoly(probMask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignores[i] = True
            else:
                polygon_shape = Polygon(polygon)
                dist = polygon_shape.area * (1 - np.power(self._shrinkRatio, 2)) / polygon_shape.length
                subject: List = [tuple(point) for point in polygon]
                shrinking = pyclipper.PyclipperOffset()
                shrinking.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinkPolygon = shrinking.Execute(-dist)
                if len(shrinkPolygon) == 0:
                    cv.fillPoly(probMask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignores[i] = True
                    continue
                shrinkPolygon = np.array(shrinkPolygon[0]).reshape((-1, 2))
                cv.fillPoly(probMap[0], [shrinkPolygon.astype(np.int32)], 1)
        data.update(probMap=probMap, probMask=probMask)
        return data

    def _valid(self, polygons: List, ignores: np.ndarray) -> tuple:
        '''
            Input:
                - polygons: Danh sách các polygon trong ảnh
                - ignores : Danh sách quy định polygon bị ignore
            Ouput:
                - polygons: Danh sách các polygon được chuẩn hóa
                - ignores: Danh sách quy định polygon bị ignore
        '''
        if len(polygons) == 0:
            return polygons, ignores
        assert len(ignores) == len(polygons)
        for i in range(len(polygons)):
            area = self._polygonArea(polygons[i])
            if abs(area) < 1:
                ignores[i] = True
            if area > 0:
                polygons[i] = polygons[i][(0, 3, 2, 1), :]
        return polygons, ignores

    def _polygonArea(self, polygon: np.ndarray) -> float:
        '''
            Input:
                - polygon: Danh sách các đỉnh của polygon
            Ouput:
                - Diện tích của polygon chưa abs
        '''
        area: float = 0
        for i in range(polygon.shape[0]):
            i_n = (i + 1) % polygon.shape[0]
            area += (polygon[i_n, 0] * polygon[i, 1] - polygon[i_n, 1] * polygon[i, 0])
        return area / 2.
