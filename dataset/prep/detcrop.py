import numpy as np
import cv2 as cv
from typing import List, Dict, Tuple


class DetCrop:
    def __init__(self, generalSize: List, maxTries: int, minCropSize: float):
        self._generalSize: List = generalSize
        self._maxTries: int = maxTries
        self._minCropSize: float = minCropSize

    def __call__(self, data: Dict, isVisual: bool = False) -> Dict:
        output: Dict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: Dict):
        img = data['img']
        tars = data['target']
        for tar in tars:
            cv.polylines(img,
                         [np.int32(tar['polygon']).reshape((-1, 1, 2))],
                         True,
                         (255, 255, 0),
                         2)
        cv.imshow('aug_visual', img)

    def _build(self, data: Dict) -> Dict:
        img: np.ndarray = data['img']
        orgAnno: List = data['target']
        polygon_list: List = [tar['polygon'] for tar in orgAnno if not tar['ignore']]
        # cropX, cropY, cropW, cropH = self._cropArea(img, polygon_list)
        cropX, cropY, cropW, cropH = 0, 0, img.shape[1], img.shape[0]

        scaleW: float = self._generalSize[0] / cropW
        scaleH: float = self._generalSize[1] / cropH
        scale: float = min(scaleH, scaleW)
        h = int(scale * cropH)
        w = int(scale * cropW)

        padImage: np.ndarray = np.zeros((self._generalSize[1], self._generalSize[0], img.shape[2]), img.dtype)
        padImage[:h, :w] = cv.resize(img[cropY:cropY + cropH, cropX:cropX + cropW],
                                     (w, h),
                                     interpolation=cv.INTER_CUBIC)

        tars: List = []
        for target in orgAnno:
            polygon = np.array(target['polygon'])
            if not self._isOutside(polygon, [cropX, cropY, cropX + cropW, cropY + cropH]):
                newPolygon: List = ((polygon - (cropX, cropY)) * scale).tolist()
                tars.append({**target, 'polygon': newPolygon})
        data['target'] = tars
        data['img'] = padImage
        return data

    def _cropArea(self, img: np.ndarray, polygons: List) -> tuple:
        '''
            Hàm thực hiện cắt ảnh.
        '''
        h, w, _ = img.shape
        yAxis: np.ndarray = np.zeros(h, dtype=np.int32)
        xAxis: np.ndarray = np.zeros(w, dtype=np.int32)

        for polygon in polygons:
            tmp: np.ndarray = np.round(polygon, decimals=0).astype(np.int32)
            xAxis = self._maskDown(xAxis, tmp, 0)
            yAxis = self._maskDown(yAxis, tmp, 1)

        yNotMask: np.ndarray = np.where(yAxis == 0)[0]
        xNotMask: np.ndarray = np.where(xAxis == 0)[0]
        if len(xNotMask) == 0 or len(yNotMask) == 0:
            return 0, 0, w, h
        xSegment: List = self._splitRegion(xNotMask)
        ySegment: List = self._splitRegion(yNotMask)
        wMin: float = self._minCropSize * w
        hMin: float = self._minCropSize * h
        for _ in range(self._maxTries):
            xMin, xMax = self._choice(xSegment, xNotMask, w)
            yMin, yMax = self._choice(ySegment, yNotMask, h)
            newW = xMax - xMin + 1
            newH = yMax - yMin + 1
            if newW < wMin or newH < hMin:
                continue
            for polygon in polygons:
                if not self._isOutside(polygon, [xMin, yMin, xMax, yMax]):
                    return xMin, yMin, newW, newH
        return 0, 0, w, h

    def _choice(self, segment: List, axis: np.ndarray, limit: int):
        if len(segment) >= 2:
            tmp: np.ndarray = np.random.choice(len(segment), 2)
            tmp1 = np.random.choice(segment[tmp[0]], 1)[0]
            tmp2 = np.random.choice(segment[tmp[1]], 1)[0]
            return min([tmp1, tmp2]), max([tmp1, tmp2])
        tmp: np.ndarray = np.random.choice(axis, 2)
        tmp = np.clip(tmp, 0, limit - 1)
        return min(tmp), max(tmp)

    def _maskDown(self, axis: np.ndarray, polygon: np.ndarray, type: int) -> np.ndarray:
        '''
            masking axis by a byte
            type =1 : y axis, type=0 : x axis
        '''
        p_axis: np.ndarray = polygon[:, type]
        minValue: int = np.min(p_axis)
        maxValue: int = np.max(p_axis)
        axis[minValue:maxValue + 1] = 1
        return axis

    def _splitRegion(self, axis: np.ndarray) -> List:
        '''
            splitting by axis
        '''
        region: List = []
        startPoint: int = 0
        if axis.shape[0] == 0:
            return region
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region.append(axis[startPoint:i])
                startPoint = i
        if startPoint < axis.shape[0]:
            region.append(axis[startPoint:])
        return region

    def _isOutside(self, polygon: np.ndarray, lim: List) -> bool:
        '''

        :param polygon: polygon is surrounding text, size: 4x2
        :param lim: limit of crop size: [xMin, yMin, xMax, yMax]
        :return: true/false
        '''
        tmp: np.ndarray = np.array(polygon)
        x_min = tmp[:, 0].min()
        x_max = tmp[:, 0].max()
        y_min = tmp[:, 1].min()
        y_max = tmp[:, 1].max()
        if x_min >= lim[0] and x_max <= lim[2] and y_min >= lim[1] and y_max <= lim[3]:
            return False
        return True

    def __crop(self, regions: List, axisNotMask: np.ndarray, size: int) -> Tuple:
        '''
            Cắt ngẫu nhiên một đoạn từ trục
        '''
        if len(regions) > 1:
            id_list: List = list(np.random.choice(len(regions), size=2))
            value_list: List = []
            for id in id_list:
                region: int = regions[id]
                x: int = int(np.random.choice(region, 1))
                value_list.append(x)
            x_min = np.clip(min(value_list), 0, size - 1)
            x_max = np.clip(max(value_list), 0, size - 1)
        else:
            x_list: np.ndarray = np.random.choice(axisNotMask, size=2)
            x_min = np.clip(np.min(x_list), 0, size - 1)
            x_max = np.clip(np.max(x_list), 0, size - 1)
        return x_min, x_max
