import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from typing import Dict, List, Tuple
import numpy as np
import math
import cv2 as cv


class DetAug:
    def __init__(self, onlyResize: bool, **kwargs):
        # loading module inside imgaug
        moduls: List = []
        for key, item in kwargs.items():
            module = getattr(iaa, key)
            if module is not None:
                moduls.append(module(**item))
        self._prep = None
        self._onlyResize: bool = onlyResize
        # creating preprocess sequent
        if len(moduls) != 0:
            self._prep = iaa.Sequential(moduls)

    def __call__(self, data: Dict, isVisual: bool = False):
        '''
        preprocessing input data with imgaug
        :param data: a dict contains: img, train, tar
        :return: data processing with imgaug: img, train, tar, anno, shape
        '''
        output = self._build(data)
        if isVisual:
            self._visual(data)
        return output

    def _visual(self, data: Dict, lineHeight: int = 2):
        img = data['img']
        tars = data['target']
        for tar in tars:
            cv.polylines(img,
                         [np.int32(tar['polygon']).reshape((1, -1, 2))],
                         True,
                         (255, 255, 0),
                         lineHeight)
        cv.imshow('aug_visual', img)

    def _build(self, data: Dict) -> Dict:
        image: np.ndarray = data['img']
        shape: Tuple = image.shape

        if self._prep is not None:
            aug = self._prep.to_deterministic()
            if not self._onlyResize:
                data['img'] = aug.augment_image(image)
                self._makeAnnotation(aug, data, shape)
            else:
                data['img'] = self._resize(image)
        # saving shape to recover
        data.update(orgShape=shape[:2])
        return data

    def _resize(self, image: np.ndarray) -> np.ndarray:
        '''
              Resize image when valid/test
        '''
        org_h, org_w, _ = image.shape
        new_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        new_image[:org_h, :org_w, :] = image
        return new_image

    def _makeAnnotation(self, aug, data: Dict, shape: Tuple) -> Dict:
        '''
           Changing bounding box coordinates
        '''
        if aug is None:
            return data

        tars: List = []
        for tar in data['target']:
            if self._onlyResize:
                newPolygon: List = [(point[0], point[1])
                                    for point in tar['bbox']]
            else:
                keyPoints: List = [Keypoint(point[0], point[1])
                                   for point in tar['bbox']]
                keyPoints = aug.augment_keypoints([
                    KeypointsOnImage(keyPoints, shape=shape)
                ])[0].keypoints
                newPolygon: List = [(keyPoint.x, keyPoint.y)
                                    for keyPoint in keyPoints]
            tars.append({
                'label': tar['text'],
                'polygon': newPolygon,
                'ignore': tar['text'] == '###'
            })
        data['target'] = tars
        return data
