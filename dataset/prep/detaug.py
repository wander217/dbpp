import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from typing import Dict, List, Tuple
import numpy as np
import math
import cv2 as cv


class DetAug:
    def __init__(self, **kwargs):
        # loading module inside imgaug
        moduls: List = []
        for key, item in kwargs.items():
            modul = getattr(iaa, key)
            if modul is not None:
                moduls.append(modul(**item))
        self._prep = None
        # creating preprocess sequent
        if len(moduls) != 0:
            self._prep = iaa.Sequential(moduls)
        self._newSize = kwargs['Resize']['size']

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
        tars = data['anno']
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
        onlyResize: bool = not data['train']

        if self._prep is not None:
            aug = self._prep.to_deterministic()
            if onlyResize:
                newImage, newH, newW = self._resize(image)
                newShape = (newH, newW, 3)
            else:
                newImage = aug.augment_image(image)
                newShape = newImage.shape
            data['img'] = newImage
            data['newShape'] = newShape[:2]
            self._makeAnnotation(aug, data, shape, newShape, onlyResize)
        # saving shape to recover
        data.update(orgShape=shape[:2])
        return data

    def _resize(self, image: np.ndarray) -> Tuple:
        '''
              Resize image when valid/test
        '''
        org_h, org_w, _ = image.shape
        scale = min([self._newSize['height'] / org_h,
                     self._newSize['width'] / org_w])
        new_h = int(scale * org_h)
        new_w = int(scale * org_w)
        new_image = np.zeros((self._newSize['height'], self._newSize['width'], 3), dtype=np.uint8)
        image = cv.resize(image, (new_w, new_h),  interpolation=cv.INTER_CUBIC)
        new_image[:new_h, :new_w, :] = image
        return new_image, new_h, new_w

    def _makeAnnotation(self,
                        aug,
                        data: Dict,
                        orgShape: Tuple,
                        newShape: Tuple,
                        onlyResize: bool) -> Dict:
        '''
           Changing bounding box coordinates
        '''
        if aug is None:
            return data

        tars: List = []
        orgH, orgW, _ = orgShape
        tarH, tarW, _ = newShape
        for tar in data['tar']:
            if onlyResize:
                newPolygon: List = [(point[0] / orgW * tarW, point[1] / orgH * tarH)
                                    for point in tar['bbox']]
            else:
                keyPoints: List = [Keypoint(point[0], point[1])
                                   for point in tar['bbox']]
                # clipping overflow bounding box
                keyPoints = aug.augment_keypoints([KeypointsOnImage(keyPoints,
                                                                    shape=orgShape)])[0].keypoints
                newPolygon: List = [(keyPoint.x, keyPoint.y)
                                    for keyPoint in keyPoints]
            tars.append({
                'label': tar['label'],
                'polygon': newPolygon,
                'ignore': tar['label'] == '###'
            })
        data['anno'] = tars
        return data
