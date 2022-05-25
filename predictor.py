import json
import os
import time
from loss_model import LossModel
import torch
import yaml
from measure.metric import DetScore
from typing import Dict, List, Tuple, OrderedDict
import numpy as np
import cv2 as cv
import math


class DBPredictor:
    def __init__(self, config: str, pretrained):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            config: Dict = yaml.safe_load(f)
        self._model = LossModel(**config['lossModel'], device=self.device)
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        state_dict = torch.load(pretrained, map_location=self.device)
        self._model.load_state_dict(state_dict['model'])
        # multi scale problem => training
        self._score: DetScore = DetScore(**config['score'])
        self._limit: int = 960

    def _resize(self, image: np.ndarray) -> Tuple:
        org_h, org_w, _ = image.shape
        # scale = min([self._limit / org_h, self._limit / org_w])
        # new_h = int(scale * org_h)
        # new_w = int(scale * org_w)
        new_image = np.zeros((self._limit, self._limit, 3), dtype=np.uint8)
        # image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        new_image[:org_h, :org_w, :] = image
        return new_image, org_h, org_w

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        mean = [122.67891434, 116.66876762, 104.00698793]
        image = image.astype(np.float64)
        image = (image - mean) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image.unsqueeze(0)

    def __call__(self, image: np.ndarray) -> Tuple:
        self._model.eval()
        bboxes: List = []
        scores: List = []
        with torch.no_grad():
            h, w, _ = image.shape
            reImage, newH, newW = self._resize(image)
            inputImage = self._normalize(reImage)
            pred: OrderedDict = self._model(dict(img=inputImage), training=False)
            bs, ss = self._score(pred, dict(img=inputImage))

            for i in range(len(bs[0])):
                if ss[0][i] > 0:
                    bboxes.append(bs[0][i])
                    # bboxes.append(bs[0][i] * np.array([w / newW, h / newH]))
                    scores.append(ss[0][i])
            return bboxes, scores


if __name__ == "__main__":
    configPath: str = r'config/dbpp_se_eb0.yaml'
    pretrainedPath: str = r'D:\python_project\dbpp\logging_1\checkpoint_2236.pth'
    predictor = DBPredictor(configPath, pretrainedPath)
    root: str = r'D:\python_project\dbpp\breg_detection\valid\image'
    count = 0
    for subRoot, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                img = cv.imread(os.path.join(subRoot, file))
                boxes, scores = predictor(img)
                with open("pred_{}.json".format(file),'w', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "bbox": [[box.tolist() for box in boxes]],
                        "scores": [[score.tolist() for score in scores]]
                    }))
                with open(r"D:\python_project\dbpp\breg_detection\valid\target.json", encoding='utf-8') as f:
                    data = json.loads(f.readline())
                for item in data:
                    if item['file_name'] == file:
                        with open("gt_{}.json".format(file), 'w', encoding='utf-8') as f:
                            new_item = {
                                "polygon": [[]],
                                "ignore": [[]]
                            }
                            for bbox in item['target']:
                                new_item['polygon'][0].append(bbox['bbox'])
                                new_item['ignore'][0].append(False)
                            f.write(json.dumps(new_item))
                        break
                break
                # for box in boxes:
                #     img = cv.polylines(img, [box.astype(np.int32)], True, (0, 0, 255), 2)
                # cv.imwrite("result/test{}.jpg".format(count), img)
                # count += 1