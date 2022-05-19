import os
import json
import cv2 as cv
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


class TargetHandler:
    def __init__(self):
        pass

    def targetName(self, imgName: str):
        return imgName

    def annotation(self, raw: str) -> Optional[Dict]:
        splitedRaw: List = raw.split(",")
        rawLen: int = len(splitedRaw)
        splitPoint: int = rawLen - 1 if rawLen % 2 != 0 else rawLen - 2
        label: str = ''.join(splitedRaw[splitPoint:]).strip().strip('\n').strip('\r\t')
        if len(label) == 0:
            return None
        polygon: np.ndarray = np.asarray(splitedRaw[:splitPoint], dtype=np.int32).reshape(-1, 2)
        return dict(label=label, polygon=polygon.tolist())


class VinTextTargetHandler(TargetHandler):
    def __init__(self):
        super().__init__()

    def targetName(self, imgName: str):
        id = int(imgName.split('.')[0][2:])
        return 'gt_{}.txt'.format(id)


class ItemHandler:
    def __init__(self,
                 orgImgDir: str,
                 orgLabelDir: str,
                 tarDir: str,
                 handler: TargetHandler,
                 imgType: int = 0):
        self._orgImgDir: str = orgImgDir
        self._orgLabelDir: str = orgLabelDir
        self._tarDir: str = tarDir
        self._imgDir: str = os.path.join(self._tarDir, "image/")
        if not os.path.isdir(self._imgDir):
            os.mkdir(self._imgDir)
        self._handler: TargetHandler = handler
        self._tars: List = []
        self._imgType: int = imgType

    def saveImage(self, orgName: str, tarName: str):
        image = cv.imread(os.path.join(self._orgImgDir, orgName))
        if self._imgType == 1:
            imgName: str = "{}.npy".format(tarName)
            imgPath: str = os.path.join(self._imgDir, imgName)
            np.save(imgPath, image)
            return imgName
        imgName: str = "{}.jpg".format(tarName)
        imgPath: str = os.path.join(self._imgDir, imgName)
        cv.imwrite(imgPath, image)
        return imgName

    def addTarget(self, orgName: str, tarName: str):
        annos: List = []
        tarFile: str = os.path.join(self._orgLabelDir, self._handler.targetName(orgName))
        with open(tarFile, 'r', encoding='utf-8') as f:
            raws = f.readlines()
            for raw in raws:
                tmp: Optional[Dict] = self._handler.annotation(raw)
                if tmp is not None:
                    annos.append(tmp)
        self._tars.append(dict(img=tarName, target=annos))

    def saveTarget(self):
        targetFile: str = os.path.join(self._tarDir, 'target.json')
        with open(targetFile, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self._tars))


class PartMaker:
    def __init__(self,
                 name: str,
                 orgImgDir: str,
                 orgTarDir: str,
                 root: str,
                 handler,
                 imgType: int = 0):
        self._name: str = name
        self._orgImgs: List = os.listdir(orgImgDir)
        tarDir = os.path.join(root, name)
        if not os.path.isdir(tarDir):
            os.mkdir(tarDir)
        self._handler: ItemHandler = ItemHandler(orgImgDir,
                                                 orgTarDir,
                                                 tarDir,
                                                 handler,
                                                 imgType)

    def __call__(self):
        print("-" * 33)
        print("Starting covert {} data...".format(self._name))
        for i in tqdm(range(len(self._orgImgs))):
            if ".jpg" in self._orgImgs[i] or ".png" in self._orgImgs[i]:
                newName: str = self._handler.saveImage(self._orgImgs[i], "img{}".format(i))
                self._handler.addTarget(self._orgImgs[i], newName)
        self._handler.saveTarget()
        print("Complete!")
        print("-" * 33)


class DatasetMaker:
    def __init__(self,
                 root: str,
                 train: Dict,
                 valid: Dict,
                 test: Dict,
                 handler: TargetHandler,
                 imgType: int = 0):
        if not os.path.isdir(root):
            os.mkdir(root)
        self._train: PartMaker = PartMaker(**train,
                                           root=root,
                                           handler=handler,
                                           imgType=imgType)
        self._valid: PartMaker = PartMaker(**valid,
                                           root=root,
                                           handler=handler,
                                           imgType=imgType)
        self._test: PartMaker = PartMaker(**test,
                                          root=root,
                                          handler=handler,
                                          imgType=imgType)

    def __call__(self):
        self._train()
        self._valid()
        self._test()


if __name__ == '__main__':
    ROOT = r'C:\Users\thinhtq\Downloads\vietnamese_original\vietnamese'
    TRAIN = dict(
        name="train",
        orgImgDir=os.path.join(ROOT, "train_images"),
        orgTarDir=os.path.join(ROOT, "labels")
    )
    VALID = dict(
        name="valid",
        orgImgDir=os.path.join(ROOT, "test_image"),
        orgTarDir=os.path.join(ROOT, "labels")
    )
    TEST = dict(
        name="test",
        orgImgDir=os.path.join(ROOT, "unseen_test_images"),
        orgTarDir=os.path.join(ROOT, "labels")
    )
    handler: VinTextTargetHandler = VinTextTargetHandler()
    datamaker = DatasetMaker(r"D:\dataset\vintext",
                             TRAIN, TEST, VALID, handler, 1)
    datamaker()
