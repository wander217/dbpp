import numpy as np
import yaml
import cv2 as cv

from dataset.prep.detaug import DetAug
from dataset.prep.detcrop import DetCrop
from dataset.prep.detform import DetForm
from dataset.prep.probmaker import ProbMaker
from dataset.prep.threshmaker import ThreshMaker
from dataset.prep.detnorm import DetNorm
from dataset.prep.detfilter import DetFilter

with open(r'D:\db_pp\config\dbpp_eb3.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)
tars = []

# annoFile = r'D:\db_pp\test\data\gt_11.txt'
# imgFile = r'D:\db_pp\test\data\im0011.jpg'

# annoFile = r'D:\db_pp\test\data\gt_1223.txt'
# imgFile = r'D:\db_pp\test\data\im1223.jpg'

annoFile = r'D:\db_pp\test\data\gt_1328.txt'
imgFile = r'D:\db_pp\test\data\im1328.jpg'

with open(annoFile, 'r', encoding='utf=8') as f:
    annos = f.readlines()
    for anno in annos:
        tmp = anno.split(",")
        tars.append({
            'polygon': np.asarray(tmp[:8], dtype=np.int32).reshape(-1, 2),
            'label': tmp[8].strip("\n").strip("\t")
        })

data = {
    'img': cv.imread(imgFile),
    'tar': tars,
    'train': False
}

aug = DetAug(**config['train']['dataset']['prep']['DetAug'])
crop = DetCrop(**config['train']['dataset']['prep']['DetCrop'])
form = DetForm(**config['train']['dataset']['prep']['DetForm'])
probMaker = ProbMaker(**config['train']['dataset']['prep']['ProbMaker'])
threshMaker = ThreshMaker(**config['train']['dataset']['prep']['ThreshMaker'])
detNorm = DetNorm(**config['train']['dataset']['prep']['DetNorm'])
detFilter = DetFilter(**config['train']['dataset']['prep']['DetFilter'])
data = aug(data, isVisual=False)
data = crop(data, isVisual=False)
data = form(data, isVisual=False)
data = probMaker(data, isVisual=False)
data = threshMaker(data, isVisual=False)
data = detNorm(data, isVisual=False)
data = detFilter(data, isVisual=True)

