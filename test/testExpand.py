import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

bbox = np.asarray([[694, 353],
                   [695, 354],
                   [694, 355]])
print(cv2.minAreaRect(bbox))
polygon = Polygon(bbox)
dist: float = polygon.area * 2 / polygon.length
expand = pyclipper.PyclipperOffset()
expand.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
expandBox: np.ndarray = np.array(expand.Execute(dist))
print(expandBox)
print(expandBox.shape)