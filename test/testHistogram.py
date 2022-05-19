import cv2 as cv
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import numpy as np

cluster: int = 20
imgPath: str = r'D:\db_pp\test_image\test1_1.png'
img: np.ndarray = cv.imread(imgPath)
tmp = img[:, :, 0].flatten()
print(len(tmp))
plt.hist(tmp, bins=20)
plt.show()
cv.imshow("abc",img)
cv.waitKey(0)
