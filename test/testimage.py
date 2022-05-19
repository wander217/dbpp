import cv2 as cv
import os
import numpy as np

imageDir: str = r'C:\Users\thinhtq\Downloads\vietnamese_original\vietnamese\test_image'
labelDir: str = r'C:\Users\thinhtq\Downloads\vietnamese_original\vietnamese\labels'

imgPaths = os.listdir(imageDir)
for i in range(0, len(imgPaths)):
    print(i)
    imgPath = imgPaths[i]
    img = cv.imread(os.path.join(imageDir, imgPath))
    tarPath = 'gt_{}.txt'.format(int(imgPath.split(".")[0][2:]))
    with open(os.path.join(labelDir, tarPath), 'r', encoding='utf-8') as f:
        datas = f.readlines()
        for data in datas:
            tmp = data.split(",")
            l: int = len(tmp) - 1 if len(tmp) % 2 != 0 else len(tmp) - 2
            point = np.asarray(tmp[:l]).astype(np.int32).reshape((1, -1, 2))
            cv.polylines(img, point, True, (0, 255, 0), 2)
    # cv.imshow("img", img)
    # cv.waitKey(0)
    cv.imwrite(r"D:\db_pp\data/{}".format(imgPath), img)
    # break
