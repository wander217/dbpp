from pdf2image import convert_from_path
from tqdm import tqdm
import os

root: str = r'C:\Users\thinhtq\Downloads\hộ KD cá thể-20220317T025628Z-001\EKYC xác thực KH tổ chức-hộ KD cá thể\Merchant VNPAY'
savePath: str = r'D:\db_pp\dkkd'
if not os.path.isdir(savePath):
    os.mkdir(savePath)
pdfList = os.listdir(root)
for i in tqdm(range(len(pdfList))):
    pdfDir = pdfList[i]
    filePath = os.path.join(savePath, 'type{}'.format(i))
    if not os.path.isdir(filePath):
        os.mkdir(filePath)
    for j, file in enumerate(os.listdir(os.path.join(root, pdfDir))):
        pdfFile = convert_from_path(os.path.join(root, pdfDir, file),
                                    poppler_path=r'D:\db_pp\poppler-22.01.0\Library\bin')
        imgDir = os.path.join(filePath, 'doc{}'.format(j))
        if not os.path.isdir(imgDir):
            os.mkdir(imgDir)
        for k, item in enumerate(pdfFile):
            item.save(os.path.join(imgDir, 'img{}.jpg'.format(k)))
