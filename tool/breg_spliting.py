import json
import math
import os.path
import numpy as np
from tqdm import tqdm
import cv2 as cv
import random

data_paths = [
    r"D:\python_project\label_tool\doanh_nghiep_0",
    r"D:\python_project\label_tool\doanh_nghiep_1",
    r"D:\python_project\label_tool\doanh_nghiep_3",
]

save_path = r'D:\python_project\dbpp\breg_detection'

total_data = []
n, m = 0, 0
for path in data_paths:
    with open(os.path.join(path, "target.json"), 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
    for item in data:
        item['file_name'] = os.path.join(path, "image\\", item['file_name'])
        item['image'] = cv.imread(item['file_name'])
        new_target = []
        document_bbox = None
        for target in item['target']:
            if target['label'] == "64.document":
                document_bbox = np.array(target['bbox']).astype(np.int32)
        if document_bbox is None:
            continue
        x_min, x_max = np.min(document_bbox[:, 0]), np.max(document_bbox[:, 0])
        y_min, y_max = np.min(document_bbox[:, 1]), np.max(document_bbox[:, 1])
        mask = np.zeros(item['image'].shape, dtype=np.int32)
        mask = cv.fillPoly(mask, [np.array(document_bbox)], (1, 1, 1))
        item['image'] = (mask * item['image'])[y_min: y_max + 1, x_min:x_max + 1]
        for target in item['target']:
            if target['label'] == "64.document":
                continue
            tmp = np.array(target['bbox'])
            points = cv.boxPoints(cv.minAreaRect(tmp))
            box = np.int16(points).reshape((-1, 2))
            print(box.shape)
            new_target.append({
                "bbox": (box - np.array([x_min, y_min])).tolist(),
                "label": target['label'],
                "text": ""
            })
        item['target'] = new_target
    total_data.append(data)

train_data = []
valid_data = []
test_data = []
for i in range(len(total_data)):
    data = total_data[i]
    random.shuffle(data)
    train_len = math.ceil(0.8 * len(data) / 8) * 8
    valid_len = math.ceil(0.1 * len(data) / 8) * 8
    train_data.extend(data[:train_len])
    valid_data.extend(data[train_len: train_len + valid_len])
    test_data.extend(data[valid_len + train_len:])

os.mkdir(os.path.join(save_path, "train\\"))
os.mkdir(os.path.join(save_path, "train\\image\\"))
for i in tqdm(range(len(train_data))):
    item = train_data[i]
    image = item['image']
    file_name = item['file_name'].split("\\")[-1]
    save_file = os.path.join(save_path, "train\\image\\", file_name)
    cv.imwrite(save_file, image)
    item['file_name'] = file_name
    del item['image']
with open(os.path.join(save_path, "train\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_data))

os.mkdir(os.path.join(save_path, "valid\\"))
os.mkdir(os.path.join(save_path, "valid\\image\\"))
for i in tqdm(range(len(valid_data))):
    item = valid_data[i]
    image = item['image']
    file_name = item['file_name'].split("\\")[-1]
    cv.imwrite(os.path.join(save_path, "valid\\image\\", file_name), image)
    item['file_name'] = file_name
    del item['image']
with open(os.path.join(save_path, "valid\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid_data))

os.mkdir(os.path.join(save_path, "test\\"))
os.mkdir(os.path.join(save_path, "test\\image\\"))
for i in tqdm(range(len(test_data))):
    item = test_data[i]
    image = item['image']
    file_name = item['file_name'].split("\\")[-1]
    cv.imwrite(os.path.join(save_path, "test\\image\\", file_name), image)
    item['file_name'] = file_name
    del item['image']
with open(os.path.join(save_path, "test\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(test_data))
