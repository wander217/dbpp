import json
import os.path

import numpy as np
from tqdm import tqdm
import cv2
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
        item['image'] = cv2.imread(item['file_name'])
        new_target = []
        document_bbox = None
        for target in item['target']:
            if target['label'] == "64.document":
                document_bbox = np.array(target['bbox']).astype(np.int32)
        if document_bbox is None:
            continue
        x_min, y_min = np.min(document_bbox[:, 0]), np.min(document_bbox[:, 1])
        x_max, y_max = np.max(document_bbox[:, 0]), np.max(document_bbox[:, 1])
        item['image'] = item['image'][y_min:y_max + 1, x_min:x_max + 1]
        for target in item['target']:
            if target['label'] == "64.document":
                continue
            tmp = np.array(target['bbox'])
            points = cv2.boxPoints(cv2.minAreaRect(tmp))
            box = np.int16(points)
            new_target.append({
                "bbox": (box - np.array([x_min, y_min])).tolist(),
                "label": target['label'],
                "text": ""
            })
        item['target'] = new_target
    total_data.extend(data)
random.shuffle(total_data)

train_len = 680
valid_len = 88
train_data = total_data[:train_len]
valid_data = total_data[train_len: train_len + valid_len]
test_data = total_data[valid_len + train_len:]

os.mkdir(os.path.join(save_path, "train\\"))
os.mkdir(os.path.join(save_path, "train\\image\\"))
for i in tqdm(range(len(train_data))):
    item = train_data[i]
    image = item['image']
    file_name = item['file_name'].split("\\")[-1]
    save_file = os.path.join(save_path, "train\\image\\", file_name)
    cv2.imwrite(save_file, image)
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
    cv2.imwrite(os.path.join(save_path, "valid\\image\\", file_name), image)
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
    cv2.imwrite(os.path.join(save_path, "test\\image\\", file_name), image)
    item['file_name'] = file_name
    del item['image']
with open(os.path.join(save_path, "test\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(test_data))
