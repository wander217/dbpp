import json
import os.path
from tqdm import tqdm
import cv2
import random

data_paths = [
    r"D:\python_project\label_tool\doanh_nghiep_0",
    r"D:\python_project\label_tool\doanh_nghiep_1",
    r"D:\python_project\label_tool\doanh_nghiep_3",
]

save_path = r'D:\db_pp\breg_detection'

total_data = []
for path in data_paths:
    with open(os.path.join(path, "target.json"), 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())
    for item in data:
        item['file_name'] = os.path.join(path, "image\\", item['file_name'])
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
    image = cv2.imread(item['file_name'])
    file_name = item['file_name'].split("\\")[-1]
    save_file = os.path.join(save_path, "train\\image\\", file_name)
    cv2.imwrite(save_file, image)
    item['file_name'] = file_name
with open(os.path.join(save_path, "train\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_data))

os.mkdir(os.path.join(save_path, "valid\\"))
os.mkdir(os.path.join(save_path, "valid\\image\\"))
for i in tqdm(range(len(valid_data))):
    item = valid_data[i]
    image = cv2.imread(item['file_name'])
    file_name = item['file_name'].split("\\")[-1]
    cv2.imwrite(os.path.join(save_path, "valid\\image\\", file_name), image)
    item['file_name'] = file_name
with open(os.path.join(save_path, "valid\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid_data))

os.mkdir(os.path.join(save_path, "test\\"))
os.mkdir(os.path.join(save_path, "test\\image\\"))
for i in tqdm(range(len(test_data))):
    item = test_data[i]
    image = cv2.imread(item['file_name'])
    file_name = item['file_name'].split("\\")[-1]
    cv2.imwrite(os.path.join(save_path, "test\\image\\", file_name), image)
    item['file_name'] = file_name
with open(os.path.join(save_path, "test\\target.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(test_data))
