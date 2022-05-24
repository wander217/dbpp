from dataset import DetLoader
from dataset.loader import DetDataset
from collections import OrderedDict
import yaml

if __name__ == "__main__":
    configPath = r'D:\python_project\dbpp\config\dbpp_se_eb1.yaml'

    with open(configPath, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    valid = DetDataset(**config['train']['dataset'])
    print(valid.__len__())

    for i in range(0, valid.__len__()):
        data: OrderedDict = valid.__getitem__(i, isVisual=True)
        print(i)

    # train = DetLoader(**config['train']).build()
    # for data in train:
    #     print(data.keys())
    #     break
