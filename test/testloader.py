from dataset import DetLoader
from dataset.loader import DetDataset
from collections import OrderedDict
import yaml

if __name__ == "__main__":
    configPath = r'D:\python_project\dbpp\config\dbpp_se_eb3.yaml'

    with open(configPath, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    valid = DetLoader(**config['valid']).build()

    for batch in valid:
        print(batch['polygon'].size())


    # train = DetLoader(**config['train']).build()
    # for data in train:
    #     print(data.keys())
    #     break
