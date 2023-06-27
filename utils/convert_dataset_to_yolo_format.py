import os
import shutil
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

source_path = '/media/n31v/data/datasets/minerals'
ds_path = '/media/n31v/data/datasets/yolo_minerals'

config = {
    'train': '../yolo_minerals/images/train/',
    'val': '../yolo_minerals/images/valid/',
    'nc': 1,
    'names': ['mineral']
}


for address in os.listdir(source_path):
    print(address)
    for file in tqdm(os.listdir(os.path.join(source_path, address))):
        name, ext = os.path.splitext(file)

        if ext == '.jpg':
            phase = 'train' if random.random() < 0.8 else 'valid'
            image_path = Path(ds_path, 'images', phase, address)
            label_path = Path(ds_path, 'labels', phase, address)
            image_path.mkdir(parents=True, exist_ok=True)
            label_path.mkdir(parents=True, exist_ok=True)

            shutil.copy(
                src=os.path.join(source_path, address, file),
                dst=os.path.join(image_path, file)
            )
            labels = np.loadtxt(os.path.join(source_path, address, f'{name}.txt'), ndmin=2)
            labels[:, 0] = 0
            np.savetxt(os.path.join(label_path, f'{name}.txt'), labels, fmt='%.6f')

with open(os.path.join(ds_path, 'yolo_minerals.yaml'), 'w') as f:
    yaml.dump(config, f)
