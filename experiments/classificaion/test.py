import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from simple_conv_net import SimpleConvNet3
from core.architecture.experiment.nn_experimenter import ClassificationExperimenter

DATASETS_ROOT = '/media/n31v/data/datasets'
MODELS_PATH = '/media/n31v/data/models'

ds = ImageFolder(
    root=os.path.join(DATASETS_ROOT, 'minerals200'),
    transform=Compose([
        ToTensor(),
        # Normalize(mean=(0.444, 0.562, 0.556), std=(0.207, 0.235, 0.231))
    ])
)

# exp=ClassificationExperimenter(
#     model=resnet18(num_classes=24),
#     weights='/home/n31v/workspace/Fedot.Industrial/cases/object_recognition/cv_stand/models/classification/minerals(200х200)/ResNet18/train.sd.pt'
# )
#
# print(exp.val_loop(dataloader=DataLoader(ds)))

model = resnet18(num_classes=24)
model.to('cuda')
model.eval()
# model.load_state_dict(torch.load(os.path.join(MODELS_PATH, 'minerals(200x200)/ResNet18_3_0/train.sd.pt')))
model.load_state_dict(torch.load('/home/n31v/workspace/Fedot.Industrial/cases/object_recognition/cv_stand/models/classification/minerals(200х200)/ResNet18/train.sd.pt'))

n = 0
with torch.no_grad():
    for i in tqdm(range(len(ds))):
        img, target = ds.__getitem__(i)
        img = img.to('cuda')
        out = model(torch.unsqueeze(img, 0)).argmax(1)
        if out.item() == target:
            n += 1
    print(n, len(ds))


# ds = MNIST(
#     root=DATASETS_ROOT,
#     transform=Compose([ToTensor(), Normalize(mean=(0.13066,), std=(0.3081,))])
# )
# model = SimpleConvNet3(num_classes=10, in_channels=1)
# model.load_state_dict(torch.load(os.path.join(MODELS_PATH, 'MNIST/SimpleConvNet_0_0/train.sd.pt')))
# n = 0
# for i in tqdm(range(len(ds))):
#     img, target = ds.__getitem__(i)
#     out = model(torch.unsqueeze(img, 0)).argmax(1)
#     if out.item() == target:
#         n +=1
# print(n, len(ds))