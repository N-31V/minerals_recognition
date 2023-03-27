import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

to_tensor = ToTensor()
dataset_path = '/media/n31v/data/datasets/New_dataset_big'
dataset = ImageFolder(root=dataset_path, transform=ToTensor())
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

model = resnet18(num_classes=24)
model.load_state_dict(torch.load('training/models/classification/minerals200/ResNet18/train.sd.pt'))
model.eval()

target_layers = [model.layer4[-1]]

img, target = dataset.__getitem__(8045)
input_tensor = img.unsqueeze(0).to('cuda')
img = img.permute((1, 2, 0)).numpy()

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor)
pred = model(input_tensor).detach().cpu().argmax(1)
print(f'target: {idx_to_class[target]}, ped: {idx_to_class[pred.item()]}')

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
plt.imshow(img)
plt.show()
plt.imshow(visualization)
plt.show()
