from typing import Optional, Tuple, Callable
import torch
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, RandomPerspective, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from fedot_ind.core.architecture.experiment.nn_experimenter import ObjectDetectionExperimenter, FitParameters, ClassificationExperimenter
from fedot_ind.core.architecture.datasets.object_detection_datasets import YOLODataset
from fedot_ind.core.architecture.datasets.splitters import train_test_split


RESULT_PATH = 'results'


def get_detection_dataset(
        batch_size: int,
        resize: Optional[Tuple[int, int]] = None
) -> Callable:
    transforms = [ToTensor()]
    if resize is not None:
        transforms.append(Resize(resize))

    def get_dataset() -> Tuple:
        dataset = YOLODataset(
            image_folder='/media/n31v/data/datasets/minerals',
            transform=Compose(transforms),
            replace_to_binary=True
        )
        n = int(0.8 * len(dataset))
        train_ds, val_ds = random_split(
            dataset, [n, len(dataset) - n], generator=torch.Generator().manual_seed(31)
        )
        dl_params = {
            'batch_size': batch_size,
            'num_workers': 8,
            'collate_fn': lambda x: tuple(zip(*x))
        }
        return train_ds, val_ds, dl_params

    return get_dataset


def get_classification_dataset(
        batch_size: int,
) -> Callable:
    def get_dataset() -> Tuple:
        dataset = ImageFolder(
            root='/media/n31v/data/datasets/minerals_200',
            transform=Compose([
                ToTensor(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ColorJitter(0.0005, 0.0005, 0.0005, 0.0),
                RandomPerspective(0.6, 1.0),
                Resize((200, 200), antialias=True)
            ]))
        train_ds, val_ds = train_test_split(dataset)
        dl_params = {'batch_size': 16, 'num_workers': 8}
        return train_ds, val_ds, dl_params

    return get_dataset


def fasterrcnn_task(get_dataset, ds_name):
    train_ds, val_ds, dl_params = get_dataset()

    def f_task(device):
        return {
            'exp': ObjectDetectionExperimenter(
                model=fasterrcnn_resnet50_fpn(num_classes=2),
                device=device
            ),
            'fit_params': FitParameters(
                dataset_name=f'detection/{ds_name}',
                train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
                val_dl=DataLoader(val_ds, **dl_params),
                num_epochs=50,
                optimizer_params={'lr': 0.005},
                lr_scheduler=StepLR,
                lr_scheduler_params={'step_size': 10, 'gamma': 0.5},
                models_path=RESULT_PATH,
                summary_path=RESULT_PATH,
            ),
            'ft_params': FitParameters(
                dataset_name=f'detection/{ds_name}',
                train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
                val_dl=DataLoader(val_ds, **dl_params),
                num_epochs=10,
                optimizer_params={'lr': 0.0001},
                lr_scheduler=StepLR,
                lr_scheduler_params={'step_size': 3, 'gamma': 0.5},
                models_path=RESULT_PATH,
                summary_path=RESULT_PATH,
            )
        }

    return f_task


def ssd_task(get_dataset, ds_name):
    train_ds, val_ds, dl_params = get_dataset()

    def s_task(device):
        return {
            'exp': ObjectDetectionExperimenter(
                model=ssdlite320_mobilenet_v3_large(num_classes=2),
                device=device
            ),
            'fit_params': FitParameters(
                dataset_name=f'detection/{ds_name}',
                train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
                val_dl=DataLoader(val_ds, **dl_params),
                num_epochs=300,
                lr_scheduler=StepLR,
                lr_scheduler_params={'step_size': 10, 'gamma': 0.5},
                models_path=RESULT_PATH,
                summary_path=RESULT_PATH,
            ),
            'ft_params': FitParameters(
                dataset_name=f'detection/{ds_name}',
                train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
                val_dl=DataLoader(val_ds, **dl_params),
                num_epochs=10,
                optimizer_params={'lr': 0.005},
                lr_scheduler=StepLR,
                lr_scheduler_params={'step_size': 3, 'gamma': 0.5},
                models_path=RESULT_PATH,
                summary_path=RESULT_PATH,
            )
        }

    return s_task


def resnet_task(get_dataset, ds_name):
    train_ds, val_ds, dl_params = get_dataset()

    def r_task(device):
        return {
            'exp': ClassificationExperimenter(
                model=resnet18(num_classes=24),
                name='ResNet18',
                device=device
            ),
            'fit_params': FitParameters(
                dataset_name=f'classification/{ds_name}',
                train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
                val_dl=DataLoader(val_ds, **dl_params),
                num_epochs=30,
                optimizer_params={'lr': 0.005},
                lr_scheduler=StepLR,
                lr_scheduler_params={'step_size': 10, 'gamma': 0.5},
                class_metrics=True,
                models_path=RESULT_PATH,
                summary_path=RESULT_PATH,
            ),
            'ft_params': FitParameters(
                dataset_name=f'classification/{ds_name}',
                train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
                val_dl=DataLoader(val_ds, **dl_params),
                num_epochs=10,
                optimizer_params={'lr': 0.005},
                lr_scheduler=StepLR,
                lr_scheduler_params={'step_size': 3, 'gamma': 0.5},
                models_path=RESULT_PATH,
                summary_path=RESULT_PATH,
            )
        }

    return r_task


TASKS = {
    'ssd': ssd_task(get_detection_dataset(batch_size=64), ds_name='binary_small'),
    'rcnn': fasterrcnn_task(get_detection_dataset(batch_size=12, resize=(922, 1228)), ds_name='binary_big'),
    'resnet': resnet_task(get_classification_dataset(batch_size=32), ds_name='minerals_200')
}


EXPS = {
    'det': [
        ('base', {}),
        # ('svd_c', {}),
        # ('svd_s', {}),
        # ('sfp_p', {'p': 0.2}),
        # ('sfp_e', {'e': 0.994})
    ],
    'clf': [
        ('base', {}),
        ('svd_c', {}),
        ('svd_s', {}),
        ('sfp_p', {}),
        ('sfp_p', {'p': 0.2}),
        ('sfp_e', {}),
        ('sfp_e', {'e': 0.994})
    ]
}

