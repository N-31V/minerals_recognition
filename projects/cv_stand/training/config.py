import torch
from torchvision.transforms import Compose, ToTensor, Resize
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from core.architecture.experiment.nn_experimenter import FasterRCNNExperimenter, FitParameters
from core.architecture.datasets.object_detection_datasets import YOLODataset


def binary_task(device):
    dataset = YOLODataset(
        image_folder='/media/n31v/data/datasets/minerals',
        transform=Compose([Resize((922, 1228)), ToTensor()]),
        replace_to_binary=True
    )
    n = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n, len(dataset) - n],
                                    generator=torch.Generator().manual_seed(31))
    dl_params = {
        'batch_size': 8,
        'num_workers': 4,
        'collate_fn': lambda x: tuple(zip(*x))
    }

    return {
        'exp': FasterRCNNExperimenter(
            num_classes=2,
            model_params={'weights': 'DEFAULT'},
            device=device
        ),
        'fit_params': FitParameters(
            dataset_name='detection/minerals(clean_binary)',
            train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
            val_dl=DataLoader(val_ds, **dl_params),
            num_epochs=30,
            optimizer_params={'lr': 0.0001},
            lr_scheduler=StepLR,
            lr_scheduler_params={'step_size': 10, 'gamma': 0.5},
        ),
        'ft_params': FitParameters(
            dataset_name='detection/minerals(clean_binary)',
            train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
            val_dl=DataLoader(val_ds, **dl_params),
            num_epochs=10,
            optimizer_params={'lr': 0.0001},
            lr_scheduler=StepLR,
            lr_scheduler_params={'step_size': 3, 'gamma': 0.5},
        )
    }


TASKS = {
    'binary': binary_task,
}


EXPS = {
    'base': {
        # 'base': {},
        # 'svd_c': {},
        # 'svd_s': {},
        # 'sfp_p': {'p': 0.2},
        'sfp_e': {'e': 0.994},
    }
}