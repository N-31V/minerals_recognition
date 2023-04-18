from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters
from core.architecture.datasets.splitters import undersampling, train_test_split, get_dataset_mean_std


dataset_path = '/media/n31v/data/datasets/New_dataset_big'

dataset = ImageFolder(root=dataset_path, transform=ToTensor())
mean, std = get_dataset_mean_std(dataset)

transform = Compose([
    ToTensor(),
    Resize((200, 200), antialias=True),
    Normalize(mean=mean, std=std)
])

dataset = ImageFolder(root=dataset_path, transform=transform)

# dataset = undersampling(dataset)
train_ds, val_ds = train_test_split(dataset)

experimenter = ClassificationExperimenter(
    model=resnet18(num_classes=24),
    name='ResNet18'
)

dl_params = {'batch_size': 16, 'num_workers': 8}

fit_params = FitParameters(
    dataset_name='classification/minerals200_n_big',
    train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
    val_dl=DataLoader(val_ds, **dl_params),
    num_epochs=30,
    optimizer_params={'lr': 0.005},
    lr_scheduler=StepLR,
    lr_scheduler_params={'step_size': 10, 'gamma': 0.5, 'verbose': True},
    class_metrics=True
)

start_t = datetime.now()
experimenter.fit(p=fit_params)
print(f'Total time: {datetime.now() - start_t}')
