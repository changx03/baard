"""Global parameters for classifiers."""

from pytorch_lightning import LightningModule

from .cifar10_resnet18 import CIFAR10_ResNet18
from .mnist_cnn import MNIST_CNN

DATASETS = ['MNIST', 'CIFAR10']


def get_lightning_module(data_name: str) -> LightningModule:
    """Get PyTorch Lightning Module based on the dataset."""
    if data_name == DATASETS[0]:  # MNIST
        return MNIST_CNN
    elif data_name == DATASETS[1]:  # CIFAR10
        return CIFAR10_ResNet18
    else:
        raise NotImplementedError()
