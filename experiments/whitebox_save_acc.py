"""Read adversarial examples and record the classifier's accuracy."""
import os
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from baard.classifiers import get_lightning_module
from baard.utils.torch_utils import get_score

SEED = 727328
PATH_ROOT = os.getcwd()
PATH_RESULTS = os.path.join(PATH_ROOT, 'results', f'exp{SEED}')
DATA_NAMES = ['MNIST', 'CIFAR10']
NORMS = ['inf', 2]


def get_model(data_name: str):
    """Get pre-trained model."""
    if data_name == 'MNIST':
        path_checkpoint = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    elif data_name == 'CIFAR10':
        path_checkpoint = os.path.join(PATH_ROOT, 'pretrained_clf', 'cifar10_resnet18.ckpt')
    else:
        raise NotImplementedError
    model = get_lightning_module(data_name).load_from_checkpoint(path_checkpoint)
    return model


def compute_list_acc(model, data_name, norm, attack_name='whitebox'):
    """Return accuracy and the corresponding dataset."""
    path_name = os.path.join(
        PATH_RESULTS, data_name, f'{attack_name}-L{norm}-1000-*.pt'
    )
    dataset_list = glob(path_name)

    acc = []
    for d in dataset_list:
        print(d)
        dataset = torch.load(d)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        _acc = get_score(model, dataloader)
        acc.append(_acc)
    return np.array(acc), dataset_list


def save_acc():
    """Save accuracy."""
    path_output = os.path.join(PATH_RESULTS, 'acc')
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    for dname in DATA_NAMES:
        model = get_model(dname)
        for n in NORMS:
            acc, dataset_list = compute_list_acc(model, dname, n, attack_name='whitebox')
            eps = [float(Path(d).stem.split('-')[-1]) for d in dataset_list]
            df = pd.DataFrame({'eps': eps, 'acc': acc})
            df = df.sort_values('eps')
            df.to_csv(os.path.join(path_output, f'{dname}-whitebox-L{n}-acc.csv'), index=False)


if __name__ == '__main__':
    save_acc()
