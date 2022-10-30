import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew
from torch.nn import Module, Sequential

from baard.classifiers import DATASETS


def con(score: ArrayLike) -> ArrayLike:
    """Mean Absolute Deviation"""
    # score (n, d)
    score = score.reshape(len(score), -1)
    score_mean = np.mean(score, -1, keepdims=True)
    c_score = score - score_mean
    c_score = np.abs(c_score)
    return np.mean(c_score, axis=-1)


def mad(score: ArrayLike) -> ArrayLike:
    """Median Absolute Deviation"""
    pd = []
    for i in range(len(score)):
        d = score[i]
        median = np.median(d)
        abs_dev = np.abs(d - median)
        med_abs_dev = np.median(abs_dev)
        pd.append(med_abs_dev)
    pd = np.array(pd)
    return pd


def med_pdist(score: ArrayLike) -> ArrayLike:
    """Median Pairwise Distance"""
    pd = []
    for i in range(len(score)):
        d = score[i]
        k = np.median(pdist(d.reshape(-1, 1)))
        pd.append(k)
    pd = np.array(pd)
    return pd


def pd(score: ArrayLike) -> ArrayLike:
    """Mean Pairwise Distance"""
    pd = []
    for i in range(len(score)):
        d = score[i]
        k = np.mean(pdist(d.reshape(-1, 1)))
        pd.append(k)
    pd = np.array(pd)
    return pd


def neg_kurtosis(score: ArrayLike) -> ArrayLike:
    """Negative Kurtosis"""
    k = []
    for i in range(len(score)):
        di = score[i]
        ki = kurtosis(di, nan_policy='raise')
        k.append(ki)
    k = np.array(k)
    return -k


def quantile(score: ArrayLike) -> ArrayLike:
    """Between 25-75 Quantile"""
    # score (n, d)
    score = score.reshape(len(score), -1)
    score_75 = np.percentile(score, 75, -1)
    score_25 = np.percentile(score, 25, -1)
    score_qt = score_75 - score_25
    return score_qt


def calculate(score: ArrayLike, stat_name: str) -> ArrayLike:
    """Compute statistics metrics."""
    if stat_name == 'variance':
        results = np.var(score, axis=-1)
    elif stat_name == 'std':
        results = np.std(score, axis=-1)
    elif stat_name == 'pdist':
        results = pd(score)
    elif stat_name == 'con':
        results = con(score)
    elif stat_name == 'med_pdist':
        results = med_pdist(score)
    elif stat_name == 'kurtosis':
        results = neg_kurtosis(score)
    elif stat_name == 'skewness':
        results = -skew(score, axis=-1)
    elif stat_name == 'quantile':
        results = quantile(score)
    elif stat_name == 'mad':
        results = mad(score)
    print('results.shape', results.shape)
    return results


def get_latent_models(model: Module, data_name):
    """Get latent models based on the dataset, e.g., MNIST expects to use a CNN,
    and CIFAR10 expects to use ResNet18.
    """
    models = []
    if data_name == DATASETS[0]:  # MNIST CNN
        latent_net1 = Sequential(*list(model.children())[:3])  # 2nd conv layer without ReLU
        latent_net2 = Sequential(*list(model.children())[:6])  # Flattened layer after MaxPool
        latent_net3 = Sequential(*list(model.children())[:7])  # Last hidden layer before output (Without ReLU)
        models = [latent_net1, latent_net2, latent_net3, model]  # Model has no SoftMax
    elif data_name == DATASETS[1]:  # CIFAR10 ResNet18
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    return models


if __name__ == '__main__':
    import os
    from pathlib import Path

    from baard.classifiers.mnist_cnn import MNIST_CNN
    from baard.utils.torch_utils import batch_forward, dataset2tensor

    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_DATA = os.path.join(PATH_ROOT, 'data')
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    print(PATH_CHECKPOINT)

    model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)
    BATCH_SIZE = model.train_dataloader().batch_size
    DEVICE = model.device
    NUM_WORKERS = model.train_dataloader().num_workers
    input_size = (BATCH_SIZE, 1, 28, 28)

    models = get_latent_models(model, DATASETS[0])

    PATH_VAL_DATA = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'ValClean.n_1000.pt')
    val_dataset = torch.load(PATH_VAL_DATA)
    X_val, y_val = dataset2tensor(val_dataset)

    for m in models:
        latent_outputs = batch_forward(m, X_val, batch_size=BATCH_SIZE, device=DEVICE, num_workers=NUM_WORKERS)
        print(latent_outputs.size(), latent_outputs.device)
