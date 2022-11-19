"""Utility functions for PyTorch."""
import logging
import os
import warnings
from glob import glob
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision as tv
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.nn import Module
from torch.utils.data import (DataLoader, IterableDataset, SequentialSampler,
                              TensorDataset)

from baard.utils.miscellaneous import norm_parser

logger = logging.getLogger(__name__)


def get_num_items_per_example(dataloader: DataLoader) -> int:
    """Get the number of items per example."""
    batch = next(iter(dataloader))
    return len(batch)


def get_dataloader_shape(dataloader: DataLoader) -> Tuple:
    """Get the shape of X in a PyTorch Dataloader. Ignore the shape of labels y."""
    n = len(dataloader.dataset)
    batch = next(iter(dataloader))
    x = batch[0]
    data_shape = tuple([n] + list(x.size()[1:]))
    return data_shape


def dataloader2tensor(dataloader: DataLoader) -> Union[tuple[Tensor, Tensor], Tensor]:
    """Convert a PyTorch dataloader to PyTorch Tensor."""
    batch = next(iter(dataloader))
    has_y = len(batch) > 1
    shape = get_dataloader_shape(dataloader)
    x = torch.zeros(shape)
    if has_y:
        y = torch.zeros(shape[0], dtype=torch.int16)
    start = 0
    for batch in dataloader:
        _x = batch[0]
        end = start + _x.size(0)
        x[start:end] = _x
        if has_y:
            _y = batch[1]
            y[start:end] = _y
        start = end
    if has_y:
        return x, y
    else:
        return x


def dataset2tensor(dataset: TensorDataset) -> Union[tuple[Tensor, Tensor], Tensor]:
    """Convert a PyTorch Dataset back to tensor."""
    one_example = dataset[0]
    has_y = len(one_example) > 1
    n = len(dataset)
    data_shape = tuple([n] + list(one_example[0].size()))
    X = one_example[0].new_zeros(data_shape)
    y = torch.zeros(n, dtype=torch.long)
    for i, example in enumerate(dataset):
        X[i] = example[0]
        if has_y:
            y[i] = example[1]
    return (X, y) if has_y else X


def get_correct_examples(model: LightningModule,
                         dataloader: DataLoader,
                         return_loader=True
                         ) -> Union[DataLoader, TensorDataset]:
    """Removes incorrect predictions from the dataloader. If `return_loader` is set
    to False, returns a TensorDataset.
    """
    if get_num_items_per_example(dataloader) != 2:
        raise Exception('Labels are not in the dataloader!')

    if check_dataloader_shuffling(dataloader):
        warnings.warn('Dataloader should not have `shuffle=True`!')

    preds = predict(model, dataloader)
    x, y_true = dataloader2tensor(dataloader)
    corrects = preds == y_true
    indices = torch.squeeze(torch.nonzero(corrects))
    correct_dataset = TensorDataset(x[indices], y_true[indices])
    if return_loader:
        batch_size = dataloader.batch_size
        num_workers = os.cpu_count()
        return DataLoader(correct_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        return correct_dataset


def get_incorrect_examples(model: LightningModule,
                           dataloader: DataLoader,
                           return_loader=True
                           ) -> Union[DataLoader, TensorDataset]:
    """Get incorrect predictions from the dataloader. This function is used to return
    successful adversarial examples. If `return_loader` is set to False, returns a TensorDataset.
    """
    if get_num_items_per_example(dataloader) != 2:
        raise Exception('Labels are not in the dataloader!')

    if check_dataloader_shuffling(dataloader):
        warnings.warn('Dataloader should not have `shuffle=True`!')

    preds = predict(model, dataloader)
    x, y_true = dataloader2tensor(dataloader)
    corrects = preds != y_true
    indices = torch.squeeze(torch.nonzero(corrects))
    correct_dataset = TensorDataset(x[indices], y_true[indices])
    if return_loader:
        batch_size = dataloader.batch_size
        num_workers = dataloader.num_workers
        return DataLoader(correct_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        return correct_dataset


def check_dataloader_shuffling(dataloader: DataLoader) -> bool:
    """Check if a dataloader shuffles the data."""
    if (hasattr(dataloader, "sampler")
            and not isinstance(dataloader.sampler, SequentialSampler)
            and not isinstance(dataloader.dataset, IterableDataset)):
        return True
    else:
        return False


def create_noisy_examples(x: Tensor, n_samples: int = 30, noise_eps: str = 'u0.1',
                          clip_range: Tuple = (0, 1)) -> Tensor:
    """Create n noise examples aground x.

    :param Tensor x: Single input example.
    :param int n_samples: Number of random samples. Default is 256.
    :param str noise_eps: noise strength. `u[VALUE]`: adds uniform noise.
        `n[VALUE]`: adds normal distributed noise. `s[VALUE]`: adds noise with fixed value.
        Default is 'u0.1'.
    :param Tuple clip_range: (optional) Clipping range. Default is (0, 1).
    :return: A Tensor noisy X in shape of (n_samples, x.size()).
    """
    kind, eps = noise_eps[:1], float(noise_eps[1:])
    shape = (n_samples,) + tuple(x.size())
    if kind == 'u':
        noise = x.new_zeros(shape).uniform_(-1., 1.)
    elif kind == 'n':
        noise = x.new_zeros(shape).normal_(0., 1.)
    elif kind == 's':
        noise = torch.sign(x.new_zeros(shape).uniform_(-1., 1.))

    x_noisy = x.unsqueeze(0) + noise * eps

    if clip_range:
        x_noisy = torch.clamp(x_noisy, clip_range[0], clip_range[1])
    return x_noisy


def predict(model: LightningModule, dataloader: DataLoader, trainer: Trainer = None
            ) -> Tensor:
    """Predict the labels."""
    if trainer is None:
        trainer = pl.Trainer(accelerator='auto',
                             logger=False,
                             enable_model_summary=False,
                             enable_progress_bar=False)
    # PyTorch Lightening trainer saves outputs as a list of mini-batches.
    outputs = torch.vstack(trainer.predict(model, dataloader))
    preds = torch.argmax(outputs, dim=1)
    return preds


def batch_forward(model: Module, X: Tensor, batch_size: int = 256, device='cuda', num_workers=-1):
    """Forward propagation in mini-batch."""
    if num_workers <= 0 or num_workers > os.cpu_count():
        num_workers = os.cpu_count()

    if not torch.cuda.is_available():
        device = 'cpu'

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        # Probe output shape
        outputs = model(X[:1].to(device))
        outputs_shape = (len(X),) + tuple(outputs.size()[1:])

        outputs = X.new_zeros(outputs_shape)
        loader = DataLoader(
            TensorDataset(X), num_workers=num_workers, shuffle=False, batch_size=batch_size
        )
        start = 0
        for batch in loader:
            _x = batch[0].to(device)
            end = start + len(_x)
            outputs[start: end] = model(_x).cpu()
            start = end
    return outputs


def show_top5_imgs(dataset, figsize=(8, 3), cmap='gray'):
    """Show top 5 images in a PyTorch dataset."""
    images, labels = dataset2tensor(dataset)
    labels = [y for x, y in dataset]
    grid = tv.utils.make_grid(images[:5], nrow=5, padding=4, pad_value=1)
    grid = grid.permute(1, 2, 0)

    plt.figure(figsize=figsize)
    plt.imshow(grid.numpy(), cmap=cmap)
    plt.axis('off')
    plt.show()

    lbl_str = ', '.join([str(l.item()) for l in labels[:5]])
    print(f'Labels: {lbl_str}')


def show_img(x: Tensor, figsize=(3, 3), cmap='gray'):
    """Display a PyTorch Tensor image example."""
    x = x.permute(1, 2, 0)

    plt.figure(figsize=figsize)
    plt.imshow(x.detach().numpy(), cmap=cmap)
    plt.axis('off')
    plt.show()


def plot_images(path_img: str,
                lnorm: Union[str, int],
                eps_list: List,
                attack_name: str,
                n: int = 100,
                ):
    """Plot top-5 images along with their adversarial examples."""
    lnorm = norm_parser(lnorm)

    dataset_clean = torch.load(os.path.join(path_img, f'AdvClean-{n}.pt'))
    show_top5_imgs(dataset_clean, cmap=None)
    print('Clean images')

    for eps in eps_list:
        path_img_adv = os.path.join(path_img, f'{attack_name}-{lnorm}-{n}-{eps}.pt')
        dataset_adv = torch.load(path_img_adv)
        show_top5_imgs(dataset_adv, cmap=None)
        print(f'{attack_name} {lnorm} eps={eps}')


def find_last_checkpoint(model_name: str, data_name: str, kernel_name: str = None, path: str = 'logs') -> str:
    """Find the path of latest PyTorch Lightening checkpoint."""
    version_name = 'version_'
    i = 0
    kernel_name = f'_{kernel_name}' if kernel_name is not None else ''
    path_last_version = os.path.join(path, f'{model_name}_{data_name}{kernel_name}', version_name)
    while os.path.exists(path_last_version + str(i)):
        i += 1
    i -= 1  # Get the last valid one.
    path_last_version = path_last_version + str(i)
    path_last_version = os.path.join(path_last_version, 'checkpoints')
    if i < 0 or not os.path.exists(path_last_version):
        return None  # checkpoint is not found.
    logger.info('Found last checkpoint: %s', path_last_version)
    files = sorted(glob(os.path.join(path_last_version, '*.ckpt')))
    path_last_checkpoint = files[-1]
    return path_last_checkpoint


# def test_find_last_checkpoint():
#     """Test find_last_checkpoint"""
#     path_last_checkpoint = find_last_checkpoint('FeatureSqueezer', 'MNIST', kernel_name='depth', path='logs')
#     print(path_last_checkpoint)


# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     test_find_last_checkpoint()
