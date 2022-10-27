from typing import Tuple, Union
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, IterableDataset


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


def get_correct_examples(model: LightningModule,
                         dataloader: DataLoader,
                         return_loader=True
                         ) -> Union[DataLoader, TensorDataset]:
    """Removes incorrect predictions from the dataloader. If `return_loader` is set
    to False, returns a TensorDataset.
    """
    if check_dataloader_shuffling(dataloader):
        warnings.warn('Dataloader should not have `shuffle=True`!')

    trainer = pl.Trainer(accelerator='auto', logger=False)
    outputs = torch.vstack(trainer.predict(model, dataloader))
    preds = torch.argmax(outputs, dim=1)

    x, y_true = dataloader2tensor(dataloader)
    corrects = preds == y_true
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
