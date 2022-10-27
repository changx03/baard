from typing import Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader


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
