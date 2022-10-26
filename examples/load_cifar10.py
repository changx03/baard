
from classifiers.cifar10_resnet18 import CIFAR10_ResNet18
import pytorch_lightning as pl
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.append(os.getcwd())


PATH_ROOT = Path(os.getcwd()).absolute()
print('PATH_ROOT:', PATH_ROOT)
PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'cifar10_resnet18.ckpt')
if os.path.isfile(PATH_CHECKPOINT):
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)
else:
    raise FileExistsError('Cannot find PyTorch lightning checkpoint. Check file name!')


model = CIFAR10_ResNet18.load_from_checkpoint(PATH_CHECKPOINT)

# trainer = pl.Trainer(accelerator='auto', logger=False)
# trainer.test(model, model.val_dataloader())


val_dataloader = model.val_dataloader()


def get_dataloader_shape(dataloader, has_y=True):
    n = len(dataloader.dataset)
    if has_y:
        x, y = next(iter(dataloader))
    else:
        x = next(iter(dataloader))
    data_shape = tuple([n] + list(x.size()[1:]))
    return data_shape


get_dataloader_shape(val_dataloader)


def dataloader2tensor(dataloader, has_y=True):
    shape = get_dataloader_shape(dataloader)
    x = torch.zeros(shape)
    if has_y:
        y = torch.zeros(shape[0], dtype=torch.int16)
    start = 0
    for _x, _y in dataloader:
        end = start + _x.size(0)
        x[start:end] = _x
        if has_y:
            y[start:end] = _y
        start = end
    if has_y:
        return x, y
    else:
        return x


x, y = dataloader2tensor(val_dataloader)
print('x', x.size())
print('y', y.size())


dataset = TensorDataset(x)
loader = DataLoader(dataset, batch_size=model.hparams.batch_size, shuffle=False)

trainer = pl.Trainer(accelerator='auto', logger=False)
predictions = torch.vstack(trainer.predict(model, loader))
print(predictions[:5])
