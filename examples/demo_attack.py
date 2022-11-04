"""Demo code for genearting adversarial examples."""
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision as tv
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset

from baard.attacks.cw2 import carlini_wagner_l2
from baard.attacks.fast_gradient_method import fast_gradient_method
from baard.attacks.projected_gradient_descent import projected_gradient_descent
from baard.classifiers.cifar10_resnet18 import CIFAR10_ResNet18
from baard.classifiers.mnist_cnn import MNIST_CNN
from baard.utils.torch_utils import (dataloader2tensor, get_correct_examples,
                                     get_dataloader_shape)

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf')
print('PATH_ROOT:', PATH_ROOT)


def generate_attack():
    """Generating adversarial examples."""
    # model = CIFAR10_ResNet18.load_from_checkpoint(os.path.join(PATH_CHECKPOINT, 'cifar10_resnet18.ckpt'))
    model = MNIST_CNN.load_from_checkpoint(
        os.path.join(PATH_CHECKPOINT, 'mnist_cnn.ckpt'))

    trainer = pl.Trainer(accelerator='auto',
                         logger=False,
                         enable_model_summary=False,
                         enable_progress_bar=False)
    trainer.test(model, model.val_dataloader())

    train_dataloader = model.train_dataloader()
    x, y = dataloader2tensor(train_dataloader)

    print('x', x.size())
    print('y', y.size())

    print('min', x.min())
    print('max', x.max())

    val_dataloader = model.val_dataloader()

    # Remove examples with incorrect predictions
    dataloader = get_correct_examples(model, val_dataloader)
    x_shape = get_dataloader_shape(dataloader)
    print(x_shape)

    x, y = dataloader2tensor(dataloader)

    dataset = TensorDataset(x[:5])
    loader = DataLoader(dataset, batch_size=val_dataloader.batch_size,
                        num_workers=os.cpu_count(), shuffle=False)

    trainer = pl.Trainer(accelerator='auto',
                         logger=False,
                         enable_model_summary=False,
                         enable_progress_bar=False)
    predictions = torch.vstack(trainer.predict(model, loader))

    preds = torch.argmax(predictions, dim=1)
    print('preds', preds)

    print('label', y[:5])

    # Run C&W L2 attack
    # adv_x = carlini_wagner_l2(model, x[:5], 10)

    # Run FGSM attack
    adv_x = fast_gradient_method(
        model, x[:5], eps=64 / 255, norm=np.inf, clip_min=0, clip_max=1)

    # Run PGD/BIM attack
    # eps = 64 / 255
    # nb_iter = 100
    # eps_iter = eps / nb_iter
    # adv_x = projected_gradient_descent(model, x[:5], eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, clip_min=0, clip_max=1)

    preds_adv = torch.argmax(model(adv_x), dim=1)
    print('adv', preds_adv)


if __name__ == '__main__':
    generate_attack()
