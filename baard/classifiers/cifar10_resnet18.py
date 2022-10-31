"""ResNet-18 model for CIFAR10 dataset."""
import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_DEFAULT_LOGS = os.path.join(PATH_ROOT, 'logs')
NUM_WORKERS = os.cpu_count() - 2
BATCH_SIZE = 256 if torch.cuda.is_available() else 32
MAX_EPOCHS = 50 if torch.cuda.is_available() else 5


def create_model():
    """Return a ResNet-18 model for CIFAR10. Input shape: (3, 32, 32)."""
    model = tv.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class CIFAR10_ResNet18(pl.LightningModule):
    """ResNet-18 model for CIFAR10 dataset."""

    def __init__(self,
                 lr=0.05,
                 batch_size=BATCH_SIZE,
                 path_data=PATH_DATA,
                 num_workers=NUM_WORKERS):
        super(CIFAR10_ResNet18, self).__init__()

        # automatically save params
        self.save_hyperparameters()

        self.model = create_model()
        self.loss_fn = F.cross_entropy

        # WARNING: Do NOT normalized the data! The default range is [0, 1]
        self.train_transforms = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            # cifar10_normalization(),
        ])
        self.test_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            # cifar10_normalization(),
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)

        preds = outputs.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y).float().mean()

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch[0])
        probs = F.softmax(outputs, dim=1)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.train_dataloader()),
            ),
            'interval': 'step',
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_dict
        }

    def train_dataloader(self):
        dataset_train = tv.datasets.CIFAR10(self.hparams.path_data, train=True,
                                            download=True, transform=self.train_transforms)
        loader_train = DataLoader(dataset_train, batch_size=self.hparams.batch_size,
                                  shuffle=True, num_workers=self.hparams.num_workers)
        return loader_train

    def val_dataloader(self):
        dataset_test = tv.datasets.CIFAR10(self.hparams.path_data, train=False,
                                           download=True, transform=self.test_transforms)
        loader_test = DataLoader(dataset_test, batch_size=self.hparams.batch_size,
                                 shuffle=False, num_workers=self.hparams.num_workers)
        return loader_test


if __name__ == '__main__':
    # Examples:
    # python ./classifiers/cifar10_resnet18.py

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    if not args.max_epochs:
        args.max_epochs = MAX_EPOCHS

    if not args.default_root_dir:
        args.logger = TensorBoardLogger(save_dir=PATH_DEFAULT_LOGS, name='cifar10_resnet18')

    args.devices = NUM_WORKERS if args.accelerator == 'cpu' else 1

    print('PATH_ROOT:', PATH_ROOT)
    print('NUM_WORKERS:', NUM_WORKERS)
    print('BATCH_SIZE:', BATCH_SIZE)
    print('MAX_EPOCHS:', args.max_epochs)
    print('SEED:', args.seed)

    pl.seed_everything(args.seed)

    my_model = CIFAR10_ResNet18()
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='auto',
        precision=16,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # EarlyStopping(monitor='val_loss', mode='min', patience=5)
        ],
    )

    ############################################################################
    # For development only:

    # trainer = pl.Trainer(fast_dev_run=True)  # For quick testing

    # To train with only 10% of data
    # trainer = pl.Trainer(
    #     accelerator='auto',
    #     limit_train_batches=0.1,
    #     max_epochs=5,
    #     default_root_dir=PATH_DEFAULT_LOGS, # Save to `./logs/lightning_logs`
    #     callbacks=[
    #         LearningRateMonitor(logging_interval='step'),
    #         EarlyStopping(monitor='train_loss', mode='min', patience=5)
    #     ],
    # )
    ############################################################################

    trainer.fit(my_model)

    print('On train set:')
    trainer.test(my_model, my_model.train_dataloader())

    print('On test set:')
    trainer.test(my_model, my_model.val_dataloader())
