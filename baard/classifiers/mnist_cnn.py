"""Convolutional Neural Network for MNIST dataset."""
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

PATH_ROOT = Path(os.getcwd())
PATH_DEFAULT_LOGS = os.path.join(PATH_ROOT, 'logs')
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 16
BATCH_SIZE = 256 if torch.cuda.is_available() else 32
MAX_EPOCHS = 50 if torch.cuda.is_available() else 5


class MNIST_CNN(pl.LightningModule):
    """Convolutional Neural Network for MNIST dataset."""

    def __init__(self,
                 lr=0.05,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        super(MNIST_CNN, self).__init__()

        # automatically save params
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(9216, 200)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(200, 10)

        self.loss_fn = F.cross_entropy
        self.transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

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
        dataset_train = tv.datasets.MNIST('data', train=True,
                                          download=True, transform=self.transforms)
        return DataLoader(dataset_train, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        dataset_test = tv.datasets.MNIST('data', train=False,
                                         download=True, transform=self.transforms)
        return DataLoader(dataset_test, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)


if __name__ == '__main__':
    # Examples:
    # python ./baard/classifiers/mnist_cnn.py

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    if not args.max_epochs:
        args.max_epochs = MAX_EPOCHS

    if not args.default_root_dir:
        args.logger = TensorBoardLogger(
            save_dir=PATH_DEFAULT_LOGS, name='mnist_cnn')

    args.devices = NUM_WORKERS if args.accelerator == 'cpu' else 1

    print('PATH_ROOT:', PATH_ROOT)
    print('NUM_WORKERS:', NUM_WORKERS)
    print('BATCH_SIZE:', BATCH_SIZE)
    print('MAX_EPOCHS:', args.max_epochs)
    print('SEED:', args.seed)

    pl.seed_everything(args.seed)

    my_model = MNIST_CNN()
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='auto',
        precision=16,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
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
