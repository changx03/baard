"""Implementing the paper "Detecting adversarial examples by positive and negative representations" -- Luo et. al. (2022)
"""
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from baard.classifiers import DATASETS
from baard.classifiers.cifar10_resnet18 import CIFAR10_ResNet18
from baard.classifiers.mnist_cnn import MNIST_CNN
from baard.utils.torch_utils import dataloader2tensor, predict


def get_lightning_module(data_name: str) -> LightningModule:
    """Get PyTorch Lightning Module based on the dataset."""
    if data_name == DATASETS[0]:  # MNIST
        return MNIST_CNN
    elif data_name == DATASETS[1]:  # CIFAR10
        return CIFAR10_ResNet18
    else:
        raise NotImplementedError()


class PNDetector:
    """Implement Positive and Negative Representations Detector in PyTorch."""

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 path_model: str,
                 max_epochs: int = 50,
                 path_log: str = 'logs',
                 seed: int = None,
                 ):
        self.model = model
        self.data_name = data_name
        self.path_model = path_model
        self.max_epochs = max_epochs
        self.path_log = path_log
        self.seed = seed

        # Parameters from LightningModule:
        self.batch_size = self.model.train_dataloader().batch_size
        self.num_workers = self.model.train_dataloader().num_workers

        # Clone the base model
        self.pn_classifier: LightningModule = get_lightning_module(data_name).load_from_checkpoint(path_model)

    def train(self, X=None, y=None):
        """Train detector."""
        if X is None or y is None:
            dataloader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(dataloader_train)
            n_train_samples = len(dataloader_train.dataset)

            # NOTE: Keeping the same training size to avoid learning rate scheduler out of index error.
            X, _, y, _ = train_test_split(X, y, test_size=0.5, random_state=self.seed)
            X = X[:n_train_samples]
            y = y[:n_train_samples]

        self.check_range(X)

        X_neg = 1 - X
        y_neg = torch.clone(y)
        dataset = TensorDataset(
            torch.vstack([X, X_neg]),
            torch.hstack([y, y_neg]).long(),
        )
        print(f'Training set size: {len(dataset)}')
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True)  # Need shuffle for training.
        trainer = pl.Trainer(
            accelerator='auto',
            precision=16,
            max_epochs=self.max_epochs,
            num_sanity_val_steps=0,
            logger=TensorBoardLogger(save_dir=self.path_log, name=f'PNClassifier_{self.data_name}'),
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
            ],
            enable_model_summary=False,  # Issues with state_dice()
        )
        trainer.fit(self.pn_classifier, train_dataloaders=dataloader)

        # Check results
        dataset = TensorDataset(X_neg, y_neg)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)  # Need to check the predictions
        preds = predict(self.pn_classifier, dataloader)
        corrects = preds == y_neg
        acc_neg = corrects.float().mean()

        preds = predict(self.pn_classifier, self.model.val_dataloader())
        _, y = dataloader2tensor(self.model.val_dataloader())
        corrects = preds == y
        acc_pos = corrects.float().mean()

        print(f'Accuracy on X+: {acc_pos}, on X-: {acc_neg}.')

    def extract_features(self, X: Tensor) -> Tensor:
        """Extract Positive Negative similarity."""
        self.check_range(X)
        X_pos = X
        X_neg = 1 - X
        loader_pos = DataLoader(TensorDataset(X_pos),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        loader_neg = DataLoader(TensorDataset(X_neg),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)

        trainer = pl.Trainer(accelerator='auto', logger=False, enable_model_summary=False)

        # PyTorch Lightening trainer saves outputs as a list of mini-batches.
        outputs_pos = torch.vstack(trainer.predict(self.pn_classifier, loader_pos))
        # Use probability
        probs_pos = F.softmax(outputs_pos, dim=1)

        outputs_neg = torch.vstack(trainer.predict(self.pn_classifier, loader_neg))
        probs_neg = F.softmax(outputs_neg, dim=1)

        cos = torch.nn.CosineSimilarity(dim=1)
        similarity = cos(probs_pos, probs_neg)
        similarity = similarity.detach().numpy()
        return similarity

    def save(self, path_output: str) -> None:
        """Save PNClassifier as binary. The ideal extension is `.pnd`. """
        # torch.save(self.pn_classifier.state_dict(), path_output)
        raise NotImplementedError('Checkpoint is automatically saved under `path_log`.')

    def load(self, path_detector: str) -> None:
        """Load pre-trained PNClassifier. The default extension is `.pnd`."""
        # self.pn_classifier.load_state_dict(torch.load(path_detector))
        # self.pn_classifier.eval()
        self.pn_classifier = get_lightning_module(self.data_name).load_from_checkpoint(path_detector)

    @classmethod
    def check_range(cls, X):
        """Check the range of X. PN Classifier only works within [0, 1]."""
        min = X.min().item()
        max = X.max().item()
        assert np.isclose(min, 0) and np.isclose(max, 1), f'Expected range is [0, 1]. Got [{min}, {max}].'
