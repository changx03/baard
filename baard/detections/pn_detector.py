"""Implementing the paper "Detecting adversarial examples by positive and
negative representations" -- Luo et. al. (2022)
"""
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from baard.classifiers import get_lightning_module
from baard.detections.baard_detector import Detector
from baard.utils.torch_utils import dataloader2tensor, predict

logger = logging.getLogger(__name__)


class PNDetector(Detector):
    """Implement Positive and Negative Representations Detector in PyTorch."""

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 path_model: str,
                 dist: str = 'cosine',
                 max_epochs: int = 30,
                 path_checkpoint: str = 'logs',
                 seed: int = None,
                 ):
        super().__init__(model, data_name)

        self.path_model = path_model
        self.dist = dist
        self.max_epochs = max_epochs
        self.path_checkpoint = path_checkpoint
        self.seed = seed

        # Clone the base model
        self.pn_classifier: LightningModule = get_lightning_module(data_name).load_from_checkpoint(path_model)

        if dist == 'cosine':  # Cosine Similarity is much better in sparse space
            # NOTE: The implementation in author's repo uses Negative Cosine Similarity.
            # Link: https://github.com/Daftstone/PNDetector/blob/f572946a7738060d9ca956a87cd0cdcc5a3007f9/util_tool/utils_tf.py#L207
            # self.dist_fn = torch.nn.CosineSimilarity(dim=1)
            self.dist_fn = self.neg_cosine_similarity
        elif dist == 'pair':
            self.dist_fn = torch.nn.PairwiseDistance(p=2)
        else:
            raise NotImplementedError()

        # Register params
        self.params['path_model'] = self.path_model
        self.params['dist'] = self.dist
        self.params['max_epochs'] = self.max_epochs
        self.params['path_checkpoint'] = self.path_checkpoint
        self.params['seed'] = self.seed

    def train(self, X: Tensor = None, y: Tensor = None):
        """Train detector. X and y are dummy variables."""
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
        logger.info('Training set size: %d', len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True)  # Need shuffle for training.
        trainer = pl.Trainer(
            accelerator='auto',
            precision=16,
            max_epochs=self.max_epochs,
            num_sanity_val_steps=0,
            logger=TensorBoardLogger(save_dir=self.path_checkpoint, name=f'PNClassifier_{self.data_name}'),
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
            ],
            enable_model_summary=False,  # Issues with state_dice()
        )
        trainer.fit(self.pn_classifier, train_dataloaders=dataloader)

        # Check results
        loader_val = self.model.val_dataloader()
        X_val, y_val = dataloader2tensor(loader_val)
        X_val_neg = 1 - X_val
        dataloader_val_neg = DataLoader(TensorDataset(X_val_neg),
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=False)  # Need to check the predictions
        preds_val_neg = predict(self.pn_classifier, dataloader_val_neg)
        corrects_neg = preds_val_neg == y_val
        acc_neg = corrects_neg.float().mean()

        dataloader_val_pos = DataLoader(TensorDataset(X_val),
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=False)
        preds = predict(self.pn_classifier, dataloader_val_pos)
        corrects = preds == y_val
        acc_pos = corrects.float().mean()

        logger.info('Accuracy on X+: %f, on X-: %f.', acc_pos, acc_neg)

    def extract_features(self, X: Tensor) -> ArrayLike:
        """Extract Positive Negative similarity."""
        self.check_range(X)
        X_pos = X
        loader_pos = DataLoader(TensorDataset(X_pos),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)

        trainer = pl.Trainer(accelerator='auto',
                             logger=False,
                             enable_model_summary=False,
                             enable_progress_bar=False)

        # PyTorch Lightening trainer saves outputs as a list of mini-batches.
        outputs_pos = torch.vstack(trainer.predict(self.pn_classifier, loader_pos))
        # Use probability
        probs_pos = F.softmax(outputs_pos, dim=1)

        X_neg = 1 - X
        loader_neg = DataLoader(TensorDataset(X_neg),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        outputs_neg = torch.vstack(trainer.predict(self.pn_classifier, loader_neg))
        probs_neg = F.softmax(outputs_neg, dim=1)

        similarity = self.dist_fn(probs_pos, probs_neg)
        similarity = similarity.detach().numpy()
        return similarity

    def load(self, path: str = None) -> None:
        """Load PyTorch Lightening checkpoint file."""
        self.pn_classifier = get_lightning_module(self.data_name).load_from_checkpoint(path)

    @classmethod
    def check_range(cls, X):
        """Check the range of X. PN Classifier only works within [0, 1]."""
        min = X.min().item()
        max = X.max().item()
        assert np.isclose(min, 0) and np.isclose(max, 1), f'Expected range is [0, 1]. Got [{min}, {max}].'

    @classmethod
    def neg_cosine_similarity(cls, a, b):
        """Compute (1 - cosine_similarity)."""
        dist_fn = torch.nn.CosineSimilarity(dim=1)
        return 1 - dist_fn(a, b)
