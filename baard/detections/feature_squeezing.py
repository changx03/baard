"""Implementing the paper "Feature Squeezing: Detecting Adversarial Examples in
Deep Neural Networks" -- Xu et. al. (2018)
"""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from baard.classifiers import DATASETS, get_lightning_module
from baard.utils.torch_utils import dataloader2tensor
from .base_detector import Detector

SQUEEZER = ('depth', 'median', 'nl_mean')


class Squeezer(ABC):
    """Base class for squeezers."""

    def __init__(self, name: str, x_min: float, x_max: float):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max

    @abstractmethod
    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform X."""
        raise NotImplementedError


class DepthSqueezer(Squeezer):
    """Bit Depth Squeezer"""

    def __init__(self, x_min=0.0, x_max=1.0, bit_depth=4):
        super().__init__('depth', x_min, x_max)
        self.bit_depth = bit_depth

    def transform(self, X):
        max_val = np.rint(2 ** self.bit_depth - 1)
        X_transformed = np.rint(X * max_val) / max_val
        X_transformed = X_transformed * (self.x_max - self.x_min)
        X_transformed += self.x_min
        return np.clip(X_transformed, self.x_min, self.x_max)


class MedianSqueezer(Squeezer):
    """Median Filter Squeezer"""

    def __init__(self, x_min=0.0, x_max=1.0, kernel_size=2):
        super().__init__('median', x_min, x_max)
        self.kernel_size = kernel_size

    def transform(self, X):
        X_transformed = np.zeros_like(X, dtype=np.float32)
        for i in range(len(X)):
            X_transformed[i] = scipy.ndimage.median_filter(
                X[i], size=self.kernel_size)
        return np.clip(X_transformed, self.x_min, self.x_max)


class NLMeansColourSqueezer(Squeezer):
    """OpenCV FastNLMeansDenoisingColored Squeezer"""

    def __init__(self, x_min=0.0, x_max=1.0, h=2, templateWindowsSize=3, searchWindowSize=13):
        super().__init__('NLMeans', x_min, x_max)
        self.h = h
        self.templateWindowsSize = templateWindowsSize
        self.searchWindowSize = searchWindowSize

    def transform(self, X):
        import cv2

        X = np.moveaxis(X, 1, -1)
        outputs = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            img = (X[i].copy() * 255.0).astype('uint8')
            outputs[i] = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=self.h,
                hColor=self.h,
                templateWindowSize=self.templateWindowsSize,
                searchWindowSize=self.searchWindowSize)
        outputs = np.moveaxis(outputs, -1, 1) / 255.0
        return np.clip(outputs, self.x_min, self.x_max)


class FeatureSqueezingDetector(Detector):
    """Implement Feature Squeezing Detector in PyTorch."""

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 path_model: str,
                 max_epochs: int = 30,
                 path_checkpoint: str = 'logs',
                 seed: int = None,
                 verbose: bool = True
                 ):
        super().__init__(model, data_name)

        self.path_model = path_model
        self.max_epochs = max_epochs
        self.path_checkpoint = path_checkpoint
        self.seed = seed
        self.verbose = verbose

        # Number of classifiers = Number of squeezers
        self.squeezers = self.get_squeezers()
        self.squeezed_models = {key: get_lightning_module(data_name).load_from_checkpoint(path_model)
                                for key in self.squeezers}

        # Register params
        self.params['path_model'] = self.path_model
        self.params['path_checkpoint'] = self.path_checkpoint
        self.params['seed'] = self.seed
        self.params['verbose'] = self.verbose

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        """Train detector. X and y are dummy variables."""
        for squeezer_name, squeezer in self.squeezers.items():
            print(f'Training {squeezer_name} classifier...')

            # Apply filter on the training set
            dataloader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(dataloader_train)
            X_transformed = squeezer.transform(X.detach().numpy())
            X_transformed = torch.from_numpy(X_transformed).type(torch.HalfTensor)
            # Float16 for X, and Long32 for y.
            dataset = TensorDataset(X_transformed.type(torch.HalfTensor), y.type(torch.long))
            dataloader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    shuffle=True)  # Need shuffle for training.

            classifier = self.squeezed_models[squeezer_name]
            model_name = f'FeatureSqueezer_{self.data_name}_{squeezer_name}'
            logger = TensorBoardLogger(save_dir=self.path_checkpoint, name=model_name)
            trainer = pl.Trainer(
                accelerator='auto',
                precision=16,
                max_epochs=self.max_epochs,
                num_sanity_val_steps=0,
                logger=logger,
                callbacks=[
                    LearningRateMonitor(logging_interval='step'),
                ],
                enable_model_summary=False,  # Issues with state_dice()
            )
            trainer.fit(classifier, train_dataloaders=dataloader)

            # Evaluation
            if self.verbose:
                dataloader_val = self.model.val_dataloader()
                X, y = dataloader2tensor(dataloader_val)
                X_transformed = squeezer.transform(X.detach().numpy())
                X_transformed = torch.from_numpy(X_transformed)
                dataset = TensorDataset(X_transformed.type(torch.HalfTensor), y.type(torch.long))
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=False)  # Must be False for testing.
                trainer.test(classifier, dataloader)

    def extract_features(self, X: Tensor) -> ArrayLike:
        """Extract Max L1 distance between squeezed models' outputs."""
        trainer = pl.Trainer(accelerator='auto',
                             logger=False,
                             enable_model_summary=False,
                             enable_progress_bar=False)
        dist_fn = torch.nn.PairwiseDistance(p=1)
        probability_outputs = []

        # Get original outputs.
        dataloader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)  # Must be False for testing.
        outputs = torch.vstack(trainer.predict(self.model, dataloader))
        probs = F.softmax(outputs, dim=1)
        probability_outputs.append(probs)

        # Get probability estimates from each squeezed model.
        for squeezer_name, squeezer in self.squeezers.items():
            X_transformed = squeezer.transform(X.detach().numpy())
            X_transformed = torch.from_numpy(X_transformed)
            # Don't need 16 precession as inference time.
            dataloader_transformed = DataLoader(TensorDataset(X_transformed),
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                shuffle=False)  # Must be False for testing.
            classifier = self.squeezed_models[squeezer_name]
            outputs = torch.vstack(trainer.predict(classifier, dataloader_transformed))
            # Use probability
            probs = F.softmax(outputs, dim=1)
            probability_outputs.append(probs)

        # Compute pairwise distance.
        scores = []
        for i in range(1, len(probability_outputs)):
            pairwise_dist = dist_fn(probability_outputs[i - 1], probability_outputs[i])
            scores.append(pairwise_dist)

        # At least 1 filter is used + outputs from original model.
        # max function returns (max, max_indices)
        max_dist = torch.vstack(scores).max(dim=0)[0].detach().numpy()
        return max_dist

    def load(self, path_list: Dict) -> None:
        """Load a dictionary of PyTorch Lightening checkpoint files, e.g.,
        [{<SQUEEZER_NAME>: <PATH_CHECKPOINT>}].
        """
        for fs_key, path_checkpoint in path_list.items():
            self.squeezed_models[fs_key] = get_lightning_module(self.data_name).load_from_checkpoint(path_checkpoint)

    def get_squeezers(self) -> Dict:
        """Return a dictionary of pre-defined squeezers (filters)."""
        squeezers = {}
        if self.data_name == DATASETS[0]:  # MNIST
            squeezers[SQUEEZER[0]] = DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=1)
            squeezers[SQUEEZER[1]] = MedianSqueezer(x_min=0.0, x_max=1.0, kernel_size=2)
        elif self.data_name == DATASETS[1]:  # CIFAR10
            squeezers[SQUEEZER[0]] = DepthSqueezer(x_min=0.0, x_max=1.0, bit_depth=4)
            squeezers[SQUEEZER[1]] = MedianSqueezer(x_min=0.0, x_max=1.0, kernel_size=2)
            squeezers[SQUEEZER[2]] = NLMeansColourSqueezer(x_min=0.0, x_max=1.0, h=2, templateWindowsSize=3, searchWindowSize=13)

        else:
            raise NotImplementedError
        return squeezers
