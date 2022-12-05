"""Base class for detectors."""
import os
from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor

from baard.utils.miscellaneous import create_parent_dir, to_json


class Detector(ABC):
    """Base class for a detector."""

    def __init__(self, model: LightningModule, data_name: str):
        self.model = model
        self.data_name = data_name

        # self.num_workers = os.cpu_count()
        self.num_workers = 16

        # Parameters from LightningModule:
        self.batch_size = self.model.train_dataloader().batch_size

        # These parameters can be saved as a JSON file.
        self.params = {
            'data_name': self.data_name,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
        }

    @abstractmethod
    def train(self, X: Tensor = None, y: Tensor = None):
        """Train detector."""
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, X: Tensor) -> ArrayLike:
        """Extract features from X."""
        raise NotImplementedError

    def save(self, path: str = None):
        """Save detector's tunable parameters."""
        print('This detector does not provide save feature.')

    def load(self, path: str = None):
        """Load pre-trained parameters."""
        print('This detector does not provide load feature.')

    def save_params(self, path: str = None):
        """Save internal parameters as a JSON file."""
        path = create_parent_dir(path, '.json')
        to_json(self.params, path)


class SklearnDetector(ABC):
    """Base class for a sklearn detector."""

    def __init__(self, model: Any, data_name: str):
        self.model = model
        self.data_name = data_name

        # These parameters can be saved as a JSON file.
        self.params = {
            'data_name': self.data_name,
        }

    @abstractmethod
    def train(self, X: ArrayLike, y: ArrayLike):
        """Train detector."""
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, X: ArrayLike) -> ArrayLike:
        """Extract features from X."""
        raise NotImplementedError

    def save(self, path: str = None):
        """Save detector's tunable parameters."""
        print('This detector does not provide save feature.')

    def load(self, path: str = None):
        """Load pre-trained parameters."""
        print('This detector does not provide load feature.')

    def save_params(self, path: str = None):
        """Save internal parameters as a JSON file."""
        path = create_parent_dir(path, '.json')
        to_json(self.params, path)
