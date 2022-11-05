"""Base class and constants for detectors."""
import os
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor

from .baard_applicability import ApplicabilityStage
from .baard_decidability import DecidabilityStage
from .baard_detector import BAARD
from .baard_reliability import ReliabilityStage
from .feature_squeezing import FeatureSqueezingDetector
from .lid import LIDDetector
from .ml_loo import MLLooDetector
from .odds_are_odd import OddsAreOddDetector
from .pn_detector import PNDetector
from .region_based_classifier import RegionBasedClassifier


class Detector(ABC):
    """Base class for a detector."""

    def __init__(self, model: LightningModule, data_name: str) -> None:
        self.model = model
        self.data_name = data_name

        self.num_workers = os.cpu_count()

        # Parameters from LightningModule:
        self.batch_size = self.model.train_dataloader().batch_size

    @abstractmethod
    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        """Train detector."""
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, X: Tensor) -> ArrayLike:
        """Extract features from X."""
        raise NotImplementedError

    def save(self, path: str = None) -> None:
        """Save detector's tunable parameters."""
        print('This detector does not provide save feature.')

    def load(self, path: str = None) -> None:
        """Load pre-trained parameters."""
        print('This detector does not provide load feature.')
