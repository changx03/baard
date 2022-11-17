"""Implementing the paper "Mitigating Evasion Attacks to Deep Neural Networks
via Region-based Classification" -- Cao and Gong (2017)
"""
import logging
from typing import Any, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from tqdm import tqdm

from baard.detections.base_detector import SklearnDetector
from baard.utils.torch_utils import create_noisy_examples

logger = logging.getLogger(__name__)


class SklearnRegionBasedClassifier(SklearnDetector):
    """Implement Region-based Classifier in PyTorch"""

    def __init__(self,
                 model: Any,
                 data_name: str,
                 n_classes: int = 10,
                 radius: float = 0.2,
                 n_noise_samples: int = 1000,
                 noise_clip_range: Tuple = (0., 1.)
                 ):
        super().__init__(model, data_name)
        self.n_classes = n_classes
        self.radius = radius
        self.n_noise_samples = n_noise_samples
        self.noise_clip_range = noise_clip_range

        # Register params
        self.params['n_classes'] = self.n_classes
        self.params['radius'] = self.radius
        self.params['n_noise_samples'] = self.n_noise_samples
        self.params['noise_clip_range'] = self.noise_clip_range

    def train(self, X: ArrayLike = None, y: ArrayLike = None) -> None:
        """Train detector. X and y are dummy variables."""
        logger.warning('Region-based classifier does not require training.')

    def extract_features(self, X: ArrayLike) -> ArrayLike:
        """Extract probability estimates based on neighbors' outputs."""
        n_samples = len(X)
        # Get original predictions
        preds_origin = self.model.predict(X)

        results = []
        pbar = tqdm(range(n_samples), total=n_samples)
        pbar.set_description('Region-based prediction', refresh=False)
        for i in pbar:
            x = X[i]
            noise_eps = f'u{self.radius}'  # Uniform noise
            x_noisy = create_noisy_examples(torch.from_numpy(x),
                                            n_samples=self.n_noise_samples,
                                            noise_eps=noise_eps,
                                            clip_range=self.noise_clip_range)
            x_noisy = x_noisy.detach().numpy()
            preds = self.model.predict(x_noisy)
            preds_oh = np.zeros((self.n_noise_samples, self.n_classes), dtype=float)
            preds_oh[np.arange(self.n_noise_samples), preds] = 1
            preds_mean = preds_oh.mean(0)
            results.append(preds_mean)
        results = np.vstack(results)

        # Get neighbor probability for each example.
        features = np.array([results[i, preds_origin[i]] for i in np.arange(n_samples)])
        return features
