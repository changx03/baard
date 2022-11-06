"""Implementing the paper "Mitigating Evasion Attacks to Deep Neural Networks
via Region-based Classification" -- Cao and Gong (2017)
"""
import logging
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.utils.torch_utils import create_noisy_examples, predict
from .base_detector import Detector

logger = logging.getLogger(__name__)


class RegionBasedClassifier(Detector):
    """Implement Region-based Classifier in PyTorch"""

    def __init__(self,
                 model: pl.LightningModule,
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

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        """Train detector. X and y are dummy variables."""
        logger.warning('Region-based classifier does not require training.')

    def extract_features(self, X: Tensor) -> ArrayLike:
        """Extract probability estimates based on neighbors' outputs."""
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        n_samples = X.size(0)
        trainer = pl.Trainer(accelerator='auto',
                             logger=False,
                             enable_model_summary=False,
                             enable_progress_bar=False)
        # Get original predictions
        dataloader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        preds_origin = predict(self.model, dataloader, trainer)

        results = []
        pbar = tqdm(range(n_samples), total=n_samples)
        pbar.set_description('Region-based prediction', refresh=False)
        for i in pbar:
            x = X[i]
            noise_eps = f'u{self.radius}'  # Uniform noise
            x_noisy = create_noisy_examples(x,
                                            n_samples=self.n_noise_samples,
                                            noise_eps=noise_eps,
                                            clip_range=self.noise_clip_range)
            dataloader = DataLoader(TensorDataset(x_noisy),
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    shuffle=False)
            preds = predict(self.model, dataloader, trainer)
            preds_oh = F.one_hot(preds, num_classes=self.n_classes)
            preds_mean = preds_oh.float().mean(dim=0)
            results.append(preds_mean)
        results = torch.vstack(results)

        # Get neighbor probability for each example.
        indices = preds_origin.long().unsqueeze(dim=1)
        results = results.gather(dim=1, index=indices).squeeze()
        return results.detach().numpy()

    def predict(self, X: Tensor) -> Tensor:
        """Making prediction using Region-based classifier."""
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        n_samples = X.size(0)
        trainer = pl.Trainer(accelerator='auto',
                             logger=False,
                             enable_model_summary=False,
                             enable_progress_bar=False)
        results = []
        pbar = tqdm(range(n_samples), total=n_samples)
        pbar.set_description('Region-based prediction', refresh=False)
        for i in pbar:
            x = X[i]
            noise_eps = f'u{self.radius}'  # Uniform noise
            x_noisy = create_noisy_examples(x,
                                            n_samples=self.n_noise_samples,
                                            noise_eps=noise_eps,
                                            clip_range=self.noise_clip_range)
            dataloader = DataLoader(TensorDataset(x_noisy),
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    shuffle=False)
            preds = predict(self.model, dataloader, trainer)
            mode = preds.mode()[0]  # (values, indices)
            results.append(mode)
        results = torch.tensor(results)
        return results
