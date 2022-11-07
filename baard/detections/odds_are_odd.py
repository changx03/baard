"""Implementing the paper "The Odds are Odd: A Statistical Test for Detecting
Adversarial Examples" -- Roth et. al. (2019)
"""
import os
import pickle
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.classifiers import DATASETS
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import (batch_forward, create_noisy_examples,
                                     dataloader2tensor, get_correct_examples,
                                     get_dataloader_shape, predict)
from .base_detector import Detector


class OddsAreOddDetector(Detector):
    """Implement Odds are odd detector in PyTorch."""

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 noise_list: List,
                 n_noise_samples: int = 100,
                 device: str = 'cuda',
                 ):
        super().__init__(model, data_name)

        self.noise_list = noise_list
        self.n_noise_samples = n_noise_samples

        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Parameters based on data:
        latent_net, weight, n_classes, clip_range = self.get_odds_params_from_data(data_name, model)
        self.latent_net = latent_net
        self.weight = weight
        self.n_classes = n_classes
        self.noise_clip_range = clip_range
        self.weight_diff = self.weight.unsqueeze(0) - self.weight.unsqueeze(1)

        # Register params
        self.params['noise_list'] = self.noise_list
        self.params['device'] = self.device
        self.params['n_classes'] = self.n_classes
        self.params['noise_clip_range'] = self.noise_clip_range

        # Tunable parameters:
        self.weights_stats = None

    def train(self, X: Tensor, y: Tensor) -> None:
        # Check predictions and true labels
        dataloader = DataLoader(TensorDataset(X, y),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        correct_dataloader = get_correct_examples(self.model, dataloader)
        x_corr_shape = get_dataloader_shape(correct_dataloader)
        if x_corr_shape[0] != len(X):
            warnings.warn(f'{len(X) - x_corr_shape.size(0)} are classified incorrectly! Use {len(x_corr_shape)} examples instead.')

        # only uses correctly classified examples.
        X, y = dataloader2tensor(correct_dataloader)

        # Train weights
        weights_stats = self.__collect_weights_stats(X, preds=y)
        self.weights_stats = weights_stats

    def extract_features(self, X: Tensor) -> ArrayLike:
        if self.weights_stats is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')

        n = X.size(0)
        dataloader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False)
        preds = predict(self.model, dataloader)
        scores = np.zeros(n)
        pbar = tqdm(range(n), total=n)
        pbar.set_description('Odds inference time', refresh=False)
        for i in pbar:
            _x = X[i]
            _pred = preds[i]
            z_max = self.detect_single_example(_x, _pred)
            scores[i] = z_max
        return scores

    def detect_single_example(self, x: Tensor, pred: int) -> ArrayLike:
        """Detect single example. Return the absolute maximum Z-score."""
        if isinstance(pred, Tensor):
            pred = int(pred.item())

        with torch.no_grad():
            alignments = self.__compute_multiple_noise_alignments(x, pred=pred)

            max_Z_scores = []
            for noise_eps in self.noise_list:
                score = alignments[noise_eps].mean(0)
                mean = self.weights_stats[(pred, noise_eps)]['mean']
                std = self.weights_stats[(pred, noise_eps)]['std']
                z_score = (score - mean) / std
                # 2 tailed Z-score
                z_max = np.abs(z_score.detach().numpy()).max()
                max_Z_scores.append(z_max)

        return np.max(max_Z_scores)

    def save(self, path: str = None) -> None:
        """Save weight statistics as binary. The ideal extension is `.odds`. """
        if self.weights_stats is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, '.odds')

        pickle.dump(self.weights_stats, open(path, 'wb'))

    def load(self, path: str = None) -> None:
        """Load pre-trained statistics. The default extension is `.odds`."""
        if os.path.isfile(path):
            self.weights_stats = pickle.load(open(path, 'rb'))
        else:
            raise FileExistsError(f'{path} does not exist!')

    @classmethod
    def get_negative_labels(cls, y, n_classes=10) -> ArrayLike:
        """Get every label except the true label."""
        labels = np.arange(n_classes)
        return labels[labels != y]

    @classmethod
    def get_odds_params_from_data(cls, data_name: str, model: LightningModule) -> Tuple[Module, Tensor, int, Tuple]:
        """Get Odds are odd parameters based on dataset."""
        if data_name == DATASETS[0]:  # MNIST
            latent_net = torch.nn.Sequential(
                model.conv1,
                model.relu1,
                model.conv2,
                model.relu2,
                model.pool1,
                model.flatten,
                model.fc1,
            )
            weight = list(model.children())[-1].weight
            n_classes = 10
            clip_range = (0, 1)
            return latent_net, weight, n_classes, clip_range
        elif data_name == DATASETS[1]:  # CIFAR10
            # TODO: sequential model and weights for CIFAR10
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __compute_single_noise_alignment(self, hidden_out_x: Tensor, hidden_out_noise: Tensor,
                                         negative_labels, weight_relevant: Tensor
                                         ) -> Tensor:
        """Compute the odds on noisy examples (1 noise type) for a single input."""
        hidden_output_diff = hidden_out_x - hidden_out_noise
        odds = torch.matmul(hidden_output_diff, weight_relevant.transpose(1, 0))   # Size: [n_samples, n_classes]
        odds = odds[:, negative_labels]  # Size: [n_samples, n_classes-1]
        return odds

    def __compute_multiple_noise_alignments(self, x: Tensor, pred: int,
                                            ) -> OrderedDict:
        """Compute all odds for every noise on a single input."""
        if isinstance(pred, Tensor):
            pred = int(pred.item())

        hidden_out_x = batch_forward(self.latent_net, x.unsqueeze(0),
                                     num_workers=self.num_workers, device=self.device)
        weight_relevant = self.weight_diff[:, pred]
        negative_labels = self.get_negative_labels(pred, self.n_classes)
        alignments = OrderedDict()
        for noise_eps in self.noise_list:
            x_noisy = create_noisy_examples(x,
                                            n_samples=self.n_noise_samples,
                                            noise_eps=noise_eps,
                                            clip_range=self.noise_clip_range)
            hidden_out_noise = batch_forward(self.latent_net, x_noisy,
                                             num_workers=self.num_workers, device=self.device)
            odds = self.__compute_single_noise_alignment(hidden_out_x,
                                                         hidden_out_noise,
                                                         negative_labels,
                                                         weight_relevant)
            alignments[noise_eps] = odds
        return alignments

    def __collect_weights_stats(self, X: Tensor, preds: Tensor) -> Dict:
        """Collect weights stats from a dataset."""
        weights_stats = {(c, noise_eps): [] for c in range(self.n_classes) for noise_eps in self.noise_list}

        with torch.no_grad():
            pbar = tqdm(zip(X, preds), total=len(X))
            pbar.set_description('Collecting weight stats', refresh=False)
            for _x, _y in pbar:
                label = int(_y.item())
                alignments = self.__compute_multiple_noise_alignments(_x, label)
                for noise_eps in alignments:
                    weights_stats[(label, noise_eps)].append(alignments[noise_eps])

        for key in weights_stats:
            weights = torch.vstack(weights_stats[key])
            # Replace the weights with mean and std.
            weights_stats[key] = {'mean': weights.mean(0), 'std': weights.std(0)}
        return weights_stats
