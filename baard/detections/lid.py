"""Implementing the paper "Characterizing Adversarial Subspaces Using Local
Intrinsic Dimensionality -- Ma et. al. (2018)
"""
import logging
import os
import pickle
import warnings
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegressionCV
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.attacks.apgd import auto_projected_gradient_descent
from baard.classifiers import DATASETS
from baard.detections.base_detector import Detector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import dataloader2tensor

logger = logging.getLogger(__name__)

ADV_BATCH_SIZE = 32


class LIDDetector(Detector):
    """Implement Local Intrinsic Dimensionality Detector in PyTorch."""

    def __init__(self,
                 model: pl.LightningModule,
                 data_name: str,
                 n_classes: int = 10,
                 clip_range: Tuple = (0., 1.),
                 attack_eps: float = 0.22,
                 attack_norm: Union[float, int] = np.inf,
                 noise_eps: float = 0.22,
                 k_neighbors: int = 20,
                 batch_size: int = 100,
                 device: str = 'cuda',
                 ):
        super().__init__(model, data_name)

        self.n_classes = n_classes
        self.clip_range = clip_range
        self.attack_eps = attack_eps
        self.attack_norm = attack_norm
        self.noise_eps = noise_eps
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size  # This batch size is different to training batch size.

        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Get all hidden layers
        self.latent_nets = self.get_hidden_layers()

        # Register params
        self.params['n_classes'] = self.n_classes
        self.params['clip_range'] = self.clip_range
        self.params['attack_eps'] = self.attack_eps
        self.params['attack_norm'] = self.attack_norm
        self.params['noise_eps'] = self.noise_eps
        self.params['k_neighbors'] = self.k_neighbors
        self.params['batch_size'] = self.batch_size
        self.params['device'] = self.device

        # Tunable parameters
        self.lid_neg = None
        self.lid_pos = None
        self.logistic_regressor = None

    def train(self, X: Tensor, y: Tensor) -> None:
        """Train detector. Train is not required for extracting features. If X and y are None,
        use the training set from the classifier.
        """
        if X is None or y is None:
            raise Exception('No sample is passed for training. LID does not use the entire training set!')

        X_noise = self.add_gaussian_noise(X, self.noise_eps, self.clip_range)

        X_adv = torch.zeros_like(X)
        # num_workers = os.cpu_count()
        num_workers = 16
        dataloader_adv = DataLoader(TensorDataset(X), batch_size=ADV_BATCH_SIZE,
                                    num_workers=num_workers, shuffle=False)
        start = 0
        pbar = tqdm(dataloader_adv, total=len(dataloader_adv))
        pbar.set_description('Running APGD mini-batch for LID', refresh=False)
        for batch in pbar:
            x_batch = batch[0]
            end = start + len(x_batch)
            X_adv[start:end] = auto_projected_gradient_descent(
                self.model,
                x_batch,
                norm=self.attack_norm,
                n_classes=self.n_classes,
                eps=self.attack_eps,
                nb_iter=100,
                clip_min=self.clip_range[0],
                clip_max=self.clip_range[1],
            )
            start = end

        n_samples = X.size(0)
        n_sequences = len(self.latent_nets)

        # NOTE: No guarantee X is not in the loader_train. Need Check the distance!
        loader_train = self.model.train_dataloader()
        if self.batch_size > loader_train.batch_size:
            # Don't have enough samples in the existing batch. Create new loader.
            X_train, y_train = dataloader2tensor(loader_train)
            loader_train = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True  # This must be True to ensure every time we have a different batch.
            )

        multi_layer_lid_clean = torch.zeros((n_samples, n_sequences))
        multi_layer_lid_noisy = torch.zeros((n_samples, n_sequences))
        multi_layer_lid_adv = torch.zeros((n_samples, n_sequences))

        pbar = tqdm(range(n_samples), total=n_samples)
        pbar.set_description('Training LID features', refresh=False)
        with torch.no_grad():
            for i in pbar:
                x_clean = X[i:i + 1]
                x_noisy = X_noise[i:i + 1]
                x_adv = X_adv[i:i + 1]
                # Find n neighbors in a mini-batch.
                x_batch, _ = next(iter(loader_train))  # Discard labels.
                x_batch = x_batch[:self.batch_size]  # Discard additional examples.

                x_mixed = torch.vstack([
                    x_clean,  # 0
                    x_noisy,  # 1
                    x_adv,    # 2
                    x_batch,  # 3
                ])

                for j, sequence in enumerate(self.latent_nets):
                    sequence = sequence.to(self.device)
                    sequence.eval()
                    latent_outputs = sequence(x_mixed.to(self.device))
                    latent_outputs = latent_outputs.flatten(start_dim=1).cpu()

                    # Outputs from the latent net:
                    out_clean = latent_outputs[:1]  # 1st is the output of x itself.
                    out_noisy = latent_outputs[1:2]
                    out_adv = latent_outputs[2:3]
                    out_neighbors = latent_outputs[3:]

                    # One sample LID:
                    one_lid_clean = self.get_MLE_LID(out_clean, out_neighbors, k=self.k_neighbors)
                    one_lid_noisy = self.get_MLE_LID(out_noisy, out_neighbors, k=self.k_neighbors)
                    one_lid_adv = self.get_MLE_LID(out_adv, out_neighbors, k=self.k_neighbors)

                    multi_layer_lid_clean[i, j] = one_lid_clean
                    multi_layer_lid_noisy[i, j] = one_lid_noisy
                    multi_layer_lid_adv[i, j] = one_lid_adv

        self.lid_neg = torch.vstack((multi_layer_lid_clean, multi_layer_lid_noisy))  # For benign examples.
        self.lid_pos = multi_layer_lid_adv  # For adversarial examples.

        # Row-wise normalize
        self.lid_neg = nn.functional.normalize(self.lid_neg, p=2, dim=1).numpy()
        self.lid_pos = nn.functional.normalize(self.lid_pos, p=2, dim=1).numpy()

        # Train Logistic Regression
        n_neg = len(self.lid_neg)
        n_pos = len(self.lid_pos)
        X_lid = np.vstack([self.lid_neg, self.lid_pos])
        y_lid = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

        self.logistic_regressor = LogisticRegressionCV(
            penalty='l1',  # Large feature space, prefer sparse weights
            solver='saga',  # Faster algorithm, but need standardized data.
            max_iter=10000,  # Default param is 100, which does not converge.
            n_jobs=self.num_workers,
        )
        self.logistic_regressor.fit(X_lid, y_lid)

    def extract_features(self, X: Tensor) -> ArrayLike:
        """Extract LID features."""
        n_samples = X.size(0)
        n_sequences = len(self.latent_nets)

        loader_train = self.model.train_dataloader()

        if self.batch_size > loader_train.batch_size:
            # Don't have enough samples in the existing batch. Create new loader.
            X_train, y_train = dataloader2tensor(loader_train)
            loader_train = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True  # This must be True to ensure every time we have a different batch.
            )

        multi_layer_lid = torch.zeros((n_samples, n_sequences))
        with torch.no_grad():
            pbar = tqdm(range(n_samples), total=n_samples)
            pbar.set_description('Extracting LID features', refresh=False)
            for i in pbar:
                x = X[i:i + 1]  # Keep dimension.
                # Find n neighbors in a mini-batch.
                X_pool, _ = next(iter(loader_train))  # Discard labels.
                X_pool = X_pool[:self.batch_size]  # To handle neighbor_batch_size < train_batch_size.
                net_inputs = torch.vstack([x, X_pool])

                for j, sequence in enumerate(self.latent_nets):
                    sequence = sequence.to(self.device)
                    sequence.eval()
                    latent_outputs = sequence(net_inputs.to(self.device))
                    latent_outputs = latent_outputs.flatten(start_dim=1).cpu()
                    out_x = latent_outputs[:1]  # 1st is the output of x itself.
                    out_neighbors = latent_outputs[1:]
                    one_sample_single_layer_lid = self.get_MLE_LID(out_x, out_neighbors, k=self.k_neighbors)
                    multi_layer_lid[i, j] = one_sample_single_layer_lid
        # multi_layer_lid = self.min_max_normalize(multi_layer_lid, dim=1)

        # Row-wise normalize
        multi_layer_lid = nn.functional.normalize(multi_layer_lid, p=2, dim=1).numpy()
        return multi_layer_lid

    def predict_proba(self, X: Tensor) -> ArrayLike:
        """Compute probability estimate based on LID."""
        if self.logistic_regressor is None:
            raise Exception('Logistic regression model is not trained yet!')
        X_lid = self.extract_features(X)
        probs = self.logistic_regressor.predict_proba(X_lid)
        return probs[:, 1]  # Only return the 2nd column

    def save(self, path: str = None) -> None:
        """Save extracted features for the training set. The ideal extension is `.lid`."""
        if self.lid_neg is None or self.lid_pos is None:
            raise Exception('No trained weights. Nothing to save.')

        path = create_parent_dir(path, file_ext='.lid')

        save_obj = {
            'lid_neg': self.lid_neg,
            'lid_pos': self.lid_pos,
            'logistic_regressor': self.logistic_regressor,
        }
        pickle.dump(save_obj, open(path, 'wb'))

    def load(self, path: str = None) -> None:
        """Load extracted features for the training set. The default extension is `.lid`."""
        if os.path.isfile(path):
            save_obj = pickle.load(open(path, 'rb'))
            self.lid_neg = save_obj['lid_neg']
            self.lid_pos = save_obj['lid_pos']
            self.logistic_regressor = save_obj['logistic_regressor']
        else:
            raise FileExistsError(f'{path} does not exist!')

    def get_hidden_layers(self) -> List:
        """Return a list of sequences which computes the latent outputs for each hidden layer.
        NOTE: ResNet18 need to call `model.model.children()` to get the list.
        """
        # get deep representations
        model = self.model
        device = self.device
        hidden_layers = []
        if self.data_name == DATASETS[0]:  # MNIST CNN
            for i in range(1, len(list(model.children()))):
                layer = nn.Sequential(*list(model.children())[:i]).to(device)
                hidden_layers.append(layer)
        elif self.data_name == DATASETS[1]:  # CIFAR10 ResNet18
            resnet18_list = list(model.model.children())
            n_layers = len(resnet18_list)
            for i in range(1, n_layers):
                layer = nn.Sequential(*resnet18_list[:i]).to(device)
                hidden_layers.append(layer)
        else:
            raise NotImplementedError
        logging.info('Found %d hidden layers.', len(hidden_layers))
        return hidden_layers

    @classmethod
    def add_gaussian_noise(cls, X: Tensor, eps: float, clip_range: Tuple = (0, 1)) -> Tensor:
        """Add Gaussian noise to X."""
        noise = torch.zeros_like(X).normal_(0., 1.)
        X_noisy = X + noise * eps
        X_noisy = torch.clamp(X_noisy, clip_range[0], clip_range[1])
        return X_noisy

    @classmethod
    def get_MLE_LID(cls, x: Tensor, batch: Tensor, k: int) -> Tensor:
        """Compute one sample Maximum Likelihood Estimator of LID within k nearest neighbors.
        It the batch contains x itself, it returns LID from k-1 neighbors.
        """
        pairwise_dist = torch.pairwise_distance(x, batch, p=2)
        # Nearest neighbors = smallest distance
        k_dist, _ = torch.topk(pairwise_dist, k=k, sorted=True, largest=False)  # Returns (values, indices)
        if np.isclose(k_dist[0].item(), 0):  # Find x itself
            k_dist = k_dist[1:]  # Discard itself.
        max_dist = k_dist[-1]  # the max neighbor distance
        lid = - len(k_dist) / (torch.log(k_dist / (max_dist + 1e-9)).sum() + 1e-9)
        return lid

    # @classmethod
    # def min_max_normalize(cls, X: Tensor, dim=1) -> Tensor:
    #     """Apply min-max normalization."""
    #     X_min = torch.min(X, dim=dim)[0].unsqueeze(dim=dim)
    #     X_max = torch.max(X, dim=dim)[0].unsqueeze(dim=dim)
    #     X_norm = (X - X_min) / (X_max - X_min)
    #     return X_norm
