"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

First Stage: Applicability
"""
import logging
import os
import pickle

import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.classifiers import DATASETS
from baard.detections.base_detector import Detector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import (batch_forward, dataloader2tensor,
                                     get_correct_examples, predict)

logger = logging.getLogger(__name__)


class ApplicabilityStage(Detector):
    """The 1st stage of BAARD framework. It check the applicability of given
    examples. How similar in the feature space between the example and the trining data.
    """

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 n_classes: int = 10,
                 device: str = 'cuda',
                 ) -> None:
        super().__init__(model, data_name)

        self.n_classes = n_classes
        self.latent_net = self.get_latent_net(self.model, self.data_name)

        if not torch.cuda.is_available() and device == 'cuda':
            logger.warning('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Register params
        self.params['n_classes'] = self.n_classes
        self.params['device'] = self.device

        # Tunable parameters
        self.zstats_dict = None

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        """Train detector. If X and y are None, use the training set from the classifier.
        """
        if X is None or y is None:
            loader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(loader_train)
            logger.warning('No sample is passed for training. Using the entire training set. %i examples.', X.size(0))

        logger.info('[Latent Net]\n%s', self.latent_net)

        # Get correctly predicted hidden feature outputs. This is the same procedure as ReliabilityStage.
        latent_features, y_correct, _ = self.compute_correct_latent_features(
            X, y,
            model=self.model,
            latent_net=self.latent_net,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device)
        y = y_correct  # Override y.

        # Initialize parameters
        self.zstats_dict = {c: {'mean': 0, 'std': 0} for c in range(self.n_classes)}

        pbar = tqdm(range(self.n_classes), total=self.n_classes)
        pbar.set_description('Training applicability', refresh=False)
        for c in pbar:
            # Get the subset that are labelled as `c`.
            indices_with_label_c = torch.where(y == c)[0]
            features_subset = latent_features[indices_with_label_c]
            # all examples in the subset have the same label, c.
            # Compute statistics on hidden features, column-wise.
            latent_feature_mean = features_subset.mean(dim=0)
            latent_feature_std = features_subset.std(dim=0)
            self.zstats_dict[c]['mean'] = latent_feature_mean
            self.zstats_dict[c]['std'] = latent_feature_std

    def extract_features(self, X: Tensor) -> ArrayLike:
        if self.zstats_dict is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')

        n_samples = X.size(0)
        hidden_features = batch_forward(self.latent_net,
                                        X=X,
                                        batch_size=self.batch_size,
                                        device=self.device,
                                        num_workers=self.num_workers,
                                        )
        dataloader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                )
        preds = predict(self.model, dataloader)
        scores = torch.zeros(n_samples)

        pbar = tqdm(range(self.n_classes), total=self.n_classes)
        pbar.set_description('Extracting applicability features', refresh=False)
        for c in pbar:
            indices_with_label_c = torch.where(preds == c)[0]
            n_subset = len(indices_with_label_c)
            if n_subset == 0:
                logger.info('No example is in Class [%i].', c)
                continue
            features_subset = hidden_features[indices_with_label_c]
            latent_feature_mean = self.zstats_dict[c]['mean']
            latent_feature_std = self.zstats_dict[c]['std']
            # Avoid divide by 0
            z_score = (features_subset - latent_feature_mean) / (latent_feature_std + 1e-9)
            # 2-tailed Z-score.
            z_max, _ = torch.abs(z_score).max(dim=1)  # Return (max, indices_max).
            scores[indices_with_label_c] = z_max
        return scores.detach().numpy()

    def save(self, path: str = None) -> object:
        """Save pre-trained statistics. The ideal extension is `.baard1`."""
        if self.zstats_dict is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, file_ext='.baard1')

        save_obj = {
            'zstats_dict': self.zstats_dict
        }
        pickle.dump(save_obj, open(path, 'wb'))
        return save_obj

    def load(self, path: str = None) -> None:
        """Load pre-trained parameters. The default extension is `.baard1`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            self.zstats_dict = obj['zstats_dict']
        else:
            raise FileExistsError(f'{path} does not exist!')

    @classmethod
    def get_latent_net(cls, model, data_name):
        """Get the latent net for extracting hidden features."""
        latent_net = None
        if data_name == DATASETS[0]:  # MNIST
            latent_net = nn.Sequential(*list(model.children())[:7])
        elif data_name == DATASETS[1]:  # CIFAR10
            latent_net = nn.Sequential(
                *list(model.model.children())[:-1],
                nn.Flatten(start_dim=1)
            )
        else:
            raise NotImplementedError()
        return latent_net

    @classmethod
    def compute_correct_latent_features(cls,
                                        X: Tensor,
                                        y: Tensor,
                                        model: LightningModule,
                                        latent_net: Module,
                                        batch_size: int,
                                        num_workers: int,
                                        device: str
                                        ) -> tuple[Tensor, Tensor]:
        """Compute latent features and labels from correctly predicted samples."""
        # Check predictions and true labels
        n_samples = X.size(0)
        dataloader = DataLoader(TensorDataset(X, y),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)
        loader_correct = get_correct_examples(model, dataloader)
        n_correct_samples = len(loader_correct.dataset)
        if len(loader_correct.dataset) != len(X):
            logger.warning('%i samples are classified incorrectly! Use %i examples instead.', n_samples - n_correct_samples, n_correct_samples)

        # only uses correctly classified examples.
        X_correct, y_correct = dataloader2tensor(loader_correct)
        latent_features = batch_forward(latent_net,
                                        X=X_correct,
                                        batch_size=batch_size,
                                        device=device,
                                        num_workers=num_workers,
                                        )
        return latent_features, y_correct, loader_correct
