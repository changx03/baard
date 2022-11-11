"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

Second Stage: Reliability

"""
import logging
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.detections.baard_applicability import ApplicabilityStage
from baard.detections.baard_detector import Detector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import batch_forward, dataloader2tensor, predict

logger = logging.getLogger(__name__)


class ReliabilityStage(Detector):
    """The 2nd stage of BAARD framework. It check the reliability of given
    examples. Can the new example be backed up by the training data?
    """

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 n_classes: int = 10,
                 k_neighbors: int = 20,
                 subsample_scale: float = 10.,
                 device: str = 'cuda',
                 ) -> None:
        super().__init__(model, data_name)

        self.n_classes = n_classes
        self.k_neighbors = k_neighbors
        self.subsample_scale = subsample_scale

        # Use the same feature space as Applicability Stage.
        self.latent_net = ApplicabilityStage.get_latent_net(model, data_name)

        if not torch.cuda.is_available() and device == 'cuda':
            logger.warning('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Register params
        self.params['n_classes'] = self.n_classes
        self.params['k_neighbors'] = self.k_neighbors
        self.params['subsample_scale'] = self.subsample_scale
        self.params['device'] = self.device

        # Tunable parameters:
        self.n_training_samples = None
        self.n_subset = None
        self.features_train = None
        self.features_labels = None

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        """Train detector. If X and y are None, use the training set from the classifier.
        """
        if X is None or y is None:
            loader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(loader_train)
            logger.warning('No sample is passed for training. Using the entire training set. %i examples.', X.size(0))

        latent_features, y_correct, _ = ApplicabilityStage.compute_correct_latent_features(
            X, y,
            model=self.model,
            latent_net=self.latent_net,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device)
        # Row-wise normalize
        self.features_train = nn.functional.normalize(latent_features, p=2, dim=1)
        self.features_labels = y_correct
        assert self.features_train.size(0) == self.features_labels.size(0)
        self.n_training_samples = self.features_train.size(0)
        self.n_subset = min(math.floor(self.subsample_scale * self.k_neighbors), self.n_training_samples)

    def extract_features(self, X: Tensor) -> ArrayLike:
        if self.features_train is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')
        n_samples = X.size(0)
        hidden_features = batch_forward(self.latent_net,
                                        X=X,
                                        batch_size=self.batch_size,
                                        device=self.device,
                                        num_workers=self.num_workers,
                                        )
        # Apply normalization
        hidden_features = nn.functional.normalize(hidden_features, p=2, dim=1)
        dataloader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                )
        preds = predict(self.model, dataloader)
        cosine_sim_fn = torch.nn.CosineSimilarity(dim=1)
        scores = torch.zeros(n_samples)
        pbar = tqdm(range(n_samples), total=n_samples)
        pbar.set_description('Extracting reliability features', refresh=False)
        for i in pbar:
            indices_train_as_sample = torch.where(self.features_labels == preds[i])[0]
            # The subset which is labelled as i is much smaller than the total training set.
            n_subset = min(len(indices_train_as_sample), self.n_subset)
            indices_subsample = np.random.choice(indices_train_as_sample,
                                                 size=n_subset,
                                                 replace=False)  # No replacement, no duplicates.
            # This subset should have the same label as X[i].
            feature_train_subset = self.features_train[indices_subsample]
            # Compute cosine similarity
            cos_similarity = cosine_sim_fn(feature_train_subset, hidden_features[i])

            # Cosine similarity is in range [-1, 1], where 0 is orthogonal vectors.
            # Compute angular distance
            angular_dist = torch.arccos(cos_similarity) / torch.pi

            # Find K-nearest neighbors. (dist, indices)
            dist_neighbors, _ = torch.topk(angular_dist, k=self.k_neighbors, largest=False)
            mean_dist = dist_neighbors.mean()
            scores[i] = mean_dist
        return scores.detach().numpy()

    def save(self, path: str = None) -> object:
        """Save pre-trained features. The ideal extension is `.baard2`."""
        if self.features_train is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, file_ext='.baard2')

        save_obj = {
            'features_train': self.features_train,
            'features_labels': self.features_labels,
            'n_training_samples': self.n_training_samples,
            'n_subset': self.n_subset,
        }
        pickle.dump(save_obj, open(path, 'wb'))
        return save_obj

    def load(self, path: str = None) -> None:
        """Load pre-trained parameters. The default extension is `.baard2`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            self.features_train = obj['features_train']
            self.features_labels = obj['features_labels']
            self.n_training_samples = obj['n_training_samples']
            self.n_subset = obj['n_subset']
        else:
            raise FileExistsError(f'{path} does not exist!')
