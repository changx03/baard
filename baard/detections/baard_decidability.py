"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

Third Stage: Decidability
"""
import logging
import math
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.detections.baard_applicability import ApplicabilityStage
from baard.detections.base_detector import Detector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import batch_forward, dataloader2tensor

logger = logging.getLogger(__name__)


class DecidabilityStage(Detector):
    """The 3rd stage of BAARD framework. It check the decidability of given
    samples. Does the output of the example match the training data?
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
        # TODO: Change Scale to m, the sample size
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
        self.probs_correct = None

    def train(self, X: Tensor = None, y: Tensor = None):
        """Train detector. If X and y are None, use the training set from the classifier.
        """
        if X is None or y is None:
            loader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(loader_train)
            logger.warning('No sample is passed for training. Using the entire training set. %i examples.', X.size(0))

        with torch.no_grad():
            # Get correctly predicted hidden feature outputs. This is the same procedure as ReliabilityStage.
            latent_features, y_correct, loader_correct = ApplicabilityStage.compute_correct_latent_features(
                X, y,
                model=self.model,
                latent_net=self.latent_net,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device)

            # Row-wise normalize.
            self.features_train = nn.functional.normalize(latent_features, p=2, dim=1)
            self.features_labels = y_correct
            assert self.features_train.size(0) == self.features_labels.size(0)
            self.n_training_samples = self.features_train.size(0)
            self.n_subset = min(
                math.floor(self.subsample_scale * self.k_neighbors),
                self.n_training_samples
            )
            logger.info('Number of subset = %i', self.n_subset)

            # Also save the model's outputs.
            trainer = pl.Trainer(accelerator='auto',
                                 logger=False,
                                 enable_model_summary=False,
                                 enable_progress_bar=False)
            outputs_correct = torch.vstack(trainer.predict(self.model, loader_correct))
            # Neural network models do NOT have Softmax layer at the end. Run Softmax and save probability estimates.
            self.probs_correct = nn.functional.softmax(outputs_correct, dim=1)

    def extract_features(self, X: Tensor) -> ArrayLike:
        if self.features_train is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')
        n_samples = X.size(0)

        with torch.no_grad():
            # Get latent outputs.
            hidden_features = batch_forward(self.latent_net,
                                            X=X,
                                            batch_size=self.batch_size,
                                            device=self.device,
                                            num_workers=self.num_workers,
                                            )
            # Apply normalization.
            hidden_features = nn.functional.normalize(hidden_features, p=2, dim=1)

            # Get probability predictions.
            trainer = pl.Trainer(accelerator='auto',
                                 logger=False,
                                 enable_model_summary=False,
                                 enable_progress_bar=False)
            loader = DataLoader(TensorDataset(X),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers
                                )
            probs = torch.vstack(trainer.predict(self.model, loader))
            probs = nn.functional.softmax(probs, dim=1)

            indices_train = np.arange(self.n_training_samples)
            # Handle value error
            n_subset = min(self.n_subset, self.n_training_samples)

            cosine_sim_fn = torch.nn.CosineSimilarity(dim=1)
            scores = torch.zeros(n_samples)
            pbar = tqdm(range(n_samples), total=n_samples)
            pbar.set_description('Extracting decidability features', refresh=False)
            for i in pbar:
                class_highest = probs[i].argmax()
                highest_prob = probs[i, class_highest]

                if n_subset == self.n_training_samples:  # Use all data, no split
                    indices_subsample = indices_train
                    y_subset = self.features_labels
                else:
                    # Use train_test_split to achieve stratified sampling
                    indices_subsample, _, y_subset, _ = train_test_split(
                        indices_train, self.features_labels,
                        train_size=n_subset)  # NOTE: shuffle=True will disable stratify!

                # The subset contains all classes.
                feature_train_subset = self.features_train[indices_subsample]
                probs_train_subset = self.probs_correct[indices_subsample]

                # Compute cosine similarity.
                cos_similarity = cosine_sim_fn(feature_train_subset, hidden_features[i])
                # Compute angular distance
                angular_dist = torch.arccos(cos_similarity) / torch.pi
                # Find K-nearest neighbors. Output shape: (dist, indices).
                _, indices_neighbor = torch.topk(angular_dist, k=self.k_neighbors, largest=False)
                probs_neighbor = probs_train_subset[indices_neighbor]
                # assert torch.all(torch.argmax(probs_neighbor, 1) == y_subset[indices_neighbor])
                if not torch.all(torch.argmax(probs_neighbor, 1) == torch.from_numpy(y_subset)[indices_neighbor]):
                    logger.warning('Unmatched labels: %s', torch.where(torch.argmax(probs_neighbor, 1) != torch.from_numpy(y_subset)[indices_neighbor])[0])

                # Get probability estimates of the corresponding label.
                probs_neighbor = probs_neighbor[:, class_highest]
                neighbor_mean = probs_neighbor.mean()
                neighbor_str = probs_neighbor.std()
                # Avoid divide by 0.
                z_score = (highest_prob - neighbor_mean) / (neighbor_str + 1e-9)
                # 2-tailed Z-score.
                z_score = torch.abs(z_score)
                scores[i] = z_score
        return scores.detach().numpy()

    def save(self, path: str = None) -> object:
        """Save pre-trained features. The ideal extension is `.baard3`."""
        if self.features_train is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, file_ext='.baard3')

        save_obj = {
            'features_train': self.features_train,
            'features_labels': self.features_labels,
            'n_training_samples': self.n_training_samples,
            'n_subset': self.n_subset,
            'probs_correct': self.probs_correct,
        }
        pickle.dump(save_obj, open(path, 'wb'))
        return save_obj

    def load(self, path: str = None):
        """Load pre-trained parameters. The default extension is `.baard3`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            self.features_train = obj['features_train']
            self.features_labels = obj['features_labels']
            self.n_training_samples = obj['n_training_samples']
            self.n_subset = obj['n_subset']
            self.probs_correct = obj['probs_correct']
        else:
            raise FileExistsError(f'{path} does not exist!')
