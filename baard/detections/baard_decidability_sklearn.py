"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

Third Stage: Decidability
"""
import logging
import os
import pickle

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from baard.detections.base_detector import SklearnDetector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.sklearn_utils import get_correct_samples

logger = logging.getLogger(__name__)


class SklearnDecidabilityStage(SklearnDetector):
    """The 3rd stage of BAARD framework. It check the decidability of given
    samples. Does the output of the example match the training data?
    """

    def __init__(self,
                 model: ClassifierMixin,
                 data_name: str,
                 n_classes: int = 2,
                 k_neighbors: int = 15,
                 sample_size: int = 1000,
                 ) -> None:
        super().__init__(model, data_name)

        self.n_classes = n_classes
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size

        # Register params
        self.params['n_classes'] = self.n_classes
        self.params['k_neighbors'] = self.k_neighbors
        self.params['sample_size'] = self.sample_size

        # Tunable parameters:
        self.n_training_samples = None
        self.features_train = None
        self.features_labels = None
        self.probs_correct = None

    def train(self, X: ArrayLike, y: ArrayLike):
        """Train detector."""
        # Check classifier's accuracy.
        X, y = get_correct_samples(self.model, X, y)

        # Expect the input has been normalized!
        self.n_training_samples = len(X)
        self.features_train = X
        self.features_labels = y
        self.probs_correct = self.model.predict_proba(X)

    def extract_features(self, X: ArrayLike) -> ArrayLike:
        if self.features_train is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')
        n_samples = len(X)

        with torch.no_grad():
            X_torch = torch.from_numpy(X)  # Use PyTOrch to compute CosineSimilarity.

            # Get probability predictions.
            probs = self.model.predict_proba(X)
            probs_torch = torch.from_numpy(probs)

            indices_train = np.arange(self.n_training_samples)
            # Handle value error
            n_subset = min(self.sample_size, self.n_training_samples)

            cosine_sim_fn = torch.nn.CosineSimilarity(dim=1)
            scores = torch.zeros(n_samples)
            pbar = tqdm(range(n_samples), total=n_samples)
            pbar.set_description('Extracting decidability features', refresh=False)
            for i in pbar:
                class_highest = probs_torch[i].argmax()
                highest_prob = probs_torch[i, class_highest]

                if n_subset == len(indices_train):  # Use all data, no split
                    indices_subsample = indices_train
                    y_subset = self.features_labels
                else:
                    # Use train_test_split to achieve stratified sampling
                    indices_subsample, _, y_subset, _ = train_test_split(
                        indices_train, self.features_labels,
                        train_size=n_subset)  # NOTE: shuffle=True will disable stratify!

                # The subset contains all classes.
                feature_train_subset = self.features_train[indices_subsample]  # numpy array
                probs_train_subset = self.probs_correct[indices_subsample]  # numpy array
                probs_train_subset_torch = torch.from_numpy(probs_train_subset)

                # Compute cosine similarity.
                cos_similarity = cosine_sim_fn(
                    torch.from_numpy(feature_train_subset), X_torch[i])
                # Compute angular distance
                angular_dist = torch.arccos(cos_similarity) / torch.pi  # torch Tensor
                # Find K-nearest neighbors. Output shape: (dist, indices).
                _, indices_neighbor = torch.topk(angular_dist, k=self.k_neighbors, largest=False)
                probs_neighbor = probs_train_subset_torch[indices_neighbor]
                # assert torch.all(torch.argmax(probs_neighbor, 1) == torch.from_numpy(y_subset)[indices_neighbor])
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
        """Save pre-trained features. The ideal extension is `.skbaard3`."""
        if self.features_train is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, file_ext='.skbaard3')

        save_obj = {
            'features_train': self.features_train,
            'features_labels': self.features_labels,
            'n_training_samples': self.n_training_samples,
            'probs_correct': self.probs_correct,
        }
        pickle.dump(save_obj, open(path, 'wb'))
        return save_obj

    def load(self, path: str = None):
        """Load pre-trained parameters. The default extension is `.skbaard3`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            self.features_train = obj['features_train']
            self.features_labels = obj['features_labels']
            self.n_training_samples = obj['n_training_samples']
            self.probs_correct = obj['probs_correct']
        else:
            raise FileExistsError(f'{path} does not exist!')
