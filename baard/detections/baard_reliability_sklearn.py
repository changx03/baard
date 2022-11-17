"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

Second Stage: Reliability

"""
import logging
import os
import pickle

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from tqdm import tqdm

from baard.detections.base_detector import SklearnDetector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.sklearn_utils import get_correct_samples

logger = logging.getLogger(__name__)


class SklearnReliabilityStage(SklearnDetector):
    """The 2nd stage of BAARD framework. It check the reliability of given
    examples. Can the new example be backed up by the training data?
    """

    def __init__(self,
                 model: ClassifierMixin,
                 data_name: str,
                 n_classes: int = 10,
                 k_neighbors: int = 1,
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

    def train(self, X: ArrayLike, y: ArrayLike):
        """Train detector."""
        # Check classifier's accuracy.
        X, y = get_correct_samples(self.model, X, y)

        # Expect the input has been normalized!
        self.n_training_samples = len(X)
        self.features_train = X
        self.features_labels = y

    def extract_features(self, X: ArrayLike) -> ArrayLike:
        if self.features_train is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')
        n_samples = len(X)

        preds = self.model.predict(X)
        cosine_sim_fn = torch.nn.CosineSimilarity(dim=1)
        scores = torch.zeros(n_samples)
        with torch.no_grad():
            pbar = tqdm(range(n_samples), total=n_samples)
            pbar.set_description('Extracting reliability features', refresh=False)
            for i in pbar:
                indices_train_as_sample = np.where(self.features_labels == preds[i])[0]
                n_same_class_samples = len(indices_train_as_sample)

                # The subset which is labelled as i is much smaller than the total training set.
                n_subset = min(n_same_class_samples, self.sample_size)
                indices_subsample = np.random.choice(indices_train_as_sample,
                                                     size=n_subset,
                                                     replace=False)  # No replacement, no duplicates.
                # This subset should have the same label as X[i].
                feature_train_subset = self.features_train[indices_subsample]
                # Compute cosine similarity
                cos_similarity = cosine_sim_fn(
                    torch.from_numpy(feature_train_subset),
                    torch.from_numpy(X[i])
                )

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
        }
        pickle.dump(save_obj, open(path, 'wb'))
        return save_obj

    def load(self, path: str = None):
        """Load pre-trained parameters. The default extension is `.baard2`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            self.features_train = obj['features_train']
            self.features_labels = obj['features_labels']
            self.n_training_samples = obj['n_training_samples']
        else:
            raise FileExistsError(f'{path} does not exist!')
