"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

First Stage: Applicability
"""
import logging
import os
import pickle
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from baard.detections.base_detector import SklearnDetector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.sklearn_utils import get_correct_samples

logger = logging.getLogger(__name__)


class SklearnApplicabilityStage(SklearnDetector):
    """The 1st stage of BAARD framework. It check the applicability of given
    examples. How similar in the feature space between the example and the trining data.
    """

    def __init__(self,
                 model: Any,
                 data_name: str,
                 n_classes: int = 2,
                 ) -> None:
        super().__init__(model, data_name)

        self.n_classes = n_classes

        # Register params
        self.params['n_classes'] = self.n_classes

        # Tunable parameters
        self.zstats_dict = None

    def train(self, X: ArrayLike, y: ArrayLike) -> None:
        """Train detector."""
        X, y = get_correct_samples(self.model, X, y)

        # Initialize parameters
        self.zstats_dict = {c: {'mean': 0, 'std': 0} for c in range(self.n_classes)}

        pbar = tqdm(range(self.n_classes), total=self.n_classes)
        pbar.set_description('Training applicability', refresh=False)
        for c in pbar:
            # Get the subset that are labelled as `c`.
            indices_with_label_c = np.where(y == c)[0]
            X_subset = X[indices_with_label_c]
            # all examples in the subset have the same label, c.
            # Compute statistics on hidden features, column-wise.
            feature_mean = X_subset.mean(0)
            feature_std = X_subset.std(0)
            self.zstats_dict[c]['mean'] = feature_mean
            self.zstats_dict[c]['std'] = feature_std

    def extract_features(self, X: ArrayLike) -> ArrayLike:
        if self.zstats_dict is None:
            raise Exception('The detector have not trained yet. Call `train` or `load` first!')

        n_samples = len(X)
        preds = self.model.predict(X)
        scores = np.zeros(n_samples, dtype=float)

        pbar = tqdm(range(self.n_classes), total=self.n_classes)
        pbar.set_description('Extracting applicability features', refresh=False)
        for c in pbar:
            indices_with_label_c = np.where(preds == c)[0]
            n_subset = len(indices_with_label_c)
            if n_subset == 0:
                logger.info('No example is in Class [%i].', c)
                continue
            features_subset = X[indices_with_label_c]
            feature_mean = self.zstats_dict[c]['mean']
            feature_std = self.zstats_dict[c]['std']
            # Avoid divide by 0
            z_score = (features_subset - feature_mean) / (feature_std + 1e-9)
            # 2-tailed Z-score.
            z_max = np.abs(z_score).max(1)  # Return (max, indices_max).
            scores[indices_with_label_c] = z_max
        return scores

    def save(self, path: str = None) -> object:
        """Save pre-trained statistics. The ideal extension is `.skbaard1`."""
        if self.zstats_dict is None:
            raise Exception('No trained parameters. Nothing to save.')

        path = create_parent_dir(path, file_ext='.skbaard1')

        save_obj = {
            'zstats_dict': self.zstats_dict
        }
        pickle.dump(save_obj, open(path, 'wb'))
        return save_obj

    def load(self, path: str = None) -> None:
        """Load pre-trained parameters. The default extension is `.skbaard1`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            self.zstats_dict = obj['zstats_dict']
        else:
            raise FileExistsError(f'{path} does not exist!')
