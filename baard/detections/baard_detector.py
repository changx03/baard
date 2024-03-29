"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.

Combine 3 Stages together.
"""
import logging
import os
import pickle

import numpy as np
import torch
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor

from baard.detections.baard_applicability import ApplicabilityStage
from baard.detections.baard_decidability import DecidabilityStage
from baard.detections.baard_reliability import ReliabilityStage
from baard.detections.base_detector import Detector
from baard.utils.miscellaneous import create_parent_dir

logger = logging.getLogger(__name__)


class BAARD(Detector):
    """ Implementing BAARD (Blocking Adversarial examples by testing
    Applicability, Reliability, and Decidability)
    """

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 n_classes: int = 10,
                 k1_neighbors: int = 5,
                 sample_size1: int = 1000,
                 k2_neighbors: int = 100,
                 sample_size2: int = 5000,
                 device: str = 'cuda',
                 ) -> None:
        super().__init__(model, data_name)
        self.n_classes = n_classes

        # Reliability Stage specific:
        self.k1_neighbors = k1_neighbors
        self.sample_size1 = sample_size1

        # Decidability Stage specific:
        self.k2_neighbors = k2_neighbors
        self.sample_size2 = sample_size2

        if not torch.cuda.is_available() and device == 'cuda':
            logger.warning('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Register params
        self.params['n_classes'] = self.n_classes
        self.params['k1_neighbors'] = self.k1_neighbors
        self.params['sample_size1'] = self.sample_size1
        self.params['k2_neighbors'] = self.k2_neighbors
        self.params['sample_size2'] = self.sample_size2
        self.params['device'] = self.device

        # Initialize all 3 stages:
        self.applicability = ApplicabilityStage(
            model,
            data_name,
            n_classes,
            device,
        )
        self.reliability = ReliabilityStage(
            model,
            data_name,
            n_classes,
            k_neighbors=self.k1_neighbors,
            sample_size=self.sample_size1,
            device=self.device,
        )
        self.decidability = DecidabilityStage(
            model,
            data_name,
            n_classes,
            k_neighbors=self.k2_neighbors,
            sample_size=self.sample_size2,
            device=self.device
        )

    def train(self, X: Tensor = None, y: Tensor = None):
        self.applicability.train(X, y)
        self.reliability.train(X, y)
        self.decidability.train(X, y)

    def extract_features(self, X: Tensor) -> ArrayLike:
        scores = []
        scores_app = self.applicability.extract_features(X)
        scores_rel = self.reliability.extract_features(X)
        scores_dec = self.decidability.extract_features(X)
        scores.append(scores_app)
        scores.append(scores_rel)
        scores.append(scores_dec)
        scores = np.stack(scores).transpose()  # 3 features per example.
        return scores

    def save(self, path: str = None):
        """Save pre-trained features. The ideal extension is `.skbaard`."""
        path = create_parent_dir(path, file_ext='.skbaard')
        filename, _ = os.path.splitext(path)  # Get filename without extension.

        obj_app = self.applicability.save(filename + '.skbaard1')
        obj_rel = self.reliability.save(filename + '.skbaard2')
        obj_dec = self.decidability.save(filename + '.skbaard3')

        obj_baard = {
            'app': obj_app,
            'rel': obj_rel,
            'dec': obj_dec,
        }
        pickle.dump(obj_baard, open(path, 'wb'))

    def load(self, path: str = None):
        """Load pre-trained parameters. The default extension is `.skbaard`."""
        if os.path.isfile(path):
            obj = pickle.load(open(path, 'rb'))
            obj_app = obj['app']
            obj_rel = obj['rel']
            obj_dec = obj['dec']

            # Load parameters for ApplicabilityStage.
            self.applicability.zstats_dict = obj_app['zstats_dict']

            # Load parameters for Reliability.
            self.reliability.features_train = obj_rel['features_train']
            self.reliability.features_labels = obj_rel['features_labels']
            self.reliability.n_training_samples = obj_rel['n_training_samples']

            # Load parameters for DecidabilityStage.
            self.decidability.features_train = obj_dec['features_train']
            self.decidability.features_labels = obj_dec['features_labels']
            self.decidability.n_training_samples = obj_dec['n_training_samples']
            self.decidability.probs_correct = obj_dec['probs_correct']
        else:
            raise FileExistsError(f'{path} does not exist!')
