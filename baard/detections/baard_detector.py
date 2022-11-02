"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.
"""
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from baard.classifiers import MNIST_CNN
from baard.utils.torch_utils import dataloader2tensor, dataset2tensor
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor
from tqdm import tqdm

from baard.detections import Detector


class ApplicabilityStage(Detector):
    """The 1st stage of BAARD framework. It check the applicability of given
    samples. How similar in the feature space between the example and the trining data.
    """

    def __init__(self,
                 model: LightningModule,
                 data_name: str,
                 n_classes: int = 10) -> None:
        super().__init__(model, data_name)

        self.n_classes = n_classes

        # Tunable parameters
        self.zstats_dict = None

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        if X is None or y is None:
            loader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(loader_train)
            print(f'Using the entire training set. {X.size(0)} samples.')

        # Initialize parameters
        self.zstats_dict = {c: {'mean': 0, 'std': 0} for c in range(self.n_classes)}

    def extract_features(self, X: Tensor) -> ArrayLike:
        return X


def run_demo():
    """Run a BAARD detector demo."""
    DATASET = 'MNIST'
    EPS = 0.22  # Epsilon controls the adversarial perturbation.

    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', DATASET, 'AdvClean.n_100.pt')
    PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', DATASET, f'APGD.Linf.n_100.e_{EPS}.pt')
    PATH_DETECTOR_DEV = os.path.join('temp', 'dev_lid_detector.lid')

    # Parameters for development:
    TINY_TEST_SIZE = 10
    SEED_DEV = 0

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    my_model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)
    detector = ApplicabilityStage(
        my_model,
        DATASET,
        10,
    )
    detector.train()


if __name__ == '__main__':
    run_demo()
