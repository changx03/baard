"""
Implementing the algorithm of Blocking Adversarial Examples by Testing
Applicability, Reliability and Decidability.
"""
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baard.classifiers import DATASETS, MNIST_CNN
from baard.detections import Detector
from baard.utils.miscellaneous import create_parent_dir
from baard.utils.torch_utils import (batch_forward, dataloader2tensor,
                                     dataset2tensor, predict)


class ApplicabilityStage(Detector):
    """The 1st stage of BAARD framework. It check the applicability of given
    samples. How similar in the feature space between the example and the trining data.
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
            warnings.warn('GPU is not available. Using CPU...')
            device = 'cpu'
        self.device = device

        # Tunable parameters
        self.zstats_dict = None

    def train(self, X: Tensor = None, y: Tensor = None) -> None:
        if X is None or y is None:
            loader_train = self.model.train_dataloader()
            X, y = dataloader2tensor(loader_train)
            print(f'Using the entire training set. {X.size(0)} samples.')

        # Initialize parameters
        self.zstats_dict = {c: {'mean': 0, 'std': 0} for c in range(self.n_classes)}

        print('[Latent Net]\n', self.latent_net)

        hidden_features = batch_forward(self.latent_net,
                                        X=X,
                                        batch_size=self.batch_size,
                                        device=self.device,
                                        num_workers=self.num_workers,
                                        )

        pbar = tqdm(range(self.n_classes), total=self.n_classes)
        pbar.set_description('Training applicability', refresh=False)
        for c in pbar:
            # Get the subset that are labelled as `c`.
            indices_with_label_c = torch.where(y == c)[0]
            features_subset = hidden_features[indices_with_label_c]
            # all examples in the subset have the same label, c.
            # Compute statistics on hidden features
            latent_feature_mean = features_subset.mean(0)
            latent_feature_std = features_subset.std(0)
            self.zstats_dict[c]['mean'] = latent_feature_mean
            self.zstats_dict[c]['std'] = latent_feature_std

    def extract_features(self, X: Tensor) -> ArrayLike:
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
        for c in range(self.n_classes):
            indices_with_label_c = torch.where(preds == c)[0]
            n_subset = len(indices_with_label_c)
            if n_subset == 0:
                logging.info('No example is in Class [%i].', c)
                continue
            features_subset = hidden_features[indices_with_label_c]
            latent_feature_mean = self.zstats_dict[c]['mean']
            latent_feature_std = self.zstats_dict[c]['std']
            z_score = (features_subset - latent_feature_mean) / latent_feature_std
            z_max, _ = torch.abs(z_score).max(dim=1)  # Return (max, indices_max)
            scores[indices_with_label_c] = z_max
        return scores.detach().numpy()

    def save(self, path: str = None) -> None:
        """Save pre-trained statistics. The ideal extension is `.baard1`."""
        create_parent_dir(path, file_ext='.baard1')

        save_obj = {
            'zstats_dict': self.zstats_dict
        }
        pickle.dump(save_obj, open(path, 'wb'))

    def load(self, path: str = None) -> None:
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
            latent_net = torch.nn.Sequential(*list(model.children())[:7])
        elif data_name == DATASETS[1]:  # CIFAR10
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return latent_net


def run_demo():
    """Run a BAARD detector demo."""
    DATASET = 'MNIST'
    EPS = 0.22  # Epsilon controls the adversarial perturbation.

    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', DATASET, 'AdvClean.n_100.pt')
    PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', DATASET, f'APGD.Linf.n_100.e_{EPS}.pt')
    PATH_DETECTOR_DEV = os.path.join('temp', 'dev_baard_applicability_{DATASET}.baard1')

    # Parameters for development:
    TINY_TEST_SIZE = 5
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
    detector.save(PATH_DETECTOR_DEV)
    detector.load(PATH_DETECTOR_DEV)

    # Load clean examples
    clean_dataset = torch.load(PATH_DATA_CLEAN)
    X_clean, y_true = dataset2tensor(clean_dataset)

    # Load adversarial examples
    adv_dataset = torch.load(PATH_DATA_ADV)
    X_adv, _ = dataset2tensor(adv_dataset)

    # Use tiny set
    X_clean = X_clean[:TINY_TEST_SIZE]
    X_adv = X_adv[:TINY_TEST_SIZE]

    # Extract features
    features_clean = detector.extract_features(X_clean)
    features_adv = detector.extract_features(X_adv)

    print('Clean:', np.round(features_clean, 2))
    print('  Adv:', np.round(features_adv, 2))


if __name__ == '__main__':
    run_demo()
