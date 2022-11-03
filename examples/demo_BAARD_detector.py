"""Basic demo for Feature Squeezing detector.

Dataset: MNIST
Required files:
1. Pre-trained classifier: `./pretrained_clf/mnist_cnn.ckpt`
2. Generated adversarial examples and their corresponding clean data:
    Clean data: `./results/exp1234/MNIST/AdvClean.n_100.pt`
    Adversarial example: `./results/exp1234/MNIST/APGD.Linf.n_100.e_0.22.pt`

To train the classifier, run:
python ./baard/classifiers/mnist_cnn.py --seed 1234

To generate adversarial examples for this demo:
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD \
    --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.22]" --n_val=1000

"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch

from baard.classifiers import DATASETS, MNIST_CNN, CIFAR10_ResNet18
from baard.detections.baard_detector import (BAARD, ApplicabilityStage,
                                             DecidabilityStage,
                                             ReliabilityStage)
from baard.utils.torch_utils import dataset2tensor

logging.basicConfig(level=logging.INFO)

PATH_ROOT = Path(os.getcwd()).absolute()

# Parameters for development:
TINY_TEST_SIZE = 5
SEED_DEV = 0
N_CLASSES = 10  # For both MNIST and CIFAR10


def run_baard(data_name: str,
              detector_class: Any,
              train: bool = False,
              params: Dict = None
              ) -> None:
    """Run BAARD demo."""
    if data_name == 'MNIST':
        eps = 0.22  # Epsilon controls the adversarial perturbation.
        path_checkpoint = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
        my_model = MNIST_CNN.load_from_checkpoint(path_checkpoint)
    elif data_name == 'CIFAR10':
        eps = 0.02  # Min L-inf attack eps for CIFAR10
        path_checkpoint = os.path.join(PATH_ROOT, 'pretrained_clf', 'cifar10_resnet18.ckpt')
        my_model = CIFAR10_ResNet18.load_from_checkpoint(path_checkpoint)
    else:
        raise NotImplementedError

    path_data_clean = os.path.join(PATH_ROOT, 'results', 'exp1234', data_name, 'AdvClean.n_100.pt')
    path_data_adv = os.path.join(PATH_ROOT, 'results', 'exp1234', data_name, f'APGD.Linf.n_100.e_{eps}.pt')

    if detector_class == ApplicabilityStage:
        file_ext = '.baard1'
    elif detector_class == ReliabilityStage:
        file_ext = '.baard2'
    elif detector_class == DecidabilityStage:
        file_ext = '.baard3'
    else:  # BAARD
        file_ext = '.baard'

    path_detector_dev = os.path.join('temp', f'dev_baard_applicability_{data_name}{file_ext}')

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', data_name)
    print('PATH_CHECKPOINT:', path_checkpoint)

    pl.seed_everything(SEED_DEV)

    if params is None:
        params = {}
    params['data_name'] = data_name
    params['model'] = my_model
    params['n_classes'] = N_CLASSES
    # print('PARAMS:', params)

    # Train detector
    ############################################################################
    if train:
        detector = detector_class(**params)
        detector.train()
        detector.save(path_detector_dev)
        del detector
    ############################################################################

    # Load clean examples
    clean_dataset = torch.load(path_data_clean)
    X_clean, y_true = dataset2tensor(clean_dataset)

    # Load adversarial examples
    adv_dataset = torch.load(path_data_adv)
    X_adv, _ = dataset2tensor(adv_dataset)

    # Use tiny set
    X_clean = X_clean[:TINY_TEST_SIZE]
    X_adv = X_adv[:TINY_TEST_SIZE]

    # Load results
    detector2 = detector_class(**params)
    detector2.load(path_detector_dev)

    # Extract features
    features_clean = detector2.extract_features(X_clean)
    features_adv = detector2.extract_features(X_adv)

    print('Clean:', np.round(features_clean, decimals=2))
    print('  Adv:', np.round(features_adv, decimals=2))


if __name__ == '__main__':
    print('Running BAARD demo...')

    # For MNIST:

    # run_baard('MNIST', ApplicabilityStage, train=True)
    # run_baard('MNIST', ReliabilityStage, train=True, params={'k_neighbors': 20, 'subsample_scale': 50})
    # run_baard('MNIST', DecidabilityStage, train=True, params={'k_neighbors': 20, 'subsample_scale': 50})

    # params_baard = {
    #     'k1_neighbors': 20,
    #     'subsample_scale1': 10,
    #     'k2_neighbors': 20,
    #     'subsample_scale2': 10,
    # }
    # run_baard('MNIST', BAARD, train=True, params=params_baard)

    # ###########################################################################
    # # For CIFAR10:

    # run_baard('CIFAR10', ApplicabilityStage)
    # run_baard('CIFAR10', ReliabilityStage, train=True, params={'k_neighbors': 20, 'subsample_scale': 50})
    # run_baard('CIFAR10', DecidabilityStage, train=True, params={'k_neighbors': 20, 'subsample_scale': 50})

    params_baard = {
        'k1_neighbors': 20,
        'subsample_scale1': 10,
        'k2_neighbors': 20,
        'subsample_scale2': 10,
    }
    run_baard('CIFAR10', BAARD, train=True, params=params_baard)
