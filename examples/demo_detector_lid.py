
"""Basic demo for ML-LOO detector.

Dataset: MNIST
Required files:
1. Pre-trained classifier: `./pretrained_clf/mnist_cnn.ckpt`
2. Generated adversarial examples and their corresponding clean data:
    Clean data: `./results/exp1234/MNIST/AdvClean.n_100.pt`
    Adversarial example: `./results/exp1234/MNIST/APGD.Linf.n_100.e_0.22.pt`

To train the classifier, run:
python ./baard/classifiers/mnist_cnn.py --seed 1234

To generate adversarial examples for this demo:
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD  \
    --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.22]" --n_val=1000

"""
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from baard.classifiers import MNIST_CNN
from baard.detections.lid import LIDDetector
from baard.utils.torch_utils import dataloader2tensor, dataset2tensor


def run_demo():
    """Test LIDDetector."""
    DATASET = 'MNIST'
    EPS = 0.22  # Epsilon controls the noise and adversarial perturbation.

    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', DATASET, 'AdvClean-100.pt')
    PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', DATASET, f'APGD-Linf-100-{EPS}.pt')
    PATH_DETECTOR_DEV = os.path.join('temp', 'dev_lid_detector.lid')

    # Parameters for development:
    TINY_TRAIN_SIZE = 100
    TINY_TEST_SIZE = 10
    SEED_DEV = 0

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    my_model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)
    detector = LIDDetector(
        my_model,
        DATASET,
        clip_range=(0., 1.),
        attack_eps=EPS,
        attack_norm=np.inf,
        noise_eps=EPS,
        batch_size=100,
        k_neighbors=20,
    )
    loader_train = my_model.train_dataloader()
    X_train, y_train = dataloader2tensor(loader_train)

    # Use a tiny training set
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=TINY_TRAIN_SIZE, random_state=SEED_DEV)

    # Save results
    ############################################################################
    # Uncomment the block below to train the detector:

    # detector.train(X_train, y_train)  # Don't need train to extract features.
    # detector.save(PATH_DETECTOR_DEV)
    ############################################################################

    detector.load(PATH_DETECTOR_DEV)

    # Clean examples
    dataset_clean = torch.load(PATH_DATA_CLEAN)
    X_clean, _ = dataset2tensor(dataset_clean)

    # Corresponding adversarial examples
    dataset_adv = torch.load(PATH_DATA_ADV)
    X_adv, _ = dataset2tensor(dataset_adv)

    # Tiny test set
    X_eval_clean = X_clean[:TINY_TEST_SIZE]
    X_eval_adv = X_adv[:TINY_TEST_SIZE]

    features_clean = detector.extract_features(X_eval_clean)
    features_adv = detector.extract_features(X_eval_adv)

    print('      [Clean] LID features:')
    print(np.round(features_clean, 3))
    print('[Adversarial] LID features:')
    print(np.round(features_adv, 3))

    # Train Logistic regression model
    ############################################################################
    regressor = LogisticRegressionCV()
    lid_train_X = np.concatenate([detector.lid_neg, detector.lid_pos]).astype(float)
    lid_train_y = np.concatenate([
        np.zeros(len(detector.lid_neg)),
        np.ones(len(detector.lid_pos)),
    ]).astype(int)
    assert len(lid_train_X) == TINY_TRAIN_SIZE * 3, f'Training set != clean + noisy + adversarial examples. Expect {TINY_TRAIN_SIZE * 3} got {len(lid_train_X)}.'
    regressor.fit(lid_train_X, lid_train_y)

    results_clean = regressor.predict_proba(features_clean)[:, 1]  # Only choose 2nd column (positive)
    results_adv = regressor.predict_proba(features_adv)[:, 1]

    print('Clean pred:', np.round(results_clean, 3))
    print('  Adv pred:', np.round(results_adv, 3))


if __name__ == '__main__':
    run_demo()
