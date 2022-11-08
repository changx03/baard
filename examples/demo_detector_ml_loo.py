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
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD \
    --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.22]" --n_val=1000

"""
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from baard.classifiers import MNIST_CNN, CIFAR10_ResNet18
from baard.detections.ml_loo import MLLooDetector
from baard.utils.torch_utils import dataset2tensor

PATH_ROOT = Path(os.getcwd()).absolute()

# Parameters for development:
SEED_DEV = 123456
SIZE_DEV = 40  # For quick development
TINY_TEST_SIZE = 10


def run_mlloo(data_name: str):
    """Test ML-LOO Detector."""
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
    path_data_clean = os.path.join(PATH_ROOT, 'results', 'exp1234', data_name, 'AdvClean-100.pt')
    path_data_adv = os.path.join(PATH_ROOT, 'results', 'exp1234', data_name, f'APGD-Linf-100-{eps}.pt')
    path_mlloo_dev = os.path.join('temp', f'dev_MLLooDetector_{data_name}.mlloo')

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', data_name)
    print('PATH_CHECKPOINT:', path_checkpoint)

    # Clean examples
    dataset_clean = torch.load(path_data_clean)
    X_clean, y_clean = dataset2tensor(dataset_clean)

    # Corresponding adversarial examples
    dataset_adv = torch.load(path_data_adv)
    X_adv, y_adv_true = dataset2tensor(dataset_adv)

    assert torch.all(y_clean == y_adv_true), 'True labels should be the same!'

    indices_train, _, indices_eval, _ = train_test_split(
        np.arange(X_clean.size(0)),
        y_clean,
        train_size=SIZE_DEV,
        random_state=SEED_DEV,
    )  # Get a stratified set

    # Tiny train set
    X_train_clean = X_clean[indices_train]
    X_train_adv = X_adv[indices_train]
    y_train_true = y_clean[indices_train]

    # Tiny test set
    X_eval_clean = X_clean[indices_eval][:TINY_TEST_SIZE]
    X_eval_adv = X_adv[indices_eval][:TINY_TEST_SIZE]
    # y_eval_true = y_clean[indices_eval][:TINY_TEST_SIZE]

    print('Pre-trained ML-LOO path:', path_mlloo_dev)

    # Save results
    ############################################################################
    # Uncomment the block below to train the detector:

    detector = MLLooDetector(my_model, data_name)
    detector.train(X_train_clean, y_train_true)
    detector.save(path_mlloo_dev)
    del detector
    ############################################################################

    # Load detector
    detector2 = MLLooDetector(my_model, data_name)
    detector2.load(path_mlloo_dev)

    # Making prediction
    score_clean = detector2.predict_proba(X_eval_clean)
    print(score_clean)

    score_adv = detector2.predict_proba(X_eval_adv)
    print(score_adv)


if __name__ == '__main__':
    # run_mlloo('MNIST')
    run_mlloo('CIFAR10')
