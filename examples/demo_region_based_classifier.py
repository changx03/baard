"""Basic demo for Region-based Classifier.

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
from torch.utils.data import DataLoader, TensorDataset

from baard.classifiers import DATASETS, MNIST_CNN
from baard.detections.region_based_classifier import RegionBasedClassifier
from baard.utils.torch_utils import dataset2tensor, predict


def run_demo():
    """Test Feature Squeezing Detector."""
    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'AdvClean.n_100.pt')
    PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'FGSM.Linf.n_100.e_0.28.pt')

    # Parameters for development:
    SEED_DEV = 0
    DATASET = DATASETS[0]
    TINY_TEST_SIZE = 10

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    my_model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)

    detector = RegionBasedClassifier(
        my_model,
        DATASET,  # MNIST,
        radius=0.3,
        n_noise_samples=100
    )

    # Clean examples
    dataset_clean = torch.load(PATH_DATA_CLEAN)
    X_clean, y_clean = dataset2tensor(dataset_clean)

    # Corresponding adversarial examples
    dataset_adv = torch.load(PATH_DATA_ADV)
    X_adv, y_adv_true = dataset2tensor(dataset_adv)

    # Tiny test set
    X_eval_clean = X_clean[:TINY_TEST_SIZE]
    X_eval_adv = X_adv[:TINY_TEST_SIZE]
    y_eval_true = y_clean[:TINY_TEST_SIZE]

    features_clean = detector.extract_features(X_eval_clean)
    print('Clean:')
    print(np.round(features_clean, 2))

    features_adv = detector.extract_features(X_eval_adv)
    print('Adversarial examples:')
    print(np.round(features_adv, 2))

    preds_clean = detector.predict(X_eval_clean)
    preds_adv = detector.predict(X_eval_adv)
    dataloader = DataLoader(TensorDataset(X_eval_adv),
                            batch_size=detector.batch_size,
                            num_workers=detector.num_workers,
                            shuffle=False)
    preds_adv_origin = predict(my_model, dataloader)
    print(' True:', y_eval_true)
    print('Clean [RC]:', preds_clean)
    print('  Adv [OG]:', preds_adv_origin)
    print('  Adv [RC]:', preds_adv)


if __name__ == '__main__':
    run_demo()
