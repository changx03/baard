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

from baard.classifiers import DATASETS
from baard.classifiers.mnist_cnn import MNIST_CNN
from baard.detections.ml_loo import MLLooDetector
from baard.utils.torch_utils import dataset2tensor

# Parameters for development:
SEED_DEV = 0
PATH_ROOT = Path(os.getcwd()).absolute()
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'AdvClean.n_100.pt')
PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'APGD.Linf.n_100.e_0.22.pt')
PATH_MLLOO_DEV = os.path.join('temp', 'dev_mlloo_detector.mlloo')
DATASET = DATASETS[0]
SIZE_DEV = 40  # For quick development


if __name__ == '__main__':
    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)

    BATCH_SIZE = model.train_dataloader().batch_size
    DEVICE = model.device
    NUM_WORKERS = model.train_dataloader().num_workers
    INPUT_SHAPE = (BATCH_SIZE, 1, 28, 28)

    # Clean examples
    dataset_clean = torch.load(PATH_DATA_CLEAN)
    X_clean, y_clean = dataset2tensor(dataset_clean)

    # Corresponding adversarial examples
    dataset_adv = torch.load(PATH_DATA_ADV)
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
    X_eval_clean = X_clean[indices_eval][:10]
    X_eval_adv = X_adv[indices_eval][:10]
    y_eval_true = y_clean[indices_eval][:10]

    print('Pre-trained ML-LOO path:', PATH_MLLOO_DEV)

    # Save results
    ############################################################################
    # Uncomment the block below to train the detector:

    detector = MLLooDetector(model, DATASET)
    detector.train(X_train_clean, y_train_true, X_train_adv)
    detector.save(PATH_MLLOO_DEV)
    ############################################################################

    # Load detector
    detector2 = MLLooDetector(model, DATASET)
    detector2.load(PATH_MLLOO_DEV)

    # Making prediction
    score_clean = detector2.predict_proba(X_eval_clean)
    print(score_clean)

    score_adv = detector2.predict_proba(X_eval_adv)
    print(score_adv)
