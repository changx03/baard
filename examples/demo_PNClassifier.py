"""Basic demo for PNClassifier detector.

Dataset: MNIST
Required files:
1. Pre-trained classifier: `./pretrained_clf/mnist_cnn.ckpt`
2. Generated adversarial examples and their corresponding clean data:
    Clean data: `./results/exp1234/MNIST/AdvClean.n_100.pt`
    Adversarial example: `./results/exp1234/MNIST/APGD.Linf.n_100.e_0.22.pt`

To train the classifier, run:
python ./baard/classifiers/mnist_cnn.py --seed 1234

To generate adversarial examples for this demo:
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM  \
    --params='{"norm":"inf"}' --eps="[0.28]" --n_val=1000

"""
import os
from pathlib import Path

import pytorch_lightning as pl
import torch

from baard.classifiers import DATASETS
from baard.classifiers.mnist_cnn import MNIST_CNN
from baard.detections.pn_detector import PNDetector
from baard.utils.torch_utils import dataset2tensor

# Parameters for development:
SEED_DEV = 0
DATASET = DATASETS[0]
MAX_EPOCHS_DEV = 30
PATH_ROOT = Path(os.getcwd()).absolute()
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'AdvClean.n_100.pt')
PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'FGSM.Linf.n_100.e_0.28.pt')

if __name__ == '__main__':
    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)

    # Train PN Classifier
    ############################################################################
    # Uncomment the block below to train the detector:

    detector = PNDetector(model, DATASET, path_model=PATH_CHECKPOINT, max_epochs=MAX_EPOCHS_DEV, seed=SEED_DEV)
    detector.train()
    ############################################################################

    # Clean examples
    dataset_clean = torch.load(PATH_DATA_CLEAN)
    X_clean, y_clean = dataset2tensor(dataset_clean)

    # Corresponding adversarial examples
    dataset_adv = torch.load(PATH_DATA_ADV)
    X_adv, y_adv_true = dataset2tensor(dataset_adv)

    # Tiny test set
    X_eval_clean = X_clean[:20]
    X_eval_adv = X_adv[:20]
    y_eval_true = y_clean[:20]

    scores_clean = detector.extract_features(X_eval_clean)
    print(scores_clean)

    scores_adv = detector.extract_features(X_eval_adv)
    print(scores_adv)

    # Load results
    ############################################################################
    # PATH_PN_CLASSIFIER_DEV = os.path.join(PATH_ROOT, 'logs', 'PNClassifier_MNIST', 'version_0', 'checkpoints', 'epoch=29-step=7050.ckpt')
    # detector2 = PNDetector(model, DATASET, path_model=PATH_CHECKPOINT)
    # detector2.load(PATH_PN_CLASSIFIER_DEV)

    # scores_clean = detector2.extract_features(X_eval_clean)
    # print(scores_clean)

    # scores_adv = detector2.extract_features(X_eval_adv)
    # print(scores_adv)
    ############################################################################