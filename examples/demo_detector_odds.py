
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

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from baard.classifiers import DATASETS
from baard.classifiers.mnist_cnn import MNIST_CNN
from baard.detections.odds_are_odd import OddsAreOddDetector
from baard.utils.torch_utils import dataset2tensor


def run_demo():
    """Test OddsAreOdd Detector"""
    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    PATH_VAL_DATA = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'ValClean.n_1000.pt')
    PATH_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'APGD.Linf.n_100.e_0.22.pt')
    PATH_WEIGHTS_DEV = os.path.join('temp', 'dev_odds_detector.odds')

    # Parameters for development:
    SEED_DEV = 0
    NOIST_LIST_DEV = ['n0.01', 'u0.01']
    N_NOISE_DEV = 30
    SIZE_DEV = 100
    DATASET = DATASETS[0]

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)

    detector = OddsAreOddDetector(model,
                                  DATASETS[0],
                                  noise_list=NOIST_LIST_DEV,
                                  n_noise_samples=N_NOISE_DEV)

    val_dataset = torch.load(PATH_VAL_DATA)
    X_val, y_val = dataset2tensor(val_dataset)
    # Limit the size for quick development. Using stratified sampling to ensure class distribution.
    _, X_dev, _, y_dev = train_test_split(X_val, y_val, test_size=SIZE_DEV, random_state=SEED_DEV)

    # Train detector
    ############################################################################
    # Uncomment the block below to train the detector:

    # detector.train(X_dev, y_dev)
    # detector.save(PATH_WEIGHTS_DEV)
    ############################################################################

    # Evaluate detector
    detector2 = OddsAreOddDetector(model,
                                   DATASETS[0],
                                   noise_list=NOIST_LIST_DEV,
                                   n_noise_samples=N_NOISE_DEV)
    detector2.load(PATH_WEIGHTS_DEV)

    scores = detector2.extract_features(X_dev[:30])
    print(scores)

    # Load adversarial examples
    adv_dataset = torch.load(PATH_ADV)
    X_adv, y_adv_true = dataset2tensor(adv_dataset)
    scores_adv = detector2.extract_features(X_adv[:30])
    print(scores_adv)


if __name__ == '__main__':
    run_demo()
