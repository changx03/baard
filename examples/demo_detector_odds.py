
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

from baard.classifiers import MNIST_CNN, CIFAR10_ResNet18
from baard.detections.odds_are_odd import OddsAreOddDetector
from baard.utils.torch_utils import dataset2tensor

PATH_ROOT = Path(os.getcwd()).absolute()
# Parameters for development:
SEED_DEV = 0
NOIST_LIST_DEV = ['n0.03', 'u0.03']
N_NOISE_DEV = 30
SIZE_DEV = 100
TINY_TEST_SIZE = 10


def run_odds_detector(data_name: str):
    """Test OddsAreOdd Detector"""
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

    path_val_data = os.path.join(PATH_ROOT, 'results', 'exp1234', data_name, 'ValClean-1000.pt')
    path_adv = os.path.join(PATH_ROOT, 'results', 'exp1234', data_name, f'APGD-Linf-100-{eps}.pt')
    path_odds_dev = os.path.join('temp', f'dev_OddsAreOddDetector_{data_name}.odds')

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', data_name)
    print('PATH_CHECKPOINT:', path_checkpoint)

    val_dataset = torch.load(path_val_data)
    X_val, y_val = dataset2tensor(val_dataset)
    # Limit the size for quick development. Using stratified sampling to ensure class distribution.
    _, X_dev, _, y_dev = train_test_split(X_val, y_val, test_size=SIZE_DEV, random_state=SEED_DEV)

    # Train detector
    ############################################################################
    # Uncomment the block below to train the detector:

    detector = OddsAreOddDetector(my_model,
                                  data_name,
                                  noise_list=NOIST_LIST_DEV,
                                  n_noise_samples=N_NOISE_DEV)
    detector.train(X_dev, y_dev)
    detector.save(path_odds_dev)
    del detector
    ############################################################################

    # Evaluate detector
    detector2 = OddsAreOddDetector(my_model,
                                   data_name,
                                   noise_list=NOIST_LIST_DEV,
                                   n_noise_samples=N_NOISE_DEV)
    detector2.load(path_odds_dev)

    scores = detector2.extract_features(X_dev[:TINY_TEST_SIZE])
    print(scores)

    # Load adversarial examples
    adv_dataset = torch.load(path_adv)
    X_adv, _ = dataset2tensor(adv_dataset)
    scores_adv = detector2.extract_features(X_adv[:TINY_TEST_SIZE])
    print(scores_adv)


if __name__ == '__main__':
    run_odds_detector('MNIST')
    run_odds_detector('CIFAR10')
