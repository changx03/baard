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

import os
from pathlib import Path

import pytorch_lightning as pl
import torch

from baard.classifiers import DATASETS, MNIST_CNN
from baard.detections.feature_squeezing import FeatureSqueezingDetector
from baard.utils.torch_utils import dataset2tensor


def run_demo():
    """Test Feature Squeezing Detector."""
    PATH_ROOT = Path(os.getcwd()).absolute()
    PATH_CHECKPOINT = os.path.join(PATH_ROOT, 'pretrained_clf', 'mnist_cnn.ckpt')
    PATH_DATA_CLEAN = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'AdvClean-100.pt')
    PATH_DATA_ADV = os.path.join(PATH_ROOT, 'results', 'exp1234', 'MNIST', 'FGSM-Linf-100-0.28.pt')

    # Parameters for development:
    SEED_DEV = 1
    DATASET = DATASETS[0]
    MAX_EPOCHS_DEV = 5
    TINY_TEST_SIZE = 10

    pl.seed_everything(SEED_DEV)

    print('PATH ROOT:', PATH_ROOT)
    print('DATASET:', DATASET)
    print('PATH_CHECKPOINT:', PATH_CHECKPOINT)

    my_model = MNIST_CNN.load_from_checkpoint(PATH_CHECKPOINT)

    detector = FeatureSqueezingDetector(
        my_model,
        DATASET,  # MNIST,
        path_model=PATH_CHECKPOINT,
        max_epochs=MAX_EPOCHS_DEV
    )

    ############################################################################
    # Uncomment the block below to train the model
    # detector.train()
    # detector.save()  # Dummy function.
    ############################################################################

    # Load pre-trained models
    squeezer_keys = detector.get_squeezers().keys()
    path_squeezers = {}
    for key in squeezer_keys:
        path = os.path.join(PATH_ROOT, 'logs', f'FeatureSqueezer_MNIST_{key}', 'version_0', 'checkpoints', 'epoch=4-step=1175.ckpt')
        path_squeezers[key] = path
    detector.load(path_squeezers)

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
    print(features_clean)

    features_adv = detector.extract_features(X_eval_adv)
    print(features_adv)


if __name__ == '__main__':
    run_demo()
