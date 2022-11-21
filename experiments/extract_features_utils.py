"""Utility functions for extracting features."""
import os
import warnings
from typing import Any, List

import torch

from baard.classifiers import DATASETS, MNIST_CNN, CIFAR10_ResNet18
from baard.detections import DETECTOR_EXTENSIONS, DETECTORS, Detector
from baard.detections.baard_detector import (BAARD, ApplicabilityStage,
                                             DecidabilityStage,
                                             ReliabilityStage)
from baard.detections.feature_squeezing import FeatureSqueezingDetector
from baard.detections.lid import LIDDetector
from baard.detections.ml_loo import MLLooDetector
from baard.detections.odds_are_odd import OddsAreOddDetector
from baard.detections.pn_detector import PNDetector
from baard.detections.region_based_classifier import RegionBasedClassifier
from baard.utils.miscellaneous import create_parent_dir, norm_parser
from baard.utils.torch_utils import dataset2tensor, find_last_checkpoint

PATH_LOGS = 'logs'
# For Feature Squeezing
FS_MAX_EPOCHS = 30  # Max. number of epochs used by the detectors.
# For LID. Used in LID original paper
LID_BATCH_SIZE = 100
LID_K_NEIGHBORS = 20
# For Odds.
# ODDS_NOISE_LIST = ['n0.003', 'n0.005', 'n0.008', 'n0.01', 'n0.02', 'n0.03',
#                    's0.003', 's0.005', 's0.008', 's0.01', 's0.02', 's0.03',
#                    'u0.003', 'u0.005', 'u0.008', 'u0.01', 'u0.02', 'u0.03']
ODDS_NOISE_LIST = ['n0.003', 's0.003', 'u0.003']  # Used in the original paper's repo
# ODDS_N_SAMPLE = 1000
ODDS_N_SAMPLE = 100  # Used in the original paper's repo
# For PNClassification
PN_MAX_EPOCHS = 30
# For Region-based classification
RC_N_SAMPLE = 1000
# For BAARD S2 - Reliability  # TODO: This need tuning!
B2_K_NEIGHBORS = 5
B2_SAMPLE_SIZE = 1000  # Number of examples in the subset: 500 * 5 = 2500
# For BAARD S3 - Decidability
B3_K_NEIGHBORS = 100
B3_SAMPLE_SIZE = 5000  # Number of examples in the subset: 50 * 100 = 5000


def get_pretrained_model_path(data_name: str) -> str:
    """Return the path of the pre-trained model based on the name of the dataset."""
    path = None
    if data_name == DATASETS[0]:  # MNIST
        path = os.path.join('pretrained_clf', 'mnist_cnn.ckpt')
    elif data_name == DATASETS[1]:  # CIFAR10
        path = os.path.join('pretrained_clf', 'cifar10_resnet18.ckpt')
    else:
        raise NotImplementedError
    return path


def get_fs_filter_names(data_name: str) -> List:
    """Return a list of filter names for Feature Squeezing."""
    filter_list = []
    if data_name == DATASETS[0]:  # MNIST
        filter_list = ['depth', 'median']
    elif data_name == DATASETS[1]:  # CIFAR10
        filter_list = ['depth', 'median', 'nl_mean']
    else:
        raise NotImplementedError
    return filter_list


def init_detector(detector_name: str, data_name: str, path_checkpoint: str, seed: int) -> tuple[Detector, str]:
    """Initialize a detector."""
    # The attack perturbation epsilon is used by LID and ML-LOO.
    if data_name == DATASETS[0]:  # MNIST
        model = MNIST_CNN.load_from_checkpoint(path_checkpoint)
        eps = 0.66  # For APGD on L-inf
    elif data_name == DATASETS[1]:  # CIFAR10
        model = CIFAR10_ResNet18.load_from_checkpoint(path_checkpoint)
        eps = 0.1  # APGD on L-inf
    else:
        raise NotImplementedError

    detector = None
    detector_ext = None
    if detector_name == DETECTORS[0]:  # FS
        detector = FeatureSqueezingDetector(model, data_name, path_checkpoint, max_epochs=FS_MAX_EPOCHS, seed=seed)
    elif detector_name == DETECTORS[1]:  # LID
        # Default attack is APGD on L-inf
        detector = LIDDetector(model, data_name, attack_eps=eps, noise_eps=eps, batch_size=LID_BATCH_SIZE,
                               k_neighbors=LID_K_NEIGHBORS)
    elif detector_name == DETECTORS[2]:  # ML-LOO
        detector = MLLooDetector(model, data_name, attack_eps=eps)
    elif detector_name == DETECTORS[3]:  # Odds
        detector = OddsAreOddDetector(model, data_name, noise_list=ODDS_NOISE_LIST, n_noise_samples=ODDS_N_SAMPLE)
    elif detector_name == DETECTORS[4]:  # PN
        detector = PNDetector(model, data_name, path_checkpoint, max_epochs=PN_MAX_EPOCHS, seed=seed)
    elif detector_name == DETECTORS[5]:  # RC
        if data_name == 'MNIST':
            RC_RADIUS = 0.3  # from original paper
        elif data_name == 'CIFAR10':
            RC_RADIUS = 0.02  # from original paper
        else:
            raise NotImplementedError
        detector = RegionBasedClassifier(model, data_name, radius=RC_RADIUS, n_noise_samples=RC_N_SAMPLE)
    elif detector_name == DETECTORS[6]:  # BAARD-S1 - Applicability
        detector = ApplicabilityStage(model, data_name)
    elif detector_name == DETECTORS[7]:  # BAARD-S2 - Reliability
        detector = ReliabilityStage(model, data_name, k_neighbors=B2_K_NEIGHBORS, sample_size=B2_SAMPLE_SIZE)
    elif detector_name == DETECTORS[8]:  # BAARD-S3 - Decidability
        detector = DecidabilityStage(model, data_name, k_neighbors=B3_K_NEIGHBORS, sample_size=B3_SAMPLE_SIZE)
    elif detector_name == DETECTORS[9]:  # BAARD Full
        detector = BAARD(model, data_name,
                         k1_neighbors=B2_K_NEIGHBORS, sample_size1=B2_SAMPLE_SIZE,
                         k2_neighbors=B3_K_NEIGHBORS, sample_size2=B3_SAMPLE_SIZE)
    else:
        raise NotImplementedError
    detector_ext = DETECTOR_EXTENSIONS[detector.__class__.__name__]
    return detector, detector_ext


def train_or_load_detector(detector: Detector,
                           detector_name: str,
                           detector_ext: str,
                           data_name: str,
                           path: str,
                           use_validation_set: bool = False,
                           ) -> Detector:
    """Train or load a detector based on previously saved file.
    NOTE: Not all detectors need a validation set. Some use the entire training set.
    """
    path_detector = os.path.join(path, detector_name, f'{detector_name}-{data_name}{detector_ext}')
    if not os.path.exists(path_detector):
        if use_validation_set:  # Load validation set
            path_validation_set = os.path.join(path, 'ValClean-1000.pt')  # Validation set has fixed name.
            dataset_val = torch.load(path_validation_set)
            X_val, y_val = dataset2tensor(dataset_val)
            detector.train(X_val, y_val)
        else:
            detector.train()
        detector.save(path_detector)
    else:
        print(f'Found pre-trained {detector_name}. Load from {path_detector}')
        detector.load(path_detector)
    return detector


def train_or_load_FS(detector: FeatureSqueezingDetector, data_name: str,
                     path_checkpoints: str = PATH_LOGS) -> FeatureSqueezingDetector:
    """Feature Squeezing need to find multiple checkpoints."""
    squeezers = get_fs_filter_names(data_name)
    path_squeezer_dict = {}
    for squeezer in squeezers:
        path_squeezer_checkpoint = find_last_checkpoint(
            'FeatureSqueezer', data_name, kernel_name=squeezer, path=path_checkpoints)
        if path_squeezer_checkpoint is None:
            warnings.warn(f'[FS] Checkpoint does not number of squeezer! Cannot find {squeezer}.')
            break
        path_squeezer_dict[squeezer] = path_squeezer_checkpoint
    if len(path_squeezer_dict) != len(squeezers):  # Train
        detector.train()
    else:
        print(f'Found pre-trained FeatureSqueezingDetector. Load from {path_checkpoints}')
        detector.load(path_squeezer_dict)
    return detector


def train_or_load_PN(detector: PNDetector, data_name: str,
                     path_checkpoints: str = PATH_LOGS) -> FeatureSqueezingDetector:
    """Positive-Negative Classification need to find single checkpoint."""
    path_checkpoint = find_last_checkpoint('PNClassifier', data_name, kernel_name=None, path=path_checkpoints)
    if path_checkpoint is None:
        detector.train()
    else:
        print(f'Found pre-trained PNClassifier. Load from {path_checkpoint}')
        detector.load(path_checkpoint)
    return detector


def prepare_detector(detector: Detector, detector_name: str, detector_ext: str, data_name: str, path: str) -> Detector:
    """Train or load a detector."""
    if detector_name in ['ApplicabilityStage', 'ReliabilityStage', 'DecidabilityStage', 'BAARD']:
        # These detectors use entire training set.
        detector = train_or_load_detector(detector, detector_name, detector_ext,
                                          data_name=data_name, path=path, use_validation_set=False)
    elif detector_name in ['LIDDetector', 'MLLooDetector', 'OddsAreOddDetector']:
        # These detectors use a validation set.
        detector = train_or_load_detector(detector, detector_name, detector_ext,
                                          data_name=data_name, path=path, use_validation_set=True)
    elif detector_name == 'FeatureSqueezingDetector':  # Require to load a list of checkpoints
        detector = train_or_load_FS(detector, data_name)
    elif detector_name == 'PNDetector':  # Require to load a single checkpoint
        detector = train_or_load_PN(detector, data_name)
    elif detector_name == 'RegionBasedClassifier':  # Don't need to train. Do nothing!
        pass
    else:
        raise NotImplementedError
    return detector


def extract_and_save_features(detector: Detector, attack_name: str, data_name: str, l_norm: Any, adv_files: List,
                              att_eps_list: List, path_output: str):
    """Extract features from a dataset."""
    detector_name = detector.__class__.__name__
    l_norm = norm_parser(l_norm)
    path_record = os.path.join(path_output, detector_name, f'{detector_name}-{data_name}-{attack_name}-{l_norm}.csv')
    with open(path_record, 'a', encoding='UTF-8') as file:
        file.write(','.join(['attack', 'path']) + '\n')
        for eps, path_data in zip(att_eps_list, adv_files):
            path_features = os.path.join(path_output, detector_name, f'{attack_name}-{l_norm}',
                                         f'{detector_name}-{data_name}-{attack_name}-{l_norm}-{eps}.pt')
            if not os.path.exists(path_features):
                file.write(','.join([str(eps), path_data]) + '\n')

                print(f'Running {detector_name} on {data_name} with eps={eps}')
                dataset = torch.load(path_data)
                X, _ = dataset2tensor(dataset)
                features = detector.extract_features(X)
                path_features = create_parent_dir(path_features, file_ext='.pt')
                print(f'Save features to: {path_features}')
                torch.save(features, path_features)
            else:
                print(f'Found {path_features} Skip!')
            print('#' * 80)

            if detector_name == 'MLLooDetector' or detector_name == 'LIDDetector':
                extract_proba(detector, detector_name, path_output, data_name, attack_name, l_norm, eps)


def extract_proba(detector: MLLooDetector, detector_name, path_output, data_name, attack_name, l_norm, eps):
    """ML-LOO and LID output a vector that cannot be fit with simple logistic regression model. Compute probabilities
    instead.
    """
    path_features = os.path.join(path_output, detector_name, f'{attack_name}-{l_norm}',
                                 f'{detector_name}-{data_name}-{attack_name}-{l_norm}-{eps}.pt')
    path_probas = os.path.join(path_output, detector_name, f'{attack_name}-{l_norm}',
                               f'{detector_name}(proba)-{data_name}-{attack_name}-{l_norm}-{eps}.pt')
    if not os.path.exists(path_probas):
        features = torch.load(path_features)
        if detector_name == 'MLLooDetector':  # Only for ML-LOO
            features = detector.scaler.transform(features)
        probs = detector.logistic_regressor.predict_proba(features)
        probs = probs[:, 1]  # Only return the 2nd column
        print(f'Save {detector_name} probabilities to: {path_probas}')
        torch.save(probs, path_probas)
    else:
        print(f'Found {path_probas} Skip!')
