"""Extract and save features from adversarial detectors."""
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List

import torch

from baard.attacks import ATTACKS, L_NORM
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
from baard.utils.miscellaneous import (create_parent_dir,
                                       find_available_attacks, norm_parser)
from baard.utils.torch_utils import dataset2tensor

# For Feature Squeezing
FS_MAX_EPOCHS = 30  # Max. number of epochs used by the detectors.
# For LID. Used in LID original paper
LID_BATCH_SIZE = 100
LID_K_NEIGHBORS = 20
# For Odds. Used in Odds original paper
ODDS_NOISE_LIST = ['n0.003', 'n0.005', 'n0.008', 'n0.01', 'n0.02', 'n0.03',
                   's0.003', 's0.005', 's0.008', 's0.01', 's0.02', 's0.03',
                   'u0.003', 'u0.005', 'u0.008', 'u0.01', 'u0.02', 'u0.03']
ODDS_N_SAMPLE = 1000
# For PNClassification
PN_MAX_EPOCHS = 30
# For Region-based classification
RC_RADIUS = 0.2  # TODO: This need tuning!
RC_N_SAMPLE = 1000
# For BAARD S2 - Reliability  # TODO: This need tuning!
B2_K_NEIGHBORS = 20
B2_SAMPLE_SCALE = 50  # Number of examples in the subset: 50 * 20 = 1000
# For BAARD S3 - Decidability
B3_K_NEIGHBORS = 20
B3_SAMPLE_SCALE = 50


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


def init_detector(detector_name: str, data_name: str, path_checkpoint: str, seed: int) -> tuple[Detector, str]:
    """Initialize a detector."""
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
        detector = MLLooDetector(model, data_name)
    elif detector_name == DETECTORS[3]:  # Odds
        detector = OddsAreOddDetector(model, data_name, noise_list=ODDS_NOISE_LIST, n_noise_samples=ODDS_N_SAMPLE)
    elif detector_name == DETECTORS[4]:  # PN
        detector = PNDetector(model, data_name, path_checkpoint, max_epochs=PN_MAX_EPOCHS, seed=seed)
    elif detector_name == DETECTORS[5]:  # RC
        detector = RegionBasedClassifier(model, data_name, radius=RC_RADIUS, n_noise_samples=RC_N_SAMPLE)
    elif detector_name == DETECTORS[6]:  # BAARD-S1 - Applicability
        detector = ApplicabilityStage(model, data_name)
    elif detector_name == DETECTORS[7]:  # BAARD-S2 - Reliability
        detector = ReliabilityStage(model, data_name, k_neighbors=B2_K_NEIGHBORS, subsample_scale=B2_SAMPLE_SCALE)
    elif detector_name == DETECTORS[8]:  # BAARD-S3 - Decidability
        detector = DecidabilityStage(model, data_name, k_neighbors=B3_K_NEIGHBORS, subsample_scale=B3_SAMPLE_SCALE)
    elif detector_name == DETECTORS[9]:  # BAARD Full
        detector = BAARD(model, data_name,
                         k1_neighbors=B2_K_NEIGHBORS, subsample_scale1=B2_SAMPLE_SCALE,
                         k2_neighbors=B3_K_NEIGHBORS, subsample_scale2=B3_SAMPLE_SCALE)
    else:
        raise NotImplementedError
    detector_ext = DETECTOR_EXTENSIONS[detector.__class__.__name__]
    return detector, detector_ext


def extract_features(detector: Detector, attack_name: str, data_name: str, l_norm: Any, adv_files: List,
                     att_eps_list: List, path_output: str):
    """Extract features from a dataset."""
    detector_name = detector.__class__.__name__
    l_norm = norm_parser(l_norm)
    path_record = os.path.join(path_output, detector_name, f'{detector_name}-{data_name}-{attack_name}-{l_norm}.csv')
    with open(path_record, 'a', encoding='UTF-8') as file:
        file.write(','.join(['attack', 'path']) + '\n')
        for eps, path_data in zip(att_eps_list, adv_files):
            file.write(','.join([str(eps), path_data]) + '\n')

            print(f'Running {detector_name} on {data_name} with eps={eps}')
            dataset = torch.load(path_data)
            X, _ = dataset2tensor(dataset)
            features = detector.extract_features(X)

            path_features = os.path.join(path_output, detector_name, f'{attack_name}-{l_norm}', f'{detector_name}-{data_name}-{attack_name}-{l_norm}-{eps}.pt')
            path_features = create_parent_dir(path_features, file_ext='.pt')
            print(f'Save features to: {path_features}')
            torch.save(features, path_features)

            print('#' * 80)


def main_pipeline():
    """Full pipeline for running a detector."""
    path_output, seed, data_name, attack_name, l_norm, adv_files, att_eps_list, detector_name = parse_arguments()

    # Initialize detector
    path_clf_checkpoint = get_pretrained_model_path(data_name)
    detector, detector_ext = init_detector(detector_name=detector_name, data_name=data_name,
                                           path_checkpoint=path_clf_checkpoint, seed=seed)
    print('DETECTOR:', detector_name)
    print('DETECTOR EXTENSION:', detector_ext)

    detector_name = detector.__class__.__name__
    path_json = create_parent_dir(os.path.join(path_output, detector_name, f'{detector_name}-{data_name}.json'), file_ext='.json')
    path_detector = os.path.join(path_output, detector_name, f'{detector_name}-{data_name}.{detector_ext}')

    no_json_params = not os.path.exists(path_json)
    detector_cant_save = detector_name in ['FeatureSqueezingDetector', 'PNDetector', 'RegionBasedClassifier']
    if no_json_params or detector_cant_save:
        # Save parameters
        detector.save_params(path_json)
        # Train
        detector.train()
        if detector_ext is not None:
            # FeatureSqueezingDetector, PNDetector, and RegionBasedClassifier can not save.
            detector.save(path_detector)
    else:
        print(f'Found pre-trained {detector_name}. Load from {path_detector}')
        detector.load(path_detector)

    extract_features(detector, attack_name, data_name, l_norm, adv_files, att_eps_list, path_output)


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/detectors_extract_features.py --s 1234 --data MNIST --attack APGD -l 2 --detector "BAARD-S2"
    """
    parser = ArgumentParser()
    # NOTE: seed, data, and detector should NOT have default value! Debug only.
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--data', choices=DATASETS, required=True)
    parser.add_argument('--detector', type=str, choices=DETECTORS, required=True)
    parser.add_argument('-a', '--attack', choices=ATTACKS, default='APGD')
    parser.add_argument('-l', '--lnorm', type=str, choices=L_NORM, default='2')
    parser.add_argument('-p', '--path', type=str, default='results',
                        help='The path for loading pre-trained adversarial examples, and saving results.')
    parser.add_argument('--eps', type=json.loads, default=None,
                        help="""A list of epsilon in JSON string format, e.g., "[0.06, 0.12]". If eps is not given,
                        search the directory for all epsilon.""")
    args = parser.parse_args()
    seed = args.seed
    data_name = args.data
    attack_name = args.attack
    l_norm = args.lnorm
    detector_name = args.detector
    path = args.path
    eps_list = args.eps

    path_attack = Path(os.path.join(path, f'exp{seed}', data_name)).absolute()
    adv_files, att_eps_list = find_available_attacks(path_attack, attack_name, l_norm, eps_list)

    print('PATH:', path_attack)
    print('ATTACK:', attack_name)
    print('L-NORM:', l_norm)
    print('EPSILON:', att_eps_list)

    return path_attack, seed, data_name, attack_name, l_norm, adv_files, att_eps_list, detector_name


if __name__ == '__main__':
    main_pipeline()
