"""Extract and save features from adversarial detectors."""
import json
import os
from argparse import ArgumentParser
from pathlib import Path

from numpy.typing import ArrayLike

from baard.attacks import ATTACKS, L_NORM
from baard.classifiers import DATASETS, MNIST_CNN, CIFAR10_ResNet18
from baard.detections import DETECTORS, Detector
from baard.detections.baard_detector import (BAARD, ApplicabilityStage,
                                             DecidabilityStage,
                                             ReliabilityStage)
from baard.detections.feature_squeezing import FeatureSqueezingDetector
from baard.detections.lid import LIDDetector
from baard.detections.ml_loo import MLLooDetector
from baard.detections.odds_are_odd import OddsAreOddDetector
from baard.detections.pn_detector import PNDetector
from baard.detections.region_based_classifier import RegionBasedClassifier
from baard.utils.miscellaneous import find_available_attacks

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


def init_detector(detector_name: str, dname: str, path_checkpoint, seed: int) -> Detector:
    """Initialize a detector."""
    if dname == DATASETS[0]:  # MNIST
        model = MNIST_CNN.load_from_checkpoint(path_checkpoint)
        eps = 0.66  # For APGD on L-inf
    elif dname == DATASETS[1]:  # CIFAR10
        model = CIFAR10_ResNet18.load_from_checkpoint(path_checkpoint)
        eps = 0.1  # APGD on L-inf
    else:
        raise NotImplementedError

    if detector_name == DETECTORS[0]:  # FS
        detector = FeatureSqueezingDetector(model, dname, path_checkpoint, max_epochs=FS_MAX_EPOCHS, seed=seed)
    elif detector_name == DETECTORS[1]:  # LID
        # Default attack is APGD on L-inf
        detector = LIDDetector(model, dname, attack_eps=eps, noise_eps=eps, batch_size=LID_BATCH_SIZE,
                               k_neighbors=LID_K_NEIGHBORS)
    elif detector_name == DETECTORS[2]:  # ML-LOO
        detector = MLLooDetector(model, dname)
    elif detector_name == DETECTORS[3]:  # Odds
        detector = OddsAreOddDetector(model, dname, noise_list=ODDS_NOISE_LIST, n_noise_samples=ODDS_N_SAMPLE)
    elif detector_name == DETECTORS[4]:  # PN
        detector = PNDetector(model, dname, path_checkpoint, max_epochs=PN_MAX_EPOCHS, seed=seed)
    elif detector_name == DETECTORS[5]:  # RC
        detector = RegionBasedClassifier(model, dname, radius=RC_RADIUS, n_noise_samples=RC_N_SAMPLE)
    elif detector_name == DETECTORS[6]:  # BAARD S1 - Applicability
        detector = ApplicabilityStage(model, dname)
    elif detector_name == DETECTORS[7]:  # BAARD S2 - Reliability
        detector = ReliabilityStage(model, dname, k_neighbors=B2_K_NEIGHBORS, subsample_scale=B2_SAMPLE_SCALE)
    elif detector_name == DETECTORS[8]:  # BAARD S3 - Decidability
        detector = DecidabilityStage(model, dname, k_neighbors=B3_K_NEIGHBORS, subsample_scale=B3_SAMPLE_SCALE)
    elif detector_name == DETECTORS[9]:  # BAARD Full
        detector = BAARD(model, dname,
                         k1_neighbors=B2_K_NEIGHBORS, subsample_scale1=B2_SAMPLE_SCALE,
                         k2_neighbors=B3_K_NEIGHBORS, subsample_scale2=B3_SAMPLE_SCALE)
    else:
        raise NotImplementedError
    return detector


def extract_features():
    """Extract features from a dataset."""
    return []


def save_features(features: ArrayLike, name: str, path: str):
    """Save features."""
    detector = None


def main_pipeline():
    """Full pipeline for running a detector."""
    path_output, seed, data_name, attack_name, l_norm, adv_files, att_eps_list, detector_name = parse_arguments()


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    # TODO: seed, data, attack, and detector should NOT have default value! Debug only.
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-d', '--data', choices=DATASETS, default='MNIST')
    parser.add_argument('-a', '--attack', choices=ATTACKS, default='APGD')
    parser.add_argument('-l', '--lnorm', choices=L_NORM, default=2)
    parser.add_argument('--detector', type=str, default='FS', choices=DETECTORS)
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
    # Example:
    # python ./experiments/detectors_extract_features.py --seed 1234 --data MNIST --attack APGD
    main_pipeline()
