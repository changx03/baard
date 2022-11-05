"""Extract and save features from adversarial detectors."""
import json
import os
from argparse import ArgumentParser
from pathlib import Path

from numpy.typing import ArrayLike

from baard.attacks import ATTACKS, L_NORM
from baard.classifiers import DATASETS
from baard.detections import DETECTORS
from baard.utils.miscellaneous import find_available_attacks


def init_detector(detector_name: str, data_name: str):
    """Initialize a detector."""
    pass


def extract_features():
    """Extract features from a dataset."""
    return []


def save_features(features: ArrayLike, name: str, path: str):
    """Save features."""
    pass


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
