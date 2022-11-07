"""Extract and save features from adversarial detectors."""
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List


from baard.attacks import ATTACKS, L_NORM
from baard.classifiers import DATASETS
from baard.detections import DETECTORS
from baard.utils.miscellaneous import create_parent_dir, find_available_attacks

from extract_features_utils import (extract_and_save_features,
                                    get_pretrained_model_path, init_detector,
                                    prepare_detector)


def detector_extract_features(path_output: str,
                              seed: int,
                              data_name: str,
                              attack_name: str,
                              l_norm: str,
                              adv_files: List,
                              att_eps_list: List,
                              detector_name: str):
    """Use a detector to extract features."""
    path_output, seed, data_name, attack_name, l_norm, adv_files, att_eps_list, detector_name = parse_arguments()

    # Initialize detector
    path_clf_checkpoint = get_pretrained_model_path(data_name)
    detector, detector_ext = init_detector(detector_name=detector_name, data_name=data_name,
                                           path_checkpoint=path_clf_checkpoint, seed=seed)
    print('DETECTOR:', detector_name)
    print('DETECTOR EXTENSION:', detector_ext)

    detector_name = detector.__class__.__name__
    path_json = create_parent_dir(
        os.path.join(path_output, detector_name, f'{detector_name}-{data_name}.json'), file_ext='.json')
    if not os.path.exists(path_json):
        detector.save_params(path_json)
    # Train or load previous results.
    detector = prepare_detector(detector, detector_name, detector_ext, data_name, path=path_output)
    # Extract features and save them.
    extract_and_save_features(detector, attack_name, data_name, l_norm, adv_files, att_eps_list, path_output)


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/detectors_extract_features.py -s 1234 --data MNIST --attack APGD -l 2 --detector "BAARD-S2"
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


def main():
    """Main pipeline for extracting features."""
    path_output, seed, data_name, attack_name, l_norm, adv_files, att_eps_list, detector_name = parse_arguments()
    detector_extract_features(path_output, seed, data_name, attack_name, l_norm, adv_files, att_eps_list, detector_name)


if __name__ == '__main__':
    main()
