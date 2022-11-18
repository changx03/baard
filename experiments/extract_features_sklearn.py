"""Extract and save features from adversarial detectors."""
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import seed_everything
from sklearn.base import ClassifierMixin

from baard.attacks import ATTACKS_SKLEARN
from baard.classifiers import TABULAR_DATASETS, TABULAR_MODELS
from baard.detections import (DETECTOR_EXTENSIONS, DETECTORS_SKLEARN,
                              SklearnApplicabilityStage, SklearnBAARD,
                              SklearnDecidabilityStage, SklearnDetector,
                              SklearnRegionBasedClassifier,
                              SklearnReliabilityStage)
from baard.utils.miscellaneous import (create_parent_dir,
                                       find_available_attacks_sklearn)

# NOTE: All tabular data are binary classification problems
N_CLASSES = 2
# Hyperparameter for RC
RC_RADIUS = 0.2
RC_N_SAMPLE = 1000
# Hyperparameter for BAARD
B2_K_NEIGHBORS = 1
B3_K_NEIGHBORS = 15


def init_detector(detector_name: str, data_name: str, model: ClassifierMixin
                  ) -> tuple[SklearnDetector, str]:
    """Initialize a detector"""
    detector = None
    if detector_name == DETECTORS_SKLEARN[0]:  # 'RC'
        detector = SklearnRegionBasedClassifier(model, data_name, N_CLASSES, RC_RADIUS, RC_N_SAMPLE)
    elif detector_name == DETECTORS_SKLEARN[1]:  # 'BAARD-S1'
        detector = SklearnApplicabilityStage(model, data_name, N_CLASSES)
    elif detector_name == DETECTORS_SKLEARN[2]:  # 'BAARD-S2'
        detector = SklearnReliabilityStage(model, data_name, N_CLASSES, B2_K_NEIGHBORS)
    elif detector_name == DETECTORS_SKLEARN[3]:  # 'BAARD-S3'
        detector = SklearnDecidabilityStage(model, data_name, N_CLASSES, B3_K_NEIGHBORS)
    elif detector_name == DETECTORS_SKLEARN[4]:  # 'BAARD'
        detector = SklearnBAARD(model, data_name, N_CLASSES,
                                k1_neighbors=B2_K_NEIGHBORS, k2_neighbors=B3_K_NEIGHBORS)
    else:
        raise NotImplementedError
    detector_ext = DETECTOR_EXTENSIONS[detector.__class__.__name__]
    return detector, detector_ext


def prepare_detector(detector, detector_name, detector_ext, data_name, path):
    """Train or load a detector."""
    if detector_name == 'SklearnRegionBasedClassifier':  # RC doesn't require training.
        return detector
    else:  # For BAARD and its stages.
        path_detector = os.path.join(path, detector_name, f'{detector_name}-{data_name}{detector_ext}')
        if os.path.exists(path_detector):
            print(f'Found pre-trained {detector_name}. Load from {path_detector}')
            detector.load(path_detector)
        else:  # Train detector.
            path_data = os.path.join(path, f'{data_name}-splitted.pickle')
            if not os.path.exists(path_data):
                raise FileNotFoundError(f'Cannot find training data: {path_data}')
            data = pickle.load(open(path_data, 'rb'))
            X_train = data['X_train']
            y_train = data['y_train']
            detector.train(X_train, y_train)
            detector.save(path_detector)
    return detector


def extract_and_save_features(detector, attack_name, data_name, adv_files, att_eps_list, path):
    """Extract and save features."""
    detector_name = detector.__class__.__name__
    path_record = os.path.join(path, detector_name, f'{detector_name}-{data_name}-{attack_name}.csv')
    with open(path_record, 'a', encoding='UTF-8') as file:
        file.write(','.join(['attack', 'path']) + '\n')
        for eps, path_data in zip(att_eps_list, adv_files):
            path_features = os.path.join(
                path, detector_name, attack_name,
                f'{detector_name}-{data_name}-{attack_name}-{eps}.pt')
            if os.path.exists(path_features):
                print(f'Found {path_features} Skip!')
            else:
                file.write(','.join([str(eps), path_data]) + '\n')
                print(f'Running {detector_name} on {data_name} with eps={eps}')
                data = pickle.load(open(path_data, 'rb'))
                X = data['X']
                features = detector.extract_features(X)
                path_features = create_parent_dir(path_features, file_ext='.pt')
                print(f'Save features to: {path_features}')
                torch.save(features, path_features)
            print('#' * 80)


def extract_features(seed: int,
                     data_name: str,
                     model_name: str,
                     detector_name: str,
                     attack_name: str,
                     path_attack: str,
                     adv_files: List,
                     att_eps_list: List):
    """Use a detector to extract features."""

    seed_everything(seed)

    # Load classifier
    if model_name == 'DecisionTree':
        model_name = 'ExtraTreeClassifier'
    path_model = os.path.join(path_attack, f'{model_name}-{data_name}.pickle')
    model = pickle.load(open(path_model, 'rb'))

    # Initialize detector
    detector, detector_ext = init_detector(
        detector_name=detector_name,
        data_name=data_name,
        model=model,
    )
    detector_name = detector.__class__.__name__
    print('DETECTOR:', detector_name)
    print('DETECTOR EXTENSION:', detector_ext)

    path_json = create_parent_dir(
        os.path.join(path_attack, detector_name, f'{detector_name}-{data_name}.json'), file_ext='.json')
    if not os.path.exists(path_json):
        detector.save_params(path_json)

    # Train or load previous results.
    detector = prepare_detector(detector, detector_name, detector_ext, data_name, path=path_attack)
    # Extract features and save them.
    extract_and_save_features(detector, attack_name, data_name, adv_files, att_eps_list, path=path_attack)


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/extract_features_sklearn.py -s 1234 --data BC --model SVM --attack PGD-Linf --detector "BAARD-S3"
    python ./experiments/extract_features_sklearn.py -s 1234 --data banknote --model SVM --attack PGD-Linf --detector "RC"
    python ./experiments/extract_features_sklearn.py -s 1234 --data banknote --model DecisionTree --attack DecisionTreeAttack --detector "BAARD-S1"
    python ./experiments/extract_features_sklearn.py -s 1234 --data banknote --model DecisionTree --attack DecisionTreeAttack --detector "RC"
    """
    parser = ArgumentParser()
    # NOTE: seed, data, and detector should NOT have default value! Debug only.
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--data', choices=TABULAR_DATASETS, required=True)
    parser.add_argument('--model', choices=TABULAR_MODELS, required=True)
    parser.add_argument('--detector', type=str, choices=DETECTORS_SKLEARN, required=True)
    parser.add_argument('-a', '--attack', choices=ATTACKS_SKLEARN, default='PGD-Linf')
    parser.add_argument('-p', '--path', type=str, default='results',
                        help='The path for loading pre-trained adversarial examples, and saving results.')
    args = parser.parse_args()
    seed = args.seed
    data_name = args.data
    model_name = args.model
    detector_name = args.detector
    attack_name = args.attack
    path = args.path

    path_attack = Path(os.path.join(path, f'exp{seed}', f'{data_name}-{model_name}')).absolute()
    adv_files, att_eps_list = find_available_attacks_sklearn(path_attack, attack_name)

    print('PATH:', path_attack)
    print('DATA:', data_name)
    print('MODEL:', model_name)
    print('DETECTOR:', detector_name)
    print('ATTACK:', attack_name)
    print('EPSILON:', att_eps_list)

    return seed, data_name, model_name, detector_name, attack_name, path_attack, adv_files, att_eps_list


def main():
    """Main pipeline for extracting features."""
    seed, data_name, model_name, detector_name, attack_name, path_attack, adv_files, att_eps_list = parse_arguments()
    extract_features(seed, data_name, model_name, detector_name, attack_name, path_attack, adv_files, att_eps_list)

    # TODO: Known issue in Decidability Stage. Unmatched label in BC trained with SVM.
    # # python ./experiments/extract_features_sklearn.py -s 543597 --data "BC" --model="SVM" --detector "BAARD" -a "PGD-Linf"
    # path_attack = Path(os.path.join('results', 'exp543597', 'BC-SVM')).absolute()
    # adv_files, att_eps_list = find_available_attacks_sklearn(path_attack, 'PGD-Linf')
    # extract_features(543597, 'BC', 'SVM', 'BAARD-S3', 'PGD-Linf', path_attack, adv_files, att_eps_list)


if __name__ == '__main__':
    main()
