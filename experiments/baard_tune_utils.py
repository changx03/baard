"""The utility functions for `baard_tune.py`."""
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch

from baard.classifiers import get_lightning_module
from baard.detections import (DETECTOR_EXTENSIONS, DecidabilityStage, Detector,
                              ReliabilityStage)
from baard.utils.miscellaneous import create_parent_dir, norm_parser
from baard.utils.torch_utils import dataset2tensor

from extract_features_utils import get_pretrained_model_path

BAARD_TUNABLE = ['BAARD-S2', 'BAARD-S3']


def baard_inner_train_extract(detector: Detector, data_name: str, eps: str, path_detector: str,
                              path_features: str, path_adv: str) -> None:
    """Train or load a BAARD detector, then extract features"""
    detector_file_name = Path(path_detector).stem
    if not os.path.exists(path_detector):
        detector.train()
        detector.save(path_detector)
    else:
        print(f'Found pre-trained {detector_file_name}')
        detector.load(path_detector)

    # Extract features and save them.
    if not os.path.exists(path_features):
        print(f'Running {detector_file_name} on {data_name} with eps={eps}')
        dataset = torch.load(path_adv)
        X, _ = dataset2tensor(dataset)
        features = detector.extract_features(X)

        path_features = create_parent_dir(path_features, file_ext='.pt')
        print(f'Save features to: {path_features}')
        torch.save(features, path_features)
    else:
        print(f'Found {path_features}. Skip!')
    print('#' * 80)


def baard_tune_k(path_output: str, detector_name: str, data_name: str, attack_name: str, l_norm: str,
                 path_adv: str, eps: str) -> None:
    """Tune BAARD Stage 2: Reliability."""
    k_list = np.concatenate([np.arange(1, 10, 1), np.arange(10, 100, 5), np.arange(10, 201, 10)])
    path_checkpoint = get_pretrained_model_path(data_name)
    model = get_lightning_module(data_name).load_from_checkpoint(path_checkpoint)
    sample_size = 50000  # The largest dataset has 60000 samples -- CIFAR10

    detector_class = ReliabilityStage if detector_name == BAARD_TUNABLE[0] else DecidabilityStage
    tune_var = 'K'
    for k in k_list:
        detector = detector_class(model, data_name, k_neighbors=k, sample_size=sample_size)
        detector_name = detector.__class__.__name__
        detector_ext = DETECTOR_EXTENSIONS[detector.__class__.__name__]

        # NOTE: Tuning uses different PATH.
        path_detector = os.path.join(
            path_output, f'{detector_name}_tune{tune_var}', f'{detector_name}-{k}-{data_name}{detector_ext}')
        path_features = os.path.join(
            path_output, f'{detector_name}_tune{tune_var}', f'{attack_name}-{l_norm}',
            f'{detector_name}-{k}-{data_name}-{attack_name}-{l_norm}-{eps}.pt')
        baard_inner_train_extract(detector, data_name, eps, path_detector, path_features, path_adv)

        # Extract features from clean data
        path_features_clean = os.path.join(
            path_output, f'{detector_name}_tune{tune_var}', f'{attack_name}-{l_norm}',
            f'{detector_name}-{k}-{data_name}-{attack_name}-{l_norm}-clean.pt')
        # NOTE: Clean dataset is hard coded to 1000!
        path_clean = os.path.join(Path(path_adv).resolve().parent, 'AdvClean-1000.pt')
        baard_inner_train_extract(detector, data_name, 'clean', path_detector, path_features_clean, path_clean)


def baard_tune_sample_size(path_output: str, detector_name: str, data_name: str, attack_name: str, l_norm: str,
                           path_adv: str, eps: str, k: str) -> None:
    """Tune BAARD Stage 2: Reliability."""
    sample_size_list = np.concatenate([100, 500], np.arange(1000, 50000, 1000)).astype(int)
    path_checkpoint = get_pretrained_model_path(data_name)
    model = get_lightning_module(data_name).load_from_checkpoint(path_checkpoint)

    detector_class = ReliabilityStage if detector_name == BAARD_TUNABLE[0] else DecidabilityStage
    tune_var = 'SampleSize'
    for sample_size in sample_size_list:
        detector = detector_class(model, data_name, k_neighbors=k, sample_size=sample_size)
        detector_name = detector.__class__.__name__
        detector_ext = DETECTOR_EXTENSIONS[detector.__class__.__name__]

        path_detector = os.path.join(
            path_output, f'{detector_name}_tune{tune_var}',
            f'{detector_name}-{sample_size}-{data_name}{detector_ext}')
        path_features = os.path.join(
            path_output, f'{detector_name}_tune{tune_var}', f'{attack_name}-{l_norm}',
            f'{detector_name}-{sample_size}-{data_name}-{attack_name}-{l_norm}-{eps}.pt')
        baard_inner_train_extract(detector, data_name, eps, path_detector, path_features, path_adv)

        # Extract features from clean data
        path_features_clean = os.path.join(
            path_output, f'{detector_name}_tune{tune_var}', f'{attack_name}-{l_norm}',
            f'{detector_name}-{sample_size}-{data_name}-{attack_name}-{l_norm}-clean.pt')
        # NOTE: Clean dataset is hard coded to 1000!
        path_clean = os.path.join(Path(path_adv).resolve().parent, 'AdvClean-1000.pt')
        baard_inner_train_extract(detector, data_name, 'clean', path_detector, path_features_clean, path_clean)


def find_attack_path(path_attack: str, attack_name: str, l_norm: str, eps: str) -> str:
    """Find a valid adversarial example path for a given epsilon."""
    def _get_file_path(path_base, attack_name, l_norm, eps):
        path_expression = os.path.join(path_base, f'{attack_name}-{l_norm}-*-{eps}.pt')
        files = glob(path_expression)
        if len(files) == 1:
            return files[0]
        elif len(files) > 1:
            raise Exception(f'Found {len(files)} from {path_expression}. Expect only 1 file!')
        return None

    l_norm = norm_parser(l_norm)
    path_adv = _get_file_path(path_attack, attack_name, l_norm, eps)  # For "4"
    if path_adv is None:
        path_adv = _get_file_path(path_attack, attack_name, l_norm, float(eps))  # For "4.0"
    return path_adv, float(eps)
