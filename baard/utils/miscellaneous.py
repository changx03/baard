"""Miscellaneous utility functions. Anything hard to category."""
import datetime
import json
import logging
import os
import warnings
from glob import glob
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def create_parent_dir(path: str, file_ext: str) -> str:
    """Check file extension and parent directory. If it's not exist, create one."""
    path = Path(path).resolve()
    filename, _file_extension = os.path.splitext(path)
    if _file_extension != file_ext:
        path = Path(filename + file_ext)
        logger.warning('Change output path to: %s', path)
    path_output_dir = path.parent
    if not os.path.exists(path_output_dir):
        logger.info('Output directory is not found. Create: %s', path_output_dir)
        os.makedirs(path_output_dir)
    return path


def argsort_by_eps(eps_list: ArrayLike) -> List:
    """Sort a list of epsilon in string. Expect the 1st item is `clean`, e.g., ['clean', 1, 2]"""
    indices = np.argsort([float(i) for i in eps_list[1:]])
    indices = indices + 1
    indices = [0] + list(indices)  # Add `clean` back
    return indices


def filter_exist_eps(eps_list: ArrayLike,
                     path: str,
                     attack_name: str,
                     lnorm: Union[str, int],
                     n: int
                     ) -> List:
    """Remove epsilon if it is already exist."""
    lnorm = norm_parser(lnorm)
    eps_list_not_trained = []
    path_files = [os.path.join(path, f'{attack_name}-{lnorm}-{n}-{e}.pt') for e in eps_list]
    for e, file in zip(eps_list, path_files):
        if not os.path.exists(file):
            eps_list_not_trained.append(e)
    eps_list_not_trained = np.round(eps_list_not_trained, 2)
    return eps_list_not_trained


def find_available_attacks(path_attack: str, attack_name: str, l_norm: str, eps_list: List) -> tuple[List, List, int]:
    """Find pre-trained adversarial examples from the directory."""
    l_norm = norm_parser(l_norm)

    files = glob(os.path.join(path_attack, f'{attack_name}-{l_norm}-*.pt'))
    file_names = [os.path.basename(f) for f in files]
    sample_size_set = set()
    eps_file_list = []
    for name in file_names:
        # Read n_samples
        sample_size = int(name.split('-')[-2])
        sample_size_set.add(sample_size)

        # Read epsilon
        filename, _ = os.path.splitext(name)
        eps = filename.split('-')[-1]
        eps_file_list.append(eps)

    # There should be only 1 sample size.
    if len(sample_size_set) != 1:
        raise Exception(f'Found different sample size! {len(sample_size_set)}')
    n_sample = list(sample_size_set)[0]

    # If eps_list exists, use it instead.
    eps_list_confirmed = ['clean']
    if eps_list is not None:
        files = []
        for eps in eps_list:
            data_path = Path(os.path.join(path_attack, f'{attack_name}-L{l_norm}-{n_sample}-{eps}.pt'))
            if data_path.is_file():
                files.append(data_path)
                eps_list_confirmed.append(eps)
            else:
                warnings.warn(f'Data does not exist. {data_path}')
    else:
        eps_list_confirmed = eps_list_confirmed + eps_file_list
    files = [os.path.join(path_attack, f'AdvClean-{n_sample}.pt')] + files
    logger.info('Found %d files for data including clean and adversarial examples.', len(files))

    # Sort list
    indices_sorted = argsort_by_eps(eps_list_confirmed)
    files = np.array(files)[indices_sorted]
    eps_list_confirmed = np.array(eps_list_confirmed)[indices_sorted]
    assert len(files) == len(eps_list_confirmed)
    return list(files), list(eps_list_confirmed)


def norm_parser(lnorm: Union[str, int]) -> str:
    """Parse L-norm string."""
    lnorm = str(lnorm).lower()
    # Handle `lnorm` without the initial `L` letter.
    if lnorm[0] != 'l':
        lnorm = f'L{lnorm}'
    # Handle where `L` is in lower case.
    lnorm = lnorm[0].upper() + lnorm[1:].lower()
    if lnorm not in ['L0', 'L1', 'L2', 'Linf']:
        raise ValueError(f'{lnorm} is not support L-norm!')
    return lnorm


def to_json(data_dict: object, path: str):
    """Save dictionary as JSON."""
    def converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return str(obj)

    with open(path, 'w', encoding='UTF-8') as file:
        logger.info('Save to: %s', path)
        json.dump(data_dict, file, default=converter)
