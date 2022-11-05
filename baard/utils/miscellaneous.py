"""Miscellaneous utility functions. Anything hard to category."""
from pathlib import Path
import os
import logging
from glob import glob
import warnings
from typing import List

import numpy as np
from numpy.typing import ArrayLike


def create_parent_dir(path: str, file_ext: str = '.np') -> str:
    """Check file extension and parent directory. If it's not exist, create one."""
    path = Path(path).resolve()
    filename, _file_extension = os.path.splitext(path)
    if _file_extension != file_ext:
        path = Path(filename + file_ext)
        logging.warning('Change output path to: %s', path)
    path_output_dir = path.parent
    if not os.path.exists(path_output_dir):
        logging.info('Output directory is not found. Create: %s', path_output_dir)
        os.makedirs(path_output_dir)
    return path


def argsort_by_eps(eps_list: ArrayLike) -> List:
    """Sort a list of epsilon in string. Expect the 1st item is `clean`."""
    indices = np.argsort([float(i) for i in eps_list[1:]])
    indices = indices + 1
    indices = [0] + list(indices)  # Add `clean` back
    return indices


def find_available_attacks(path_attack: str, attack_name: str, l_norm: str, eps_list: List) -> tuple[List, List]:
    """Find pre-trained adversarial examples from the directory."""
    files = glob(os.path.join(path_attack, f'{attack_name}.L{l_norm}.n_*.pt'))
    file_names = [os.path.basename(f) for f in files]
    sample_size_set = set()
    eps_file_list = []
    for name in file_names:
        # Read n_samples
        sample_size = name.split('.')[2].split('_')[-1]
        sample_size_set.add(sample_size)

        # Read epsilon
        filename, _ = os.path.splitext(name)
        eps = filename.split('_')[-1]
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
            data_path = Path(os.path.join(path_attack, f'{attack_name}.L{l_norm}.n_{n_sample}.e_{eps}.pt'))
            if data_path.is_file():
                files.append(data_path)
                eps_list_confirmed.append(eps)
            else:
                warnings.warn(f'Data does not exist. {data_path}')
    else:
        eps_list_confirmed = eps_list_confirmed + eps_file_list
    files = [os.path.join(path_attack, f'AdvClean.n_{n_sample}.pt')] + files
    print(f'Found {len(files)} files for data including clean and adversarial examples.')

    # Sort list
    indices_sorted = argsort_by_eps(eps_list_confirmed)
    files = np.array(files)[indices_sorted]
    eps_list_confirmed = np.array(eps_list_confirmed)[indices_sorted]
    assert len(files) == len(eps_list_confirmed)
    return list(files), list(eps_list_confirmed)
