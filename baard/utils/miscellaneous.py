"""Miscellaneous utility functions. Anything hard to category."""
import logging
import os
import warnings
from glob import glob
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike

from .torch_utils import show_top5_imgs


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
    """Sort a list of epsilon in string. Expect the 1st item is `clean`, e.g., ['clean', 1, 2]"""
    indices = np.argsort([float(i) for i in eps_list[1:]])
    indices = indices + 1
    indices = [0] + list(indices)  # Add `clean` back
    return indices


def find_available_attacks(path_attack: str, attack_name: str, l_norm: str, eps_list: List) -> tuple[List, List]:
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
    print(f'Found {len(files)} files for data including clean and adversarial examples.')

    # Sort list
    indices_sorted = argsort_by_eps(eps_list_confirmed)
    files = np.array(files)[indices_sorted]
    eps_list_confirmed = np.array(eps_list_confirmed)[indices_sorted]
    assert len(files) == len(eps_list_confirmed)
    return list(files), list(eps_list_confirmed)


def plot_images(path_img: str,
                lnorm: Union[str, int],
                eps_list: List,
                attack_name: str,
                n: int = 100,
                ):
    """Plot top-5 images along with their adversarial examples."""
    lnorm = norm_parser(lnorm)

    show_top5_imgs(os.path.join(path_img, f'AdvClean-{n}.pt'), cmap=None)
    print('Clean images')

    for eps in eps_list:
        path_img_adv = os.path.join(path_img, f'{attack_name}-{lnorm}-{n}-{eps}.pt')
        show_top5_imgs(path_img_adv, cmap=None)
        print(f'{attack_name} {lnorm} eps={eps}')


def norm_parser(lnorm: Union[str, int]) -> str:
    """Parse L-norm string."""

    # Handle `lnorm` without the initial `L` letter.
    if lnorm == 'int' or isinstance(lnorm, int):
        lnorm = f'L{lnorm}'

    # Handle where `L` is in lower case.
    lnorm = lnorm[0].upper() + lnorm[1:].lower()

    valid_norm_list = ['L0', 'L1', 'L2', 'Linf']
    if not lnorm in valid_norm_list:
        raise ValueError(f'{lnorm} is not support L-norm!')
    return lnorm
