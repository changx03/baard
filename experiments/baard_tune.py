"""This script tunes BAARD using the min. epsilon."""
import os
from argparse import ArgumentParser
from pathlib import Path

from baard.attacks import ATTACKS, L_NORM
from baard.classifiers import DATASETS

from baard_tune_utils import baard_tune_k, baard_tune_sample_size, find_attack_path

BAARD_TUNABLE = ['BAARD-S2', 'BAARD-S3']


def baard_tune(path_output: str, data_name: str, attack_name: str, l_norm: str, path_adv: str, eps: str,
               detector_name: str, k_neighbors: int = None):
    """Tune BAARD."""
    if path_adv is None:
        raise Exception(f'Cannot find {attack_name} eps={eps}. Try another format, e.g., from `2.0` to `2`.')

    if k_neighbors is None:  # Tune the parameter K
        print('Start tuning `k_neighbors`...')
        baard_tune_k(path_output, detector_name, data_name, attack_name, l_norm, path_adv, eps)
    else:
        print('Start tuning `SampleSize`...')
        baard_tune_sample_size(path_output, detector_name, data_name, attack_name, l_norm, path_adv, eps, k_neighbors)


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/baard_tune.py -s 1234 --data MNIST --detector "BAARD-S2" -l inf --eps 0.22
    python ./experiments/baard_tune.py -s 1234 --data MNIST --detector "BAARD-S2" -l inf --eps 0.22 --k 20
    """
    parser = ArgumentParser()
    # NOTE: seed, data, and detector should NOT have default value! Debug only.
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--data', choices=DATASETS, required=True)
    parser.add_argument('--detector', type=str, choices=BAARD_TUNABLE, required=True)
    parser.add_argument('-a', '--attack', choices=ATTACKS, default='APGD')
    parser.add_argument('-l', '--lnorm', type=str, choices=L_NORM, default='2')
    parser.add_argument('-p', '--path', type=str, default='results',
                        help='The path for loading pre-trained adversarial examples, and saving results.')
    parser.add_argument('--eps', type=str, required=True,
                        help="""The epsilon can be both float and int. The code will search existing files for both format.""")
    parser.add_argument('-k', '--k', type=int, default=None,
                        help='When k is None, the script searches for optimal k. Otherwise, it tunes the sample size.')
    args = parser.parse_args()
    seed = args.seed
    data_name = args.data
    attack_name = args.attack
    l_norm = args.lnorm
    detector_name = args.detector
    path = args.path
    eps = args.eps
    k_neighbors = args.k

    path_attack = Path(os.path.join(path, f'exp{seed}', data_name)).absolute()
    path_adv, eps = find_attack_path(path_attack, attack_name, l_norm, eps)

    print('PATH:', path_attack)
    print('ATTACK:', attack_name)
    print('L-NORM:', l_norm)
    print('EPSILON:', eps)
    print('PATH_ADV:', path_adv)
    print('K:', k_neighbors)

    return path_attack, data_name, attack_name, l_norm, path_adv, eps, detector_name, k_neighbors


def main():
    """Main pipeline of tuning BAARD."""
    path_output, data_name, attack_name, l_norm, path_adv, eps, detector_name, k_neighbors = parse_arguments()
    baard_tune(path_output, data_name, attack_name, l_norm, path_adv, eps, detector_name, k_neighbors)


if __name__ == '__main__':
    main()
