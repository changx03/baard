"""Evaluating extracted features from detectors"""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from baard.attacks import ATTACKS, L_NORM
from baard.classifiers import DATASETS
from baard.detections import DETECTOR_CLASS_NAMES, DETECTORS
from baard.utils.eval_utils import compute_roc_auc, tpr_at_n_fpr
from baard.utils.miscellaneous import norm_parser

logger = logging.getLogger(__name__)


def eval_features(path_features, filename_adv_feature, filename_clean_feature, path_out):
    """Compute ans save ROC, AUC, TPR@1FPR and TPR@5FPR."""
    # TODO: Implement this!
    pass


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/eval_features.py -s 1234 --data MNIST --detector FS --eps 0.5
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--data', type=str, choices=DATASETS, required=True)
    parser.add_argument('--detector', type=str, choices=DETECTORS, required=True)
    parser.add_argument('-a', '--attack', choices=ATTACKS, default='APGD')
    parser.add_argument('-l', '--lnorm', type=str, choices=L_NORM, default='2')
    parser.add_argument('--eps', type=str, required=True)
    parser.add_argument('-p', '--path', type=str, default='results')
    args = parser.parse_args()
    seed = args.seed
    data_name = args.data
    detector_name = args.detector
    attack_name = args.attack
    l_norm = args.lnorm
    path = args.path
    eps = float(args.eps)

    l_norm = norm_parser(l_norm)
    detector_class_name = DETECTOR_CLASS_NAMES[detector_name]
    filename_adv_feature = f'{detector_class_name}-{data_name}-{attack_name}-{l_norm}-{eps}'
    filename_clean_feature = f'{detector_class_name}-{data_name}-{attack_name}-{l_norm}-clean'
    path_features = Path(
        os.path.join(path, f'exp{seed}', data_name, detector_class_name,
                     f'{attack_name}-{l_norm}')
    ).absolute()
    path_out = Path(os.path.join(path, f'exp{seed}', 'roc')).resolve()
    if not os.path.exists(path_out):
        logger.info('Create folder: %s', path_out)
        os.makedirs(path_out)

    print(' PATH FEATURES:', path_features)
    print('   PATH OUTPUT:', path_out)
    print('  ADV FILENAME:', filename_adv_feature)
    print('CLEAN FILENAME:', filename_clean_feature)

    if not (os.path.exists(os.path.join(path_features, f'{filename_adv_feature}.pt'))
            and os.path.exists(os.path.join(path_features, f'{filename_clean_feature}.pt'))):
        raise FileNotFoundError('Cannot find trained features!')
    return path_features, filename_adv_feature, filename_clean_feature, path_out


def main():
    """Main pipeline for evaluating extracted features from detectors"""
    path_features, filename_adv_feature, filename_clean_feature, path_out = parse_arguments()
    eval_features(path_features, filename_adv_feature, filename_clean_feature, path_out)


if __name__ == '__main__':
    main()
