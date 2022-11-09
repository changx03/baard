"""Evaluating extracted features from detectors"""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import torch
from pandas import DataFrame

from baard.utils.eval_utils import compute_roc_auc, tpr_at_n_fpr

logger = logging.getLogger(__name__)


def eval_features(path_input, path_output, file_clean, file_adv,
                  filename_output: None) -> Union[DataFrame, DataFrame]:
    """Compute ans save ROC, AUC, TPR at 1%, 5% and 10% FPR. Returns (ROC, others).
    """
    features_clean = torch.load(os.path.join(path_input, file_clean))
    features_adv = torch.load(os.path.join(path_input, file_adv))
    fpr, tpr, auc_score, thresholds = compute_roc_auc(features_clean, features_adv)

    tpr_1fpr, _ = tpr_at_n_fpr(fpr, tpr, thresholds, n_fpr=0.01)
    tpr_5fpr, _ = tpr_at_n_fpr(fpr, tpr, thresholds, n_fpr=0.05)
    tpr_10fpr, _ = tpr_at_n_fpr(fpr, tpr, thresholds, n_fpr=0.1)

    # Use `file_adv` name to store
    if filename_output is None:
        filename_output = Path(file_adv).stem  # Remove extension
    # Use DataFrame to save ROC
    data_roc = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }
    df_roc = DataFrame(data_roc)
    path_roc = os.path.join(path_output, f'{filename_output}-roc.csv')
    df_roc.to_csv(path_roc, index=False)
    logger.info('Save ROC to: %s', path_roc)

    # Save AUC and TPR@?FPR
    data_other = {
        'auc': [auc_score],
        '1fpr': [tpr_1fpr],
        '5fpr': [tpr_5fpr],
        '10fpr': [tpr_10fpr],
    }
    df_auc_tpr = DataFrame(data_other)
    path_auc_tpr = os.path.join(path_output, f'{filename_output}-auc_tpr.csv')
    df_auc_tpr.to_csv(path_auc_tpr, index=False)
    logger.info('Save AUC and TPRs to: %s', path_auc_tpr)
    return df_roc, df_auc_tpr


def parse_arguments():
    """Parse command line arguments.
    Example:
    python ./experiments/eval_features.py -i "./results/exp1234/MNIST/ApplicabilityStage/APGD-L2" \
        -o "./results/exp1234/roc" \
        --clean "ApplicabilityStage-MNIST-APGD-L2-clean" \
        --adv "ApplicabilityStage-MNIST-APGD-L2-0.5"
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--clean', type=str, required=True)
    parser.add_argument('--adv', type=str, required=True)

    args = parser.parse_args()
    path_input = args.input
    path_output = args.output
    file_clean = args.clean
    file_adv = args.adv

    # Check extension. Features are saved using torch.save();
    if file_clean[-3:] != '.pt':
        file_clean = file_clean + '.pt'
    if file_adv[-3:] != '.pt':
        file_adv = file_adv + '.pt'

    # Check file existence.
    for file in [file_clean, file_adv]:
        path_file = os.path.join(path_input, file)
        if not os.path.exists(path_file):
            raise FileNotFoundError(f'Cannot find {path_file}')

    # Check output directory existence.
    path_output = Path(path_output).absolute()
    if not path_output.is_dir():
        logger.warning('Directory does not exist. Create %s', path_output)
        os.makedirs(path_output)

    return path_input, path_output, file_clean, file_adv


def main():
    """Main pipeline for evaluating extracted features from detectors"""
    path_input, path_output, file_clean, file_adv = parse_arguments()
    eval_features(path_input, path_output, file_clean, file_adv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
