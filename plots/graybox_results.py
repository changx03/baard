"""Compute graybox results over 5 runs."""
import os
from pathlib import Path

import pandas as pd

from experiments.eval_features import eval_features

PATH_ROOT = Path(os.getcwd()).absolute()
print('ROOT:', PATH_ROOT)

DETECTORS = [
    'RegionBasedClassifier',
    'FeatureSqueezingDetector',
    'LIDDetector',
    'OddsAreOddDetector',
    'MLLooDetector',
    'PNDetector',
    'BAARD',
]
ATTACKS = ['APGD-L2', 'APGD-Linf', 'CW2-L2']
DETECTOR_MAPPING = {
    'RegionBasedClassifier': 'RC',
    'FeatureSqueezingDetector': 'FS',
    'LIDDetector': 'LID',
    'OddsAreOddDetector': 'Odds',
    'MLLooDetector': 'ML-LOO',
    'PNDetector': 'PN',
    'BAARD': 'BAARD',
}
SEEDS = [
    '188283',
    '292478',
    '382347',
    '466364',
    '543597',
]


def get_result_one_detector(detector_name,
                            data_name,
                            attack_name,
                            eps,
                            path_input,
                            path_output):
    """Get DataFrame from one dataset."""
    # NOTE: LID and ML-LOO need to call proba.
    if detector_name == 'MLLooDetector' or detector_name == 'LIDDetector':
        _detector_name = detector_name + '(proba)'
    else:
        _detector_name = detector_name

    file_adv = f'{_detector_name}-{data_name}-{attack_name}-{eps}.pt'
    file_clean = f'{_detector_name}-{data_name}-{attack_name}-clean.pt'
    _, df_auc_tpr = eval_features(path_input, path_output, file_clean, file_adv)
    df_auc_tpr = (df_auc_tpr * 100).round(1)
    df_auc_tpr['data'] = data_name
    df_auc_tpr['detector'] = detector_name
    df_auc_tpr['attack'] = attack_name
    df_auc_tpr['eps'] = eps
    df_auc_tpr = df_auc_tpr[['data', 'detector', 'attack', 'eps', 'auc', '1fpr', '5fpr', '10fpr']]
    return df_auc_tpr


def get_df_list(data_name,
                seed_list,
                attack_list,
                detector_list,
                detector_mapping,
                eps_list,
                eps_mapping,
                ):
    """Get DataFrame for 1 dataset from multiple runs."""
    df_list = []
    for seed in seed_list:
        _df = pd.DataFrame()
        path_auc_out = os.path.join(PATH_ROOT, 'results', f'exp{seed}', 'roc')
        if not os.path.exists(path_auc_out):
            os.makedirs(path_auc_out)

        path_df = os.path.join(path_auc_out, f'graybox-{data_name}.csv')
        if os.path.exists(path_df):
            _df = pd.read_csv(path_df)
        else:
            for attack_name in attack_list:
                for eps in eps_list[attack_name]:
                    for detector_name in detector_list:
                        path_input = os.path.join(
                            PATH_ROOT, 'results', f'exp{seed}', data_name, detector_name, attack_name)
                        _df_row = get_result_one_detector(
                            detector_name,
                            data_name,
                            attack_name,
                            eps,
                            path_input,
                            path_auc_out,
                        )
                        _df = pd.concat([_df, _df_row])

            _df['detector'] = _df['detector'].map(detector_mapping)
            _df['eps'] = _df['eps'].map(eps_mapping)
            _df.to_csv(path_df)
        df_list.append(_df)
    return df_list


def save_df_list(df_list, path_output, data_name):
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    df_mean = pd.concat(df_list, ignore_index=True).groupby(['data', 'detector', 'attack', 'eps']).mean()
    df_std = pd.concat(df_list, ignore_index=True).groupby(['data', 'detector', 'attack', 'eps']).std()

    df_mean.round(2).to_csv(os.path.join(path_output, f'graybox_{data_name}_mean.csv'))
    df_std.round(2).to_csv(os.path.join(path_output, f'graybox_{data_name}_std.csv'))


def compute_df_minst():
    """Get a list of DataFrame for MNIST."""
    data_name = 'MNIST'
    eps_list = {
        'APGD-L2': ['4.0', '8.0'],
        'APGD-Linf': ['0.22', '0.66'],
        'CW2-L2': ['0.0']
    }
    mnist_eps_mapping = {
        '4.0': 'Low',
        '8.0': 'High',
        '0.22': 'Low',
        '0.66': 'High',
        '0.0': 'NA',
    }

    df_mnist_list = get_df_list(
        data_name,
        seed_list=SEEDS,
        attack_list=ATTACKS,
        detector_list=DETECTORS,
        detector_mapping=DETECTOR_MAPPING,
        eps_list=eps_list,
        eps_mapping=mnist_eps_mapping,
    )
    return df_mnist_list


def compute_df_cifar10():
    """Get a list of DataFrame for CIFAR10."""
    data_name = 'CIFAR10'
    eps_list = {
        'APGD-L2': ['0.3', '3.0'],
        'APGD-Linf': ['0.01', '0.1'],
        'CW2-L2': ['0.0']
    }
    cifar10_eps_mapping = {
        '0.3': 'Low',
        '3.0': 'High',
        '0.01': 'Low',
        '0.1': 'High',
        '0.0': 'NA',
    }
    df_cifar10_list = get_df_list(
        data_name,
        seed_list=SEEDS,
        attack_list=ATTACKS,
        detector_list=DETECTORS,
        detector_mapping=DETECTOR_MAPPING,
        eps_list=eps_list,
        eps_mapping=cifar10_eps_mapping,
    )
    return df_cifar10_list


def main():
    """Main pipeline."""
    mnist_list = compute_df_minst()
    cifar10_list = compute_df_cifar10()

    path_graybox_output = os.path.join(PATH_ROOT, 'plots')
    save_df_list(mnist_list, path_graybox_output, 'MNIST')
    save_df_list(cifar10_list, path_graybox_output, 'CIFAR10')


if __name__ == '__main__':
    main()
