"""Train SVM and generate adversarial examples using the PGD attack.
"""
import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from baard.classifiers import TABULAR_DATA_LOOKUP
from baard.utils.miscellaneous import filter_exist_eps, load_csv

PATH_ROOT = os.getcwd()

ATTACKS = ['PGD']


def get_attack(clf, attack_name, clip, verbose=True, eps=None):
    """Get attack class."""
    art_clf = SklearnClassifier(model=clf, clip_values=clip)
    attack = None
    if attack_name == ATTACKS[0]:  # PGD
        attack = BasicIterativeMethod(
            estimator=art_clf,
            eps=eps,
            verbose=verbose,
            eps_step=0.01,
        )
    else:
        raise NotImplementedError(f'{attack_name} is NOT implemented!')
    return attack


def train_clf_generate_adv(data_name, path_outputs, path_input, seed, attack_name, eps_list):
    """Train the classifier, and generate adversarial examples."""
    pl.seed_everything(seed)

    data_filename = TABULAR_DATA_LOOKUP[data_name]
    path_data = os.path.join(path_input, data_filename)
    if not os.path.exists(path_data):
        raise FileNotFoundError(f'Cannot find {path_data}')

    ############################################################################
    # Split data.
    path_data_output = os.path.join(path_outputs, f'{data_name}-splitted.pickle')
    if os.path.exists(path_data_output):
        print(f'Load data from: {path_data_output}')
        data = pickle.load(open(path_data_output, 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        X, y = load_csv(path_data)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # NOTE: Use fixed 60-20-20 split
        X_train, X_mixed, y_train, y_mixed = train_test_split(X, y, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_mixed, y_mixed, test_size=0.5, random_state=seed)

        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
        }
        print(f'Save to: {path_data_output}')
        pickle.dump(data, open(path_data_output, 'wb'))
    print(f'Train: {len(X_train)} Validation: {len(X_val)} Test: {len(X_test)}')
    print(f'Min: {X_train.min()} Max:{X_train.max()}')

    ############################################################################
    # Train classifier and save model.
    path_model = os.path.join(path_outputs, f'SVM-{data_name}.pickle')
    if os.path.exists(path_model):
        print(f'Load model from: {path_model}')
        model = pickle.load(open(path_model, 'rb'))
    else:
        model = SVC(probability=True)
        model.fit(X_train, y_train)
        print(f'Save model to: {path_model}')
        pickle.dump(model, open(path_model, 'wb'))
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print((f'[Accuracy] Train: {acc_train:.4f}, Test: {acc_test:.4f}'))

    ############################################################################
    # Generate adversarial examples and save results.
    # Filter correct examples
    path_val_clean = os.path.join(path_outputs, 'ValClean.pickle')
    path_adv_clean = os.path.join(path_outputs, 'AdvClean.pickle')
    if os.path.exists(path_adv_clean) and os.path.exists(path_val_clean):
        print('Found existing correct data split!')
        data_val = pickle.load(open(path_val_clean, 'rb'))
        X_val = data_val['X']
        y_val = data_val['y']

        data_test = pickle.load(open(path_adv_clean, 'rb'))
        X_test = data_test['X']
        y_test = data_test['y']
    else:
        pred_val = model.predict(X_val)
        indices_val_correct = np.where(pred_val == y_val)[0]
        X_val = X_val[indices_val_correct]
        y_val = y_val[indices_val_correct]
        if len(X_val) > 1000:
            X_val = X_val[:1000]
            y_val = y_val[:1000]
        pickle.dump({'X': X_val, 'y': y_val}, open(path_val_clean, 'wb'))

        pred_test = model.predict(X_test)
        indices_test_correct = np.where(pred_test == y_test)[0]
        X_test = X_test[indices_test_correct]
        y_test = y_test[indices_test_correct]
        if len(X_test) > 1000:
            X_test = X_test[:1000]
            y_test = y_test[:1000]
        pickle.dump({'X': X_test, 'y': y_test}, open(path_adv_clean, 'wb'))

    acc_val = model.score(X_val, y_val)
    acc_test = model.score(X_test, y_test)
    print((f'[After filter] Val: {acc_val:.4f}, Test: {acc_test:.4f}'))

    # Generate adversarial examples
    attack_norm = 'Linf'
    print(f'Attack Norm: {attack_norm}')

    eps_list = filter_exist_eps(eps_list,
                                path_outputs,
                                'PGD',
                                lnorm=attack_norm,
                                n=len(X_test))
    print('Uncompleted Epsilon', eps_list)

    path_log_results = os.path.join(path_outputs, f'{attack_name}-{attack_norm}-SuccessRate.csv')
    with open(path_log_results, 'a', encoding='UTF-8') as file:
        file.write(','.join(['eps', 'success_rate']) + '\n')
        # pbar = tqdm(eps_list, total=len(eps_list))
        for e in eps_list:
            # pbar.set_description(f'Train at {e}', refresh=True)
            try:
                path_adv = os.path.join(path_outputs, f'{attack_name}-{attack_norm}-{e}.pickle')
                if os.path.exists(path_adv):
                    print(f'Found adversarial examples: {path_adv} Skip!')
                    data_adv = pickle.load(open(path_adv, 'rb'))
                    X_adv = data_adv['X']
                else:
                    clip_range = (X_train.min(), X_train.max())
                    attack = get_attack(model, attack_name, clip_range, eps=e)
                    X_adv = attack.generate(X_test)
                    data_adv = {
                        'X': X_adv,
                        'y': y_test,
                    }
                    pickle.dump(data_adv, open(path_adv, 'wb'))
                acc_adv = model.score(X_adv, y_test)
                print(f'[Accuracy] Eps: {e:.2f} Adv: {acc_adv:.4f}')
                file.write(','.join([f'{i}' for i in [e, acc_adv]]) + '\n')
            except BaseException as err:
                print(f'WARNING: Catch an exception: {err}')


def parse_arguments():
    """Parse command line arguments.
    Examples:
    python ./experiments/svm_train_adv.py -d=banknote --eps="[0.01,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1]"
    python ./experiments/svm_train_adv.py -d=BC --eps="[0.01,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1]"
    python ./experiments/svm_train_adv.py -d=HTRU2 --eps="[0.01,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1]"
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-d', '--data', type=str, default='banknote')
    parser.add_argument('-a', '--attack', default='PGD', choices=ATTACKS)
    parser.add_argument('-i', '--input', type=str, default='data')
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument('--eps', type=json.loads, default='[0.22,0.66]',
                        help='A list of epsilons as a JSON string. e.g., "[0.22,0.66]".')
    args = parser.parse_args()
    seed = args.seed
    data = args.data
    attack_name = args.attack
    path_input = os.path.join(PATH_ROOT, args.input)
    eps_list = np.round(args.eps, 2).astype(float)  # Use float numbers.

    print('PATH_ROOT', PATH_ROOT)
    print('SEED:', seed)
    print('DATA:', data)
    print('ATTACK:', attack_name)
    print('EPS:', eps_list)

    path_outputs = os.path.join(PATH_ROOT, args.output, f'exp{seed}', f'{data}-SVM')
    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)
        print(f'Creates dir: {path_outputs}')
    return data, path_outputs, path_input, seed, attack_name, eps_list


def main():
    """Main pipeline for generating adversarial examples."""
    data, path_outputs, path_input, seed, attack_name, eps_list = parse_arguments()
    train_clf_generate_adv(data, path_outputs, path_input, seed, attack_name, eps_list)


if __name__ == '__main__':
    main()
