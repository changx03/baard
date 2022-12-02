"""Train Decision tree and generate adversarial examples using the Decision Tree
Attack.
"""
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import ExtraTreeClassifier

from baard.classifiers import TABULAR_DATA_LOOKUP
from baard.utils.miscellaneous import load_csv

PATH_ROOT = os.getcwd()


def train_clf_generate_adv(data_name, path_outputs, path_input, seed):
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
    path_model = os.path.join(path_outputs, f'ExtraTreeClassifier-{data_name}.pickle')
    if os.path.exists(path_model):
        print(f'Load model from: {path_model}')
        model = pickle.load(open(path_model, 'rb'))
    else:
        model = ExtraTreeClassifier(random_state=seed)
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
    path_adv = os.path.join(path_outputs, 'DecisionTreeAttack.pickle')
    if os.path.exists(path_adv):
        print(f'Found adversarial examples: {path_adv}')
        data_adv = pickle.load(open(path_adv, 'rb'))
        X_adv = data_adv['X']
    else:
        print('Train advx on test set...')
        art_clf = SklearnClassifier(model=model, clip_values=(X_train.min(), X_train.max()))
        attack = DecisionTreeAttack(classifier=art_clf, verbose=False)
        X_adv = attack.generate(X_test)
        print(f'Save adversarial examples to: {path_adv}')
        data_adv = {
            'X': X_adv,
            'y': y_test,
        }
        print(f'Save to {path_adv}')
        pickle.dump(data_adv, open(path_adv, 'wb'))
    acc_adv = model.score(X_adv, y_test)
    print(f'[Accuracy] Adv: {acc_adv:.4f}')

    # Generate advx on validation set
    path_adv_val = os.path.join(path_outputs, 'DecisionTreeAttack-val.pickle')
    if os.path.exists(path_adv_val):
        print(f'Found adversarial examples: {path_adv_val}')
    else:
        print('Train advx on validation set...')
        art_clf = SklearnClassifier(model=model, clip_values=(X_train.min(), X_train.max()))
        attack = DecisionTreeAttack(classifier=art_clf, verbose=False)
        X_adv_val = attack.generate(X_val)
        print(f'Save adversarial examples to: {path_adv_val}')
        data_adv_val = {
            'X': X_adv_val,
            'y': y_val,
        }
        print(f'Save to {path_adv_val}')
        pickle.dump(data_adv_val, open(path_adv_val, 'wb'))


def parse_arguments():
    """Parse command line arguments.
    Examples:
    python ./experiments/tree_train_adv.py -d=banknote
    python ./experiments/tree_train_adv.py -d=BC
    python ./experiments/tree_train_adv.py -d=HTRU2
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1234)
    # parser.add_argument('-d', '--data', type=str, required=True)
    # TODO: for development only
    parser.add_argument('-d', '--data', type=str, default='banknote')
    parser.add_argument('-i', '--input', type=str, default='data')
    parser.add_argument('-o', '--output', type=str, default='results')
    args = parser.parse_args()
    seed = args.seed
    data = args.data
    path_input = os.path.join(PATH_ROOT, args.input)

    print('PATH_ROOT', PATH_ROOT)
    print('SEED:', seed)
    print('DATA:', data)

    path_outputs = os.path.join(PATH_ROOT, args.output, f'exp{seed}', f'{data}-DecisionTree')
    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)
        print(f'Creates dir: {path_outputs}')
    return data, path_outputs, path_input, seed


def main():
    """Main pipeline for generating adversarial examples."""
    data, path_outputs, path_input, seed = parse_arguments()
    train_clf_generate_adv(data, path_outputs, path_input, seed)


if __name__ == '__main__':
    main()
