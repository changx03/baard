import os
import pickle
from pathlib import Path

import numpy as np

from baard.detections.region_based_classifier_sklearn import \
    SklearnRegionBasedClassifier


def test_rb_clf_sklearn():
    """Test RegionBasedClassifier"""

    PATH_ROOT = Path(os.getcwd()).absolute()
    path_data = os.path.join(PATH_ROOT, 'results', 'exp1234', 'banknote-SVM')

    model = pickle.load(open(os.path.join(path_data, 'SVM-banknote.pickle'), 'rb'))
    detector = SklearnRegionBasedClassifier(model, 'banknote', 2)

    data_clean = pickle.load(open(os.path.join(path_data, 'ValClean.pickle'), 'rb'))
    X = data_clean['X']
    y = data_clean['y']
    print(f'Acc: {model.score(X, y)}')

    features = detector.extract_features(X)
    print(features.mean())

    data_adv = pickle.load(open(os.path.join(path_data, 'PGD-Linf-0.2.pickle'), 'rb'))
    X = data_adv['X']
    y = data_adv['y']
    print(f'Adv: {model.score(X, y)}')

    features = detector.extract_features(X)
    print(features.mean())


if __name__ == '__main__':
    test_rb_clf_sklearn()
