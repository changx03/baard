import logging
import os
import pickle
from pathlib import Path

from baard.detections.baard_applicability_sklearn import \
    SklearnApplicabilityStage


def test_SklearnApplicabilityStage():
    """Test SklearnApplicabilityStage"""

    logging.basicConfig(level=logging.INFO)

    PATH_ROOT = Path(os.getcwd()).absolute()

    data_name = 'banknote'
    clf_name = 'SVM'
    path_data = os.path.join(PATH_ROOT, 'results', 'exp1234', f'{data_name}-{clf_name}')

    model = pickle.load(open(os.path.join(path_data, f'{clf_name}-{data_name}.pickle'), 'rb'))
    detector = SklearnApplicabilityStage(model, 'banknote', 2)

    file_ext = '.skbaard1'
    path_detector_dev = os.path.join('temp', f'dev_baard_applicability_sklearn_{data_name}{file_ext}')
    if os.path.exists(path_detector_dev):
        detector.load(path_detector_dev)
    else:
        data = pickle.load(open(os.path.join(path_data, f'{data_name}-splitted.pickle'), 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']
        detector.train(X_train, y_train)
        detector.save(path_detector_dev)

    data_clean = pickle.load(open(os.path.join(path_data, 'ValClean.pickle'), 'rb'))
    X = data_clean['X']
    y = data_clean['y']
    print(f'Acc: {model.score(X, y)}')

    features = detector.extract_features(X)
    print(features[:10])

    data_adv = pickle.load(open(os.path.join(path_data, 'PGD-Linf-0.2.pickle'), 'rb'))
    X = data_adv['X']
    y = data_adv['y']
    print(f'Adv: {model.score(X, y)}')

    features = detector.extract_features(X)
    print(features[:10])


if __name__ == '__main__':
    test_SklearnApplicabilityStage()
