"""Demo for BAARD on sklearn classifier."""
import logging
import os
import pickle
from pathlib import Path

from baard.detections import DETECTOR_EXTENSIONS
from baard.detections.baard_applicability_sklearn import \
    SklearnApplicabilityStage
from baard.detections.baard_decidability_sklearn import \
    SklearnDecidabilityStage
from baard.detections.baard_detector_sklearn import SklearnBAARD
from baard.detections.baard_reliability_sklearn import SklearnReliabilityStage


def get_stage_instance(stage_name, data_name, model, n_classes=2):
    """Get one of BAARD instance."""
    if stage_name == 'applicability':
        return SklearnApplicabilityStage(model, data_name, n_classes)
    elif stage_name == 'reliability':
        return SklearnReliabilityStage(model, data_name, n_classes)
    elif stage_name == 'decidability':
        return SklearnDecidabilityStage(model, data_name, n_classes)
    elif stage_name == 'baard':
        return SklearnBAARD(model, data_name, n_classes)
    else:
        raise NotImplementedError


def test_baard_stage(stage_name, data_name='banknote', clf_name='SVM'):
    """Test one of the BAARD's stages."""
    logging.basicConfig(level=logging.INFO)

    PATH_ROOT = Path(os.getcwd()).absolute()

    path_data = os.path.join(PATH_ROOT, 'results', 'exp1234', f'{data_name}-{clf_name}')

    model = pickle.load(open(os.path.join(path_data, f'{clf_name}-{data_name}.pickle'), 'rb'))
    detector = get_stage_instance(stage_name, data_name, model, n_classes=2)
    detector_name = detector.__class__.__name__

    file_ext = DETECTOR_EXTENSIONS[detector_name]
    path_detector_dev = os.path.join('temp', f'dev_baard_{stage_name}_sklearn_{data_name}{file_ext}')
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
    # test_baard_stage('applicability')
    # test_baard_stage('reliability')
    # test_baard_stage('decidability', data_name='BC')
    test_baard_stage('baard')
    test_baard_stage('baard', data_name='BC')
