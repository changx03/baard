"""Constants for detectors."""
from .baard_applicability import ApplicabilityStage
from .baard_applicability_sklearn import SklearnApplicabilityStage
from .baard_decidability import DecidabilityStage
from .baard_decidability_sklearn import SklearnDecidabilityStage
from .baard_detector import BAARD
from .baard_detector_sklearn import SklearnBAARD
from .baard_reliability import ReliabilityStage
from .baard_reliability_sklearn import SklearnReliabilityStage
from .base_detector import Detector, SklearnDetector
from .feature_squeezing import FeatureSqueezingDetector
from .lid import LIDDetector
from .ml_loo import MLLooDetector
from .odds_are_odd import OddsAreOddDetector
from .pn_detector import PNDetector
from .region_based_classifier import RegionBasedClassifier
from .region_based_classifier_sklearn import SklearnRegionBasedClassifier

DETECTORS = ['FS', 'LID', 'ML-LOO', 'Odds', 'PN', 'RC', 'BAARD-S1', 'BAARD-S2', 'BAARD-S3', 'BAARD']
DETECTORS_SKLEARN = ['RC', 'BAARD-S1', 'BAARD-S2', 'BAARD-S3', 'BAARD']
# NOTE: FeatureSqueezingDetector holds a list of `.ckpt` (PyTorch Lightening checkpoint).
# PNDetector uses `.ckpt` (PyTorch Lightening checkpoint).
# RegionBasedClassifier does NOT require training.
DETECTOR_EXTENSIONS = {
    'FeatureSqueezingDetector': None,
    'LIDDetector': '.lid',
    'MLLooDetector': '.mlloo',
    'OddsAreOddDetector': '.odds',
    'PNDetector': '.ckpt',
    'RegionBasedClassifier': None,
    'ApplicabilityStage': '.baard1',
    'ReliabilityStage': '.baard2',
    'DecidabilityStage': '.baard3',
    'BAARD': '.baard',
    'SklearnRegionBasedClassifier': None,
    'SklearnApplicabilityStage': '.skbaard1',
    'SklearnReliabilityStage': '.skbaard2',
    'SklearnDecidabilityStage': '.skbaard3',
    'SklearnBAARD': '.skbaard',
}
DETECTOR_CLASS_NAMES = {
    'FS': 'FeatureSqueezingDetector',
    'LID': 'LIDDetector',
    'ML-LOO': 'MLLooDetector',
    'Odds': 'OddsAreOddDetector',
    'PN': 'PNDetector',
    'RC': 'RegionBasedClassifier',
    'BAARD-S1': 'ApplicabilityStage',
    'BAARD-S2': 'ReliabilityStage',
    'BAARD-S3': 'DecidabilityStage',
    'BAARD': 'BAARD',
}
DETECTOR_CLASS_NAMES_SKLEARN = {
    'RC': 'SklearnRegionBasedClassifier',
    'BAARD-S1': 'SklearnApplicabilityStage',
    'BAARD-S2': 'SklearnReliabilityStage',
    'BAARD-S3': 'SklearnDecidabilityStage',
    'BAARD': 'SklearnBAARD',
}
