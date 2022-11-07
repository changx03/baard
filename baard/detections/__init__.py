"""Constants for detectors."""
from .baard_applicability import ApplicabilityStage
from .baard_decidability import DecidabilityStage
from .baard_detector import BAARD
from .baard_reliability import ReliabilityStage
from .base_detector import Detector
from .feature_squeezing import FeatureSqueezingDetector
from .lid import LIDDetector
from .ml_loo import MLLooDetector
from .odds_are_odd import OddsAreOddDetector
from .pn_detector import PNDetector
from .region_based_classifier import RegionBasedClassifier

DETECTORS = ['FS', 'LID', 'ML-LOO', 'Odds', 'PN', 'RC', 'BAARD-S1', 'BAARD-S2', 'BAARD-S3', 'BAARD']
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
