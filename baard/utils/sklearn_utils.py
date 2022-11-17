"""Utility functions fro sklearn classifier."""
import logging

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


def get_correct_samples(model: ClassifierMixin, X: ArrayLike, y: ArrayLike
                        ) -> tuple[ArrayLike, ArrayLike]:
    """Check the model's prediction, and return the correct samples."""
    n_before = len(X)
    pred = model.predict(X)
    indices_correct = np.where(pred == y)[0]
    X = X[indices_correct]
    y = y[indices_correct]
    n_after = len(X)
    if n_after != n_before:
        logger.warning('Misclassification in the training set. Before %d != After %d.', n_before, n_after)
    return X, y
