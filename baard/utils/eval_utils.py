"""Utility functions for evaluation metrics."""
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegressionCV
from numpy.typing import ArrayLike


def tpr_at_n_fpr(fprs, tprs, thresholds, n_fpr: float = 0.05):
    """Get TPR and threshold based on FPR at certain percentage.

    :param fprs: List of FPRs.
    :param tprs: List of TPRs.
    :param thresholds: List of thresholds.
    :param float n_fpr: float. Default is
    :returns:
        - tpr - TPR at corresponding FPR.
        - threshold - Threshold at corresponding FPR.
    """
    err = 1e-3
    idx_at_fpr = np.where(fprs <= n_fpr + err)[0][-1]
    threshold_at_fpr = thresholds[idx_at_fpr]
    tpr_at_fpr = tprs[idx_at_fpr]
    return tpr_at_fpr, threshold_at_fpr


def combine_extracted_features(X_clean: ArrayLike, X_adv: ArrayLike):
    """Transform clean and adversarial features into one dataset, with adversarial
    labelled as 1 and clean labeled as 0.
    """
    assert X_clean.shape == X_adv.shape, \
        'Clean and their counter adv examples should have the same shape!'
    features_mix = np.concatenate([X_clean, X_adv])
    n_mix = len(features_mix)
    # Fit regression even with only 1 feature per sample!
    features_mix = np.reshape(features_mix, (n_mix, -1))

    # Fix nan
    features_mix = np.nan_to_num(features_mix, nan=9999, posinf=9999, neginf=-9999)

    # Prepare labels: clean: 0, adversarial examples: 1
    labels_mix = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_adv))])
    return features_mix, labels_mix


def compute_roc_auc(test_clean: ArrayLike, test_adv: ArrayLike,
                    train_clean: ArrayLike, train_adv: ArrayLike
                    ) -> tuple[ArrayLike, ArrayLike, float, float]:
    """Return FPR, TPR, ROC_AUC and thresholds."""
    X_train, y_train = combine_extracted_features(train_clean, train_adv)
    X_test, y_test = combine_extracted_features(test_clean, test_adv)

    regressor = LogisticRegressionCV(max_iter=50000)
    regressor.fit(X_train, y_train)
    pred = regressor.predict_proba(X_test)
    pred = pred[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    auc_score = metrics.auc(fpr, tpr)

    return fpr, tpr, auc_score, thresholds
