"""Utility functions for evaluation metrics."""
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegressionCV


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


def compute_roc_auc(features_clean, features_adv):
    """Return FPR, TPR, ROC_AUC and thresholds."""
    assert features_clean.shape == features_adv.shape, \
        'Clean and their counter adv examples should have the same shape!'
    features_mix = np.concatenate([features_clean, features_adv])
    n_mix = len(features_mix)
    # Fit regression even with only 1 feature per sample!
    features_mix = np.reshape(features_mix, (n_mix, -1))
    # Prepare labels: clean: 0, adversarial examples: 1
    labels_mix = np.concatenate(
        [np.zeros(len(features_clean)), np.ones(len(features_adv))])

    regressor = LogisticRegressionCV()
    regressor.fit(features_mix, labels_mix)
    pred = regressor.predict_proba(features_mix)
    pred = pred[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(labels_mix, pred)
    auc_score = metrics.auc(fpr, tpr)

    return fpr, tpr, auc_score, thresholds
