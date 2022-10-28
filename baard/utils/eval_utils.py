"""Utility functions for evaluation metrics."""
import numpy as np


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
