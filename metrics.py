import numpy as np
from sklearn import metrics

# AUPR ERROR
def calc_fpr_aupr(confidence, correct):
    correctness = np.array(correct)
    fpr, tpr, _ = metrics.roc_curve(correctness, confidence)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= .95)]

    precision, recall, _ = metrics.precision_recall_curve(correctness, confidence)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * confidence)

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95