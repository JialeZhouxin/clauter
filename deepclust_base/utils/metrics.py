from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    matched = sum(w[r, c] for r, c in zip(row_ind, col_ind))
    return matched / y_pred.size
