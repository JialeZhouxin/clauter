from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..utils.metrics import clustering_accuracy


@dataclass
class EvalResult:
    nmi: float
    ari: float
    acc: float


class Evaluator:
    def __init__(self, num_clusters: int, random_state: int = 42):
        self.num_clusters = num_clusters
        self.random_state = random_state

    def evaluate_embeddings(self, z: np.ndarray, y_true: Optional[np.ndarray]) -> Dict[str, float]:
        pred = KMeans(n_clusters=self.num_clusters, n_init=20, random_state=self.random_state).fit_predict(z)
        return self.evaluate_labels(pred, y_true)

    def evaluate_labels(self, y_pred: np.ndarray, y_true: Optional[np.ndarray]) -> Dict[str, float]:
        if y_true is None:
            return {"nmi": float("nan"), "ari": float("nan"), "acc": float("nan")}

        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        acc = clustering_accuracy(y_true, y_pred)
        return {"nmi": float(nmi), "ari": float(ari), "acc": float(acc)}
