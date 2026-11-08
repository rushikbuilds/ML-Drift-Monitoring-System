"""Label-free harm estimation from the labeled reference set (v8).

A shift is harmful when it moves data into regions where the monitored model makes
more mistakes. The labeled reference rows show where those regions are. For each live
batch we estimate the model's accuracy without new labels:

* ``knn_accuracy``: for every live row, the share of its k nearest reference rows (in
  reference-standardized feature space) that the model classifies correctly; averaged.
* ``iw_accuracy``: importance-weighted reference accuracy. A domain classifier separates
  reference rows from the live batch; its odds weight each reference row's correctness
  (self-normalized, weights capped).

Under pure covariate shift (P(y | x) unchanged) both target the true batch accuracy.
Neither can see concept change, which needs new labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

K_NEIGHBORS = 25
WEIGHT_CAP = 20.0


def harm_signals(reference, reference_labels, model, batches, *, k: int = K_NEIGHBORS) -> dict[str, np.ndarray]:
    ref = np.asarray(reference, float)
    correct = (np.asarray(model.predict(reference)) == np.asarray(reference_labels)).astype(float)
    scaler = StandardScaler().fit(ref)
    ref_z = scaler.transform(ref)
    nn = NearestNeighbors(n_neighbors=k).fit(ref_z)
    knn, iw = [], []
    for batch in batches:
        live_z = scaler.transform(np.asarray(batch, float))
        _, idx = nn.kneighbors(live_z)
        knn.append(float(correct[idx].mean()))
        X = np.vstack([ref_z, live_z])
        y = np.r_[np.zeros(len(ref_z)), np.ones(len(live_z))]
        p = LogisticRegression(C=1.0, max_iter=300).fit(X, y).predict_proba(ref_z)[:, 1]
        w = np.clip(p / np.clip(1 - p, 1e-6, None), 0, None)
        w = np.minimum(w / w.mean(), WEIGHT_CAP)
        iw.append(float(np.sum(w * correct) / np.sum(w)))
    return {"knn_accuracy": np.asarray(knn), "iw_accuracy": np.asarray(iw)}


def estimated_drop(acc_estimate: np.ndarray, burn_in: int) -> np.ndarray:
    """Estimated accuracy drop relative to the burn-in median, in accuracy units; NaN during burn-in."""
    est = np.asarray(acc_estimate, float)
    drop = np.median(est[:burn_in]) - est
    drop[:burn_in] = np.nan
    return drop
