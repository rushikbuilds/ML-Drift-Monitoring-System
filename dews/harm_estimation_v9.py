"""v9 candidate harm estimators (experiment branch; see docs/v9_protocol.md).

Both estimate a batch's accuracy without its labels, using the labeled reference set:

* ``gbm_iw_accuracy``: importance-weighted reference accuracy with a nonlinear
  (gradient-boosting) domain classifier instead of the linear one in v8.
* ``error_model_accuracy``: an error predictor. The monitored model's correctness on
  reference rows is estimated out-of-fold (5-fold refits of the same model spec), then a
  gradient-boosting classifier learns P(wrong | features, model confidence). A batch's
  estimated accuracy is 1 minus the mean predicted error over its rows.

Out-of-fold correctness avoids scoring the model on rows it was trained on.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

WEIGHT_CAP = 20.0


def _confidence(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X)).max(axis=1)


def out_of_fold_correctness(model, reference, labels, seed: int = 0) -> np.ndarray:
    X, y = np.asarray(reference, float), np.asarray(labels)
    correct = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = clone(model).fit(reference.iloc[tr], y[tr])
        correct[te] = (m.predict(reference.iloc[te]) == y[te]).astype(float)
    return correct


def harm_signals_v9(reference, reference_labels, model, batches, seed: int = 0) -> dict[str, np.ndarray]:
    ref = np.asarray(reference, float)
    correct = out_of_fold_correctness(model, reference, reference_labels, seed)

    error_model = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05, random_state=seed)
    ref_meta = np.column_stack([ref, _confidence(model, reference)])
    error_model.fit(ref_meta, 1 - correct)

    gbm_iw, err = [], []
    for batch in batches:
        live = np.asarray(batch, float)
        live_meta = np.column_stack([live, _confidence(model, batch)])
        err.append(1.0 - float(error_model.predict_proba(live_meta)[:, 1].mean()))
        dom = HistGradientBoostingClassifier(max_iter=40, max_depth=3, random_state=seed)
        dom.fit(np.vstack([ref, live]), np.r_[np.zeros(len(ref)), np.ones(len(live))])
        p = dom.predict_proba(ref)[:, 1]
        w = np.clip(p / np.clip(1 - p, 1e-6, None), 0, None)
        w = np.minimum(w / w.mean(), WEIGHT_CAP)
        gbm_iw.append(float(np.sum(w * correct) / np.sum(w)))
    return {"gbm_iw_accuracy": np.asarray(gbm_iw), "error_model_accuracy": np.asarray(err)}
