from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass(slots=True)
class FeatureSchema:
    continuous: list[str]
    categorical: list[str]


@dataclass(slots=True)
class DemoScenario:
    reference_frame: pd.DataFrame
    reference_labels: pd.Series
    stream_batches: list[pd.DataFrame]
    stream_labels: list[pd.Series]
    feature_schema: FeatureSchema
    drift_start_batch: int
    metadata: dict = field(default_factory=dict)


def _build_frame(features: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
    frame["segment"] = pd.qcut(frame["feature_0"], q=3, labels=[0, 1, 2]).astype(int)
    return frame


def _inject_drift(features: np.ndarray, batch_index: int) -> np.ndarray:
    drifted = features.copy()
    drift_factor = 0.15 + 0.09 * (batch_index + 1)
    drifted[:, 0] += drift_factor * 1.2
    drifted[:, 1] -= drift_factor * 0.8
    drifted[:, 2] += np.sin(batch_index / 1.5) * 0.6
    drifted[:, 3] *= 1.0 + drift_factor * 0.45
    if drifted.shape[1] > 4:
        noise = np.random.default_rng(1000 + batch_index).normal(0.0, 0.6 + drift_factor * 0.4, size=drifted[:, 4].shape)
        drifted[:, 4] += noise
    if drifted.shape[1] > 5:
        drifted[:, 5] = np.where(drifted[:, 5] > 0, drifted[:, 5] - drift_factor * 0.5, drifted[:, 5] + drift_factor * 0.5)
    return drifted


def build_demo_scenario(
    *,
    seed: int = 42,
    reference_size: int = 320,
    stream_batches: int = 18,
    batch_size: int = 80,
    inject_drift: bool = True,
) -> DemoScenario:
    rng = np.random.default_rng(seed)
    total_stream = stream_batches * batch_size
    total_samples = reference_size + total_stream

    features, labels = make_classification(
        n_samples=total_samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        class_sep=1.8,
        flip_y=0.01,
        random_state=seed,
    )

    frame = _build_frame(features)
    labels_series = pd.Series(labels, name="target")

    reference_frame = frame.iloc[:reference_size].reset_index(drop=True)
    reference_labels = labels_series.iloc[:reference_size].reset_index(drop=True)

    stream_features = frame.iloc[reference_size:].reset_index(drop=True)
    stream_labels = labels_series.iloc[reference_size:].reset_index(drop=True)

    drift_start_batch = max(4, stream_batches // 2)
    batches: list[pd.DataFrame] = []
    batch_labels: list[pd.Series] = []

    for batch_index in range(stream_batches):
        start = batch_index * batch_size
        end = start + batch_size
        batch_frame = stream_features.iloc[start:end].copy().reset_index(drop=True)
        batch_target = stream_labels.iloc[start:end].copy().reset_index(drop=True)

        if inject_drift and batch_index >= drift_start_batch:
            drifted_numeric = _inject_drift(batch_frame[[f"feature_{index}" for index in range(6)]].to_numpy(), batch_index - drift_start_batch)
            drifted = pd.DataFrame(drifted_numeric, columns=[f"feature_{index}" for index in range(6)])
            drifted["segment"] = pd.qcut(drifted["feature_0"], q=3, labels=[0, 1, 2], duplicates="drop").astype(int)
            batch_frame = drifted
            batch_target = pd.Series((batch_target.to_numpy() ^ (rng.random(len(batch_target)) > 0.85)).astype(int), name="target")

        batches.append(batch_frame)
        batch_labels.append(batch_target)

    return DemoScenario(
        reference_frame=reference_frame,
        reference_labels=reference_labels,
        stream_batches=batches,
        stream_labels=batch_labels,
        feature_schema=FeatureSchema(
            continuous=[f"feature_{index}" for index in range(6)],
            categorical=["segment"],
        ),
        drift_start_batch=drift_start_batch,
        metadata={
            "seed": seed,
            "reference_size": reference_size,
            "stream_batches": stream_batches,
            "batch_size": batch_size,
            "inject_drift": inject_drift,
        },
    )
