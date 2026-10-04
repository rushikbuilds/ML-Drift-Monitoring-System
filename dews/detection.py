from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from .config import MonitorConfig
from .data import FeatureSchema


def _safe_hist(values: np.ndarray, bins: Sequence[float] | int) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return np.full_like(hist, 1.0 / max(len(hist), 1), dtype=float)
    return np.clip(hist / total, 1e-12, 1.0)


def _normalize_raw_score(value: float, scale: float = 1.0) -> float:
    value = max(float(value), 0.0)
    return float(1.0 - np.exp(-value / max(scale, 1e-9)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p / p.sum(), 1e-12, 1.0)
    q = np.clip(q / q.sum(), 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def population_stability_index(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    edges = np.histogram_bin_edges(reference, bins=bins)
    reference_hist = _safe_hist(reference, edges)
    live_hist = _safe_hist(live, edges)
    return float(np.sum((live_hist - reference_hist) * np.log(live_hist / reference_hist)))


def kl_divergence(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    reference_hist = _safe_hist(reference, edges)
    live_hist = _safe_hist(live, edges)
    return float(stats.entropy(live_hist, reference_hist))


@dataclass(slots=True)
class FeatureDriftRecord:
    feature: str
    kind: str
    ks_statistic: float | None
    psi: float
    chi_square: float | None
    jsd: float
    score: float


@dataclass(slots=True)
class StatisticalDriftResult:
    per_feature: list[FeatureDriftRecord]
    s_stat: float
    mean_ks: float
    mean_psi: float
    mean_jsd: float


@dataclass(slots=True)
class EmbeddingDriftResult:
    centroid_distance: float
    mahalanobis_distance: float
    mmd: float
    s_emb: float


@dataclass(slots=True)
class ConfidenceDriftResult:
    low_confidence_rate: float
    reference_low_confidence_rate: float
    kl_divergence: float
    s_conf: float


class StatisticalDriftDetector:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config

    def evaluate(self, reference: pd.DataFrame, live: pd.DataFrame, schema: FeatureSchema) -> StatisticalDriftResult:
        records: list[FeatureDriftRecord] = []
        ks_values: list[float] = []
        psi_values: list[float] = []
        jsd_values: list[float] = []

        for feature in schema.continuous:
            reference_values = reference[feature].to_numpy(dtype=float)
            live_values = live[feature].to_numpy(dtype=float)
            ks_stat = float(stats.ks_2samp(reference_values, live_values).statistic)
            psi = population_stability_index(reference_values, live_values, bins=self.config.statistical_bins)
            edges = np.histogram_bin_edges(reference_values, bins=self.config.statistical_bins)
            reference_hist = _safe_hist(reference_values, edges)
            live_hist = _safe_hist(live_values, edges)
            jsd = jensen_shannon_divergence(reference_hist, live_hist)
            score = float(np.mean([
                _normalize_raw_score(ks_stat, 0.5),
                _normalize_raw_score(psi, 0.5),
                _normalize_raw_score(jsd, 0.25),
            ]))
            records.append(FeatureDriftRecord(feature, "continuous", ks_stat, psi, None, jsd, score))
            ks_values.append(ks_stat)
            psi_values.append(psi)
            jsd_values.append(jsd)

        for feature in schema.categorical:
            reference_values = reference[feature].astype(int).to_numpy()
            live_values = live[feature].astype(int).to_numpy()
            categories = np.union1d(reference_values, live_values)
            ref_counts = np.array([(reference_values == category).sum() for category in categories], dtype=float)
            live_counts = np.array([(live_values == category).sum() for category in categories], dtype=float)
            contingency = np.vstack([ref_counts, live_counts])
            chi_square = float(stats.chi2_contingency(contingency, correction=False)[0])
            psi = population_stability_index(reference_values.astype(float), live_values.astype(float), bins=min(len(categories), self.config.statistical_bins))
            ref_hist = np.clip(ref_counts / max(ref_counts.sum(), 1.0), 1e-12, 1.0)
            live_hist = np.clip(live_counts / max(live_counts.sum(), 1.0), 1e-12, 1.0)
            jsd = jensen_shannon_divergence(ref_hist, live_hist)
            score = float(np.mean([
                _normalize_raw_score(chi_square, 1.0),
                _normalize_raw_score(psi, 0.5),
                _normalize_raw_score(jsd, 0.25),
            ]))
            records.append(FeatureDriftRecord(feature, "categorical", None, psi, chi_square, jsd, score))
            psi_values.append(psi)
            jsd_values.append(jsd)

        mean_score = float(np.mean([record.score for record in records])) if records else 0.0
        return StatisticalDriftResult(
            per_feature=records,
            s_stat=mean_score,
            mean_ks=float(np.mean(ks_values)) if ks_values else 0.0,
            mean_psi=float(np.mean(psi_values)) if psi_values else 0.0,
            mean_jsd=float(np.mean(jsd_values)) if jsd_values else 0.0,
        )


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    distances = cdist(x, y, metric="sqeuclidean")
    return np.exp(-gamma * distances)


def maximum_mean_discrepancy(reference: np.ndarray, live: np.ndarray) -> float:
    combined = np.vstack([reference, live])
    pairwise_distances = cdist(combined, combined, metric="sqeuclidean")
    non_zero = pairwise_distances[pairwise_distances > 0]
    median_distance = np.median(non_zero) if non_zero.size else 1.0
    gamma = 1.0 / max(median_distance, 1e-6)

    k_xx = _rbf_kernel(reference, reference, gamma)
    k_yy = _rbf_kernel(live, live, gamma)
    k_xy = _rbf_kernel(reference, live, gamma)
    n = reference.shape[0]
    m = live.shape[0]
    term_xx = (k_xx.sum() - np.trace(k_xx)) / max(n * (n - 1), 1)
    term_yy = (k_yy.sum() - np.trace(k_yy)) / max(m * (m - 1), 1)
    term_xy = k_xy.mean()
    return float(max(term_xx + term_yy - 2.0 * term_xy, 0.0))


class EmbeddingDriftDetector:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.reference_embeddings: np.ndarray | None = None
        self.reference_covariance_inverse: np.ndarray | None = None
        self.reference_mean: np.ndarray | None = None

    def fit(self, reference: np.ndarray | pd.DataFrame) -> None:
        embeddings = self._to_embedding_array(reference)
        self.scaler.fit(embeddings)
        scaled_embeddings = self.scaler.transform(embeddings)
        self.reference_embeddings = scaled_embeddings
        self.reference_mean = scaled_embeddings.mean(axis=0)
        covariance = np.cov(scaled_embeddings, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0]) * 1e-6
        self.reference_covariance_inverse = np.linalg.pinv(covariance)

    def evaluate(self, reference: np.ndarray | pd.DataFrame, live: np.ndarray | pd.DataFrame) -> EmbeddingDriftResult:
        if self.reference_embeddings is None or self.reference_covariance_inverse is None or self.reference_mean is None:
            self.fit(reference)
        assert self.reference_embeddings is not None
        assert self.reference_covariance_inverse is not None
        assert self.reference_mean is not None

        live_embeddings = self.scaler.transform(self._to_embedding_array(live))
        live_mean = live_embeddings.mean(axis=0)
        delta = live_mean - self.reference_mean
        centroid_distance = float(np.linalg.norm(delta))
        mahalanobis_distance = float(np.sqrt(delta.T @ self.reference_covariance_inverse @ delta))
        mmd = float(maximum_mean_discrepancy(self.reference_embeddings, live_embeddings))
        s_emb = float(np.mean([
            _normalize_raw_score(centroid_distance, 1.0),
            _normalize_raw_score(mahalanobis_distance, 1.0),
            _normalize_raw_score(mmd, 0.5),
        ]))
        return EmbeddingDriftResult(
            centroid_distance=centroid_distance,
            mahalanobis_distance=mahalanobis_distance,
            mmd=mmd,
            s_emb=s_emb,
        )

    def _to_embedding_array(self, frame: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(frame, pd.DataFrame):
            array = frame.to_numpy(dtype=float)
        else:
            array = np.asarray(frame, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array


class ConfidenceDriftDetector:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.reference_distribution: np.ndarray | None = None
        self.reference_low_confidence_rate: float = 0.0

    def fit(self, model, reference: pd.DataFrame) -> None:
        probabilities = model.predict_proba(reference)
        confidence = probabilities.max(axis=1)
        self.reference_distribution = confidence
        self.reference_low_confidence_rate = float(np.mean(confidence < self.config.low_confidence_threshold))

    def evaluate(self, model, reference: pd.DataFrame, live: pd.DataFrame) -> ConfidenceDriftResult:
        if self.reference_distribution is None:
            self.fit(model, reference)
        assert self.reference_distribution is not None

        live_probabilities = model.predict_proba(live)
        live_confidence = live_probabilities.max(axis=1)
        low_confidence_rate = float(np.mean(live_confidence < self.config.low_confidence_threshold))
        kl = kl_divergence(self.reference_distribution, live_confidence, bins=self.config.confidence_bins)
        s_conf = float(np.mean([
            _normalize_raw_score(abs(low_confidence_rate - self.reference_low_confidence_rate), 1.0),
            _normalize_raw_score(kl, 0.75),
        ]))
        return ConfidenceDriftResult(
            low_confidence_rate=low_confidence_rate,
            reference_low_confidence_rate=self.reference_low_confidence_rate,
            kl_divergence=kl,
            s_conf=s_conf,
        )
