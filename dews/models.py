from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    def fit_reference(self, reference_frame: pd.DataFrame) -> None: ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...

    def embedding_features(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass(slots=True)
class SklearnModelAdapter:
    model: object

    def fit_reference(self, reference_frame: pd.DataFrame) -> None:
        return None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The wrapped model does not expose predict_proba")
        return np.asarray(self.model.predict_proba(frame))

    def embedding_features(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(frame.to_numpy(dtype=float))


@dataclass(slots=True)
class TorchMLPAdapter:
    model: object
    feature_columns: list[str]
    device: str = "cpu"
    embedding_fn: Callable[[pd.DataFrame], np.ndarray] | None = None
    preprocessor_fn: Callable[[pd.DataFrame], np.ndarray] | None = None

    def fit_reference(self, reference_frame: pd.DataFrame) -> None:
        return None

    def _tensor(self, frame: pd.DataFrame):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError("PyTorch is required for TorchMLPAdapter") from exc

        if self.preprocessor_fn is not None:
            array = self.preprocessor_fn(frame)
        else:
            array = frame[self.feature_columns].to_numpy(dtype=np.float32)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _forward(self, frame: pd.DataFrame):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError("PyTorch is required for TorchMLPAdapter") from exc

        tensor = self._tensor(frame)
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError("PyTorch is required for TorchMLPAdapter") from exc

        logits = self._forward(frame)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(frame)
        return np.asarray(np.argmax(probabilities, axis=1))

    def embedding_features(self, frame: pd.DataFrame) -> np.ndarray:
        if self.embedding_fn is not None:
            return np.asarray(self.embedding_fn(frame))
        return np.asarray(self._forward(frame).detach().cpu().numpy())


def build_model_adapter(model: object, feature_columns: list[str], embedding_fn: Callable[[pd.DataFrame], np.ndarray] | None = None) -> ModelAdapter:
    if isinstance(model, SklearnModelAdapter):
        return model
    if hasattr(model, "predict_proba") and hasattr(model, "predict"):
        return SklearnModelAdapter(model=model)
    return TorchMLPAdapter(model=model, feature_columns=feature_columns, embedding_fn=embedding_fn)
