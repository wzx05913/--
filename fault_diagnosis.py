from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from .config import PipelineConfig, resolve_device
except ImportError:  # pragma: no cover
    from config import PipelineConfig, resolve_device


@dataclass
class OfflineFaultDiagnoser:
    cfg: PipelineConfig
    _clf: object = None
    _kind: str = "fallback"

    def __post_init__(self) -> None:
        try:
            from tabpfn import TabPFNClassifier  # type: ignore

            self._clf = TabPFNClassifier(
                model_path=str(self.cfg.tabpfn_classifier_model_path),
                device=resolve_device(self.cfg.device),
            )
            self._kind = "tabpfn"
            return
        except Exception:
            pass

        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore

            self._clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, multi_class="auto", random_state=self.cfg.random_seed)),
            ])
            self._kind = "logreg"
            return
        except Exception:
            self._clf = None
            self._kind = "fallback"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        if self._clf is None:
            return
        self._clf.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=int))

    def predict_proba(self, x_test: np.ndarray, n_classes: int = 3) -> np.ndarray:
        x_test = np.asarray(x_test, dtype=float)
        if len(x_test) == 0:
            return np.zeros((0, n_classes), dtype=float)

        if self._clf is None:
            out = np.zeros((len(x_test), n_classes), dtype=float)
            out[:, 0] = 1.0
            return out

        raw = np.asarray(self._clf.predict_proba(x_test), dtype=float)
        classes = getattr(self._clf, "classes_", np.arange(raw.shape[1]))
        out = np.zeros((len(x_test), n_classes), dtype=float)
        for j, c in enumerate(classes):
            if 0 <= int(c) < n_classes:
                out[:, int(c)] = raw[:, j]
        sums = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        return out / sums

    @property
    def model_kind(self) -> str:
        return self._kind


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "support": 0, "confusion_matrix": [[0] * n_classes for _ in range(n_classes)]}

    acc = float(np.mean(y_true == y_pred))
    f1s: List[float] = []
    weights: List[float] = []
    cm: List[List[int]] = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else 0.0)
        weights.append(float(np.mean(y_true == c)))
        cm.append([int(np.sum((y_true == c) & (y_pred == j))) for j in range(n_classes)])

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.asarray(weights) * np.asarray(f1s))),
        "support": int(len(y_true)),
        "confusion_matrix": cm,
    }


def top_feature_effects(row: np.ndarray, feature_names: List[str], top_k: int = 8) -> List[Tuple[str, float]]:
    if len(row) == 0:
        return []
    idx = np.argsort(np.abs(row))[::-1][:top_k]
    return [(feature_names[i], float(row[i])) for i in idx]
