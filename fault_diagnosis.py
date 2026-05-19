from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FAULT_TO_CLASS, PipelineConfig, resolve_device


def _soft_one_hot(label: int, confidence: float = 0.92) -> np.ndarray:
    p = np.full(6, (1.0 - confidence) / 5.0, dtype=float)
    p[int(label)] = confidence
    return p


def build_weak_label(window_id: int, total_seen: int, fault: str, peak: float, healthy_peak_ref: float,
                     cfg: PipelineConfig) -> Tuple[int, str]:
    if window_id <= cfg.healthy_reference_windows:
        return 0, "healthy_reference"

    if total_seen <= 2 * cfg.healthy_reference_windows:
        return 0, "insufficient_support"

    # 恢复基于元数据的真实故障类别注入
    target_label = FAULT_TO_CLASS.get(fault, 1)

    if healthy_peak_ref > 0 and peak > cfg.failure_multiplier * healthy_peak_ref:
        return target_label, "anomaly_detected"

    damage_ratio = peak / (cfg.failure_multiplier * healthy_peak_ref + 1e-12)
    if damage_ratio >= cfg.weak_label_degrade_ratio:
        return target_label, "degradation_warning"

    return 0, "pre_degradation"


def add_lag_prob_features(x: np.ndarray, probs: Sequence[np.ndarray]) -> np.ndarray:
    pri = np.vstack(probs) if probs else np.zeros((len(x), 6))
    if len(pri) < len(x):
        pad = np.tile(np.array([[1, 0, 0, 0, 0, 0]], dtype=float), (len(x) - len(pri), 1))
        pri = np.vstack([pri, pad])
    lag = np.vstack([np.array([[1, 0, 0, 0, 0, 0]], dtype=float), pri[:-1]])[: len(x)]
    return np.hstack([x, lag])


@dataclass
class DiagnosisResult:
    probs: np.ndarray
    label: int
    weak_label: int
    weak_reason: str
    shap_top: List[Tuple[str, float]]
    metrics: Dict[str, float]
    calibration: List[Tuple[float, float]]


@dataclass
class OnlineFaultDiagnoser:
    cfg: PipelineConfig
    feature_names: Optional[List[str]] = None
    x_hist: List[np.ndarray] = field(default_factory=list)
    y_weak: List[int] = field(default_factory=list)
    p_hist: List[np.ndarray] = field(default_factory=list)
    y_pred: List[int] = field(default_factory=list)
    healthy_peaks: List[float] = field(default_factory=list)
    _clf: object = None

    def __post_init__(self) -> None:
        try:
            import torch
            print("[TABPFN] CUDA available:", torch.cuda.is_available())
        except Exception:
            print("[TABPFN] torch not found")

        try:
            from tabpfn import TabPFNClassifier
            self._clf = TabPFNClassifier(
                model_path=str(self.cfg.tabpfn_classifier_model_path),
                device=resolve_device(self.cfg.device)
            )
            print("[TABPFN] ✅ Classifier loaded successfully")
            print("[TABPFN] model_path =", self.cfg.tabpfn_classifier_model_path)
        except Exception as e:
            self._clf = None
            print("[TABPFN] ❌ Classifier NOT loaded")
            print("[TABPFN] error:", e)

    def _predict_with_tabpfn(self, X: np.ndarray, y: np.ndarray, priors: List[np.ndarray]) -> np.ndarray:
        print("\n[PREDICT] window:", len(self.x_hist))
        print("[PREDICT] clf is None:", self._clf is None)
        print("[PREDICT] y[:-1] unique:", np.unique(y[:-1], return_counts=True))

        X2 = add_lag_prob_features(X, priors)

        # ✅ 关键修复：TabPFN 至少需要 2 条训练样本
        if self._clf is None or len(X2) < 3:
            print("[PREDICT] ⚠️ fallback (TabPFN needs ≥2 train samples)")
            return _soft_one_hot(int(y[-1]))

        # 第一轮
        self._clf.fit(X2[:-1], y[:-1])
        p1_raw = np.asarray(self._clf.predict_proba(X2[-1:]))[0]

        p1 = np.zeros(6, dtype=float)
        classes1 = getattr(self._clf, "classes_", np.arange(len(p1_raw)))
        for c, val in zip(classes1, p1_raw):
            p1[int(c)] = float(val)
        p1 /= max(p1.sum(), 1e-12)

        # 第二轮
        pri2 = list(priors[:-1]) + [p1]
        X3 = add_lag_prob_features(X, pri2)

        self._clf.fit(X3[:-1], y[:-1])
        p2_raw = np.asarray(self._clf.predict_proba(X3[-1:]))[0]

        out = np.zeros(6, dtype=float)
        classes = getattr(self._clf, "classes_", np.arange(len(p2_raw)))
        for c, val in zip(classes, p2_raw):
            out[int(c)] = float(val)

        out /= max(out.sum(), 1e-12)
        return out

    def _shap_top(self, x: np.ndarray, p: np.ndarray) -> List[Tuple[str, float]]:
        names = self.feature_names or [f"f{i}" for i in range(len(x))]
        if self._clf is not None and len(self.x_hist) >= 3:
            try:
                import shap  # type: ignore
                X = np.vstack(self.x_hist)
                bg = X[max(0, len(X) - self.cfg.shap_background):]
                target = int(np.argmax(p))

                def f(z: np.ndarray) -> np.ndarray:
                    pri = self.p_hist[-len(z):] if self.p_hist else [_soft_one_hot(0)] * len(z)
                    zz = add_lag_prob_features(np.asarray(z, dtype=float), pri)
                    pp = np.asarray(self._clf.predict_proba(zz))
                    classes = getattr(self._clf, "classes_", np.arange(pp.shape[1]))
                    out = np.zeros((len(z), 6), dtype=float)
                    for j, c in enumerate(classes):
                        out[:, int(c)] = pp[:, j]
                    return out[:, target]

                explainer = shap.KernelExplainer(f, bg)
                vals = np.asarray(explainer.shap_values(x.reshape(1, -1), nsamples=self.cfg.shap_nsamples))[0]
                idx = np.argsort(np.abs(vals))[::-1][: self.cfg.shap_top_k]
                return [(names[i], float(vals[i])) for i in idx]
            except Exception:
                pass
        # Fast causal proxy if SHAP/TabPFN is unavailable; used only for explanation/HI weights, never for probability.
        vals = (x - np.nanmean(np.vstack(self.x_hist), axis=0)) if len(self.x_hist) > 1 else x * 0
        vals = vals / (np.nanstd(np.vstack(self.x_hist), axis=0) + 1e-8) * float(np.max(p))
        idx = np.argsort(np.abs(vals))[::-1][: self.cfg.shap_top_k]
        return [(names[i], float(vals[i])) for i in idx]

    def _metrics(self) -> Dict[str, float]:
        y = np.asarray(self.y_weak, dtype=int); yp = np.asarray(self.y_pred, dtype=int)
        acc = float(np.mean(y == yp)) if len(y) else 0.0
        f1s = []
        for c in range(6):
            tp = np.sum((y == c) & (yp == c)); fp = np.sum((y != c) & (yp == c)); fn = np.sum((y == c) & (yp != c))
            f1s.append(float(2*tp / max(2*tp + fp + fn, 1)))
        weights = np.array([np.mean(y == c) for c in range(6)]) if len(y) else np.zeros(6)
        brier = float(np.mean([np.sum((p - np.eye(6)[yy])**2) for p, yy in zip(self.p_hist, y)])) if len(y) else 0.0
        cm = [[int(np.sum((y == i) & (yp == j))) for j in range(6)] for i in range(6)]
        return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "weighted_f1": float(np.sum(weights * f1s)), "brier": brier, "confusion_matrix": cm}

    def update(self, feature_vector: np.ndarray, feature_names: List[str], window_id: int, fault: str, peak: float) -> DiagnosisResult:
        self.feature_names = feature_names
        if window_id <= self.cfg.healthy_reference_windows:
            self.healthy_peaks.append(float(peak))
        href = float(np.percentile(self.healthy_peaks, 95)) if self.healthy_peaks else float(peak)
        weak, reason = build_weak_label(window_id, len(self.x_hist) + 1, fault, peak, href, self.cfg)
        self.x_hist.append(np.asarray(feature_vector, dtype=float)); self.y_weak.append(int(weak))
        X = np.vstack(self.x_hist); y = np.asarray(self.y_weak, dtype=int)
        priors = self.p_hist + [_soft_one_hot(weak)]
        p = self._predict_with_tabpfn(X, y, priors)
        label = int(np.argmax(p)); self.p_hist.append(p); self.y_pred.append(label)
        conf = np.max(np.vstack(self.p_hist), axis=1); correct = (np.asarray(self.y_pred) == y)
        calib = []
        for lo, hi in zip(np.linspace(0.0, 0.8, 5), np.linspace(0.2, 1.0, 5)):
            m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            calib.append((float((lo + hi) / 2), float(np.mean(correct[m])) if np.any(m) else float("nan")))
        return DiagnosisResult(p, label, weak, reason, self._shap_top(feature_vector, p), self._metrics(), calib)
