from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from .config import PipelineConfig, resolve_device
except ImportError:  # pragma: no cover
    from config import PipelineConfig, resolve_device


@dataclass
class RULPredictor:
    cfg: PipelineConfig
    hi_raw: List[float] = field(default_factory=list)
    hi_mono: List[float] = field(default_factory=list)
    _reg: object = None

    def __post_init__(self) -> None:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore
            self._reg = TabPFNRegressor(model_path=str(self.cfg.tabpfn_regressor_model_path), device=resolve_device(self.cfg.device))
        except Exception:
            self._reg = None

    def update(self, hi: float) -> Dict[str, object]:
        self.hi_raw.append(float(hi))
        prev = self.hi_mono[-1] if self.hi_mono else float(hi)
        alpha = self.cfg.rul_ema_alpha
        smooth = alpha * float(hi) + (1 - alpha) * prev
        mono = min(prev, smooth) if self.hi_mono else smooth
        self.hi_mono.append(float(mono))
        y = np.array(self.hi_mono[-self.cfg.rul_recent_points:], dtype=float)

        if mono <= self.cfg.tau_fail:
            return {"rul": 1, "ci_low": 1, "ci_high": 1, "method": "already_failed", "future_hi": [mono]}

        if len(y) < 2:
            rul = self.cfg.rul_max_steps; method = "insufficient_history"; sigma = 0.0; beta_val = -1.0; pred = np.repeat(mono, self.cfg.rul_max_steps)
        else:
            t = np.arange(len(y), dtype=float)
            beta_fit, alpha_fit = np.polyfit(t, y, 1)
            resid_lin = y - (alpha_fit + beta_fit * t)
            sigma = float(np.std(resid_lin)) if len(resid_lin) > 1 else 0.0
            pred = alpha_fit + beta_fit * np.arange(len(y), len(y) + self.cfg.rul_max_steps)
            beta_val = beta_fit

            # ---- 尝试 TabPFN 回归器预测 ----
            if self._reg is not None and len(self.hi_mono) >= 4:
                try:
                    t_all = np.arange(len(self.hi_mono), dtype=float).reshape(-1, 1)
                    y_all = np.array(self.hi_mono, dtype=float)
                    self._reg.fit(t_all, y_all)  # type: ignore[union-attr]
                    t_future = np.arange(len(self.hi_mono), len(self.hi_mono) + self.cfg.rul_max_steps, dtype=float).reshape(-1, 1)
                    pred_tabpfn = np.asarray(self._reg.predict(t_future), dtype=float).reshape(-1)  # type: ignore[union-attr]
                    cross_tabpfn = np.where(pred_tabpfn <= self.cfg.tau_fail)[0]
                    if len(cross_tabpfn):
                        rul = int(cross_tabpfn[0] + 1)
                    else:
                        rul = self.cfg.rul_max_steps
                    # 用训练集残差估计 sigma
                    y_train_pred = np.asarray(self._reg.predict(t_all), dtype=float).reshape(-1)  # type: ignore[union-attr]
                    sigma = float(np.std(y_all - y_train_pred)) if len(y_all) > 1 else sigma
                    method = "tabpfn_forecast"
                    pred = pred_tabpfn
                    beta_val = -(sigma / max(mono, 1e-6))  # 供 CI 计算使用
                except Exception:
                    # TabPFN 失败，回退到线性/指数
                    rul, method, sigma, beta_val, pred = self._fallback_predict(y, mono, sigma)
            else:
                rul, method, sigma, beta_val, pred = self._fallback_predict(y, mono, sigma)

        rul = int(np.clip(rul, 1, self.cfg.rul_max_steps))
        half = self.cfg.rul_ci_z * sigma / max(abs(float(beta_val)), 1e-6)
        return {"rul": rul, "ci_low": int(np.clip(np.floor(rul - half), 1, self.cfg.rul_max_steps)), "ci_high": int(np.clip(np.ceil(rul + half), 1, self.cfg.rul_max_steps)), "method": method, "future_hi": [float(v) for v in pred[: self.cfg.rul_max_steps]]}

    def _fallback_predict(self, y: np.ndarray, mono: float, sigma_lin: float):
        """线性/指数回退预测，sigma 使用残差估计。"""
        t = np.arange(len(y), dtype=float)
        beta_fit, alpha_fit = np.polyfit(t, y, 1)
        cross = np.where((alpha_fit + beta_fit * np.arange(len(y), len(y) + self.cfg.rul_max_steps)) <= self.cfg.tau_fail)[0]

        if len(cross):
            rul = int(cross[0] + 1); method = "linear_cross"
            pred = alpha_fit + beta_fit * np.arange(len(y), len(y) + self.cfg.rul_max_steps)
            resid = y - (alpha_fit + beta_fit * t)
            sigma = float(np.std(resid)) if len(resid) > 1 else sigma_lin
            return rul, method, sigma, beta_fit, pred

        if beta_fit < -1e-6:
            rul = int(np.ceil((self.cfg.tau_fail - mono) / beta_fit)); method = "linear_extrapolate"
            pred = alpha_fit + beta_fit * np.arange(len(y), len(y) + self.cfg.rul_max_steps)
            resid = y - (alpha_fit + beta_fit * t)
            sigma = float(np.std(resid)) if len(resid) > 1 else sigma_lin
            return rul, method, sigma, beta_fit, pred

        # 指数衰减：用 polyfit 拟合 log(y) 以获得更准确的 decay 和残差
        log_y = np.log(np.maximum(y, 1e-6))
        log_beta, log_alpha = np.polyfit(t, log_y, 1)
        decay = max(1e-4, -log_beta)
        rul = int(np.ceil(np.log(self.cfg.tau_fail / mono) / -decay)); method = "exponential_decay"
        pred = mono * np.exp(-decay * np.arange(1, self.cfg.rul_max_steps + 1))
        # 从指数拟合残差估计 sigma
        y_fit = np.exp(log_alpha + log_beta * t)
        resid_exp = y - y_fit
        sigma = float(np.std(resid_exp)) if len(resid_exp) > 1 else sigma_lin
        beta_val = -decay * mono
        return rul, method, sigma, beta_val, pred
