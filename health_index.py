from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig


@dataclass
class HealthIndexBuilder:
    cfg: PipelineConfig
    history: List[Dict[str, float]] = field(default_factory=list)
    last_damage: Optional[float] = None

    def _selected_features(self, row: Dict[str, float]) -> List[str]:
        return [k for k in self.cfg.degradation_directions if k in row]

    def update(self, row: Dict[str, float], shap_top: List[Tuple[str, float]]) -> Dict[str, object]:
        self.history.append(row)
        feats = self._selected_features(row)
        arr = {k: np.array([h[k] for h in self.history if k in h], dtype=float) for k in feats}
        healthy = self.history[: max(1, min(len(self.history), self.cfg.healthy_reference_windows))]
        recent = self.history[-max(1, min(len(self.history), self.cfg.healthy_reference_windows)):]
        shap_abs = {k: 0.0 for k in feats}
        for name, val in shap_top:
            base = name[6:] if name.startswith("delta_") else name
            if base in shap_abs:
                # 取 max 而非累加，避免 base/delta 同名特征双重计数
                shap_abs[base] = max(shap_abs[base], abs(float(val)))
        ssum = sum(shap_abs.values())
        weights = {k: (shap_abs[k] / ssum if ssum > 0 else 1.0 / max(len(feats), 1)) for k in feats}
        damages = {}
        for k in feats:
            hvals = np.array([r[k] for r in healthy if k in r], dtype=float)
            rvals = np.array([r[k] for r in recent if k in r], dtype=float)
            qh = float(np.quantile(hvals, self.cfg.healthy_quantile)) if len(hvals) else row[k]
            direction = self.cfg.degradation_directions.get(k, "positive")
            if direction == "reverse":
                qf = float(np.quantile(rvals, self.cfg.failure_quantile)) if len(rvals) else row[k]
                d = (qh - row[k]) / (qh - qf + self.cfg.hi_epsilon)
            else:
                # 正方向退化：取 recent 高分位数作为退化参考上界
                qf = float(np.quantile(rvals, 1.0 - self.cfg.failure_quantile)) if len(rvals) else row[k]
                if qf <= qh:
                    qf = max(float(np.max(arr[k])), qh + self.cfg.hi_epsilon)
                d = (row[k] - qh) / (qf - qh + self.cfg.hi_epsilon)
            damages[k] = float(np.clip(d, 0.0, 1.0))
        D = float(sum(weights[k] * damages[k] for k in feats)) if feats else 0.0
        if self.last_damage is not None:
            D = self.cfg.hi_ema_alpha * D + (1 - self.cfg.hi_ema_alpha) * self.last_damage
        self.last_damage = D
        hi = float(np.exp(-self.cfg.hi_lambda * D))
        t0, t1, t2 = self.cfg.hi_thresholds
        level = 0 if hi >= t0 else 1 if hi >= t1 else 2 if hi >= t2 else 3
        return {"HI": hi, "D": D, "level": level, "weights": weights, "damages": damages}
