from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .config import BEARING_CONFIG, PipelineConfig
    from .data_loader import iter_windows
    from .feature_engineering import extract_features
except ImportError:  # pragma: no cover
    from config import BEARING_CONFIG, PipelineConfig
    from data_loader import iter_windows
    from feature_engineering import extract_features


def _health_stage_from_ratio(ratio: float, cfg: PipelineConfig) -> int:
    t1, t2 = cfg.stage_thresholds
    if ratio < t1:
        return 0
    if ratio < t2:
        return 1
    return 2


def _add_time_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    t = out["file_index"].astype(float)
    out["t_index"] = t
    out["t_sqrt"] = np.sqrt(t)
    out["t_log1p"] = np.log1p(t)
    for period in cfg.calendar_periods:
        omega = 2.0 * np.pi / float(period)
        out[f"cal_sin_{period}"] = np.sin(omega * t)
        out[f"cal_cos_{period}"] = np.cos(omega * t)
    return out


def _add_lag_roll_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    for base_col in cfg.lag_feature_bases:
        if base_col not in out.columns:
            continue
        s = out[base_col].astype(float)
        for lag in cfg.lag_steps:
            out[f"{base_col}_lag{lag}"] = s.shift(lag).fillna(0.0)
            out[f"{base_col}_diff{lag}"] = (s - s.shift(lag)).fillna(0.0)
        out[f"{base_col}_roll5_mean"] = s.shift(1).rolling(5, min_periods=1).mean().fillna(0.0)
        out[f"{base_col}_roll5_std"] = s.shift(1).rolling(5, min_periods=1).std().fillna(0.0)
    return out


def build_bearing_dataframe(bearing_id: str, cfg: PipelineConfig, max_windows: int = 0) -> pd.DataFrame:
    prev_base = None
    rows: List[Dict[str, float]] = []

    for record in iter_windows(bearing_id, cfg):
        if max_windows and record.window_id > max_windows:
            break
        feats, prev_base = extract_features(record, cfg, prev_base)
        rows.append({
            "bearing_id": bearing_id,
            "file_index": int(record.window_id),
            "fault": str(record.fault),
            **{k: float(v) for k, v in feats.items() if np.isscalar(v)},
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("file_index").reset_index(drop=True)
    if "rms_combined" not in df.columns:
        return pd.DataFrame()
    baseline_count = min(max(1, cfg.baseline_windows), len(df))
    baseline_rms = float(df.loc[: baseline_count - 1, "rms_combined"].mean())
    if baseline_rms <= 1e-12:
        baseline_rms = 1.0
    df["baseline_rms"] = baseline_rms
    df["rms_over_baseline"] = (df["rms_combined"] / baseline_rms).astype(float)
    df["health_stage"] = df["rms_over_baseline"].map(lambda x: _health_stage_from_ratio(float(x), cfg)).astype(int)
    df["is_fault"] = (df["health_stage"] > 0).astype(int)

    df = _add_time_features(df, cfg)
    df = _add_lag_roll_features(df, cfg)

    horizon = max(1, int(cfg.forecast_horizon))
    df["future_health_stage"] = df["health_stage"].shift(-horizon)
    df = df.iloc[:-horizon] if len(df) > horizon else df.iloc[0:0]
    if not df.empty:
        df["future_health_stage"] = df["future_health_stage"].astype(int)

    return df.reset_index(drop=True)


def build_all_bearings(cfg: PipelineConfig, bearings: List[str], max_windows: int = 0) -> Dict[str, pd.DataFrame]:
    return {bearing_id: build_bearing_dataframe(bearing_id, cfg, max_windows=max_windows) for bearing_id in bearings if bearing_id in BEARING_CONFIG}
