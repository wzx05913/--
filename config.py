from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

BEARING_GEOMETRY = {"n_b": 8, "d_mm": 7.92, "D_mm": 34.55, "alpha_deg": 0.0}
SAMPLE_RATE_HZ = 25600.0
CSV_ROWS = 32768
CHANNEL_NAMES = ("horizontal", "vertical")

BEARING_CONFIG: Dict[str, Dict[str, object]] = {
    "Bearing1_1": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Outer race"},
    "Bearing1_2": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Outer race"},
    "Bearing1_3": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Outer race"},
    "Bearing1_4": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Cage"},
    "Bearing1_5": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Mixed"},
    "Bearing2_1": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Inner race"},
    "Bearing2_2": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Outer race"},
    "Bearing2_3": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Cage"},
    "Bearing2_4": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Outer race"},
    "Bearing2_5": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Outer race"},
    "Bearing3_1": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Outer race"},
    "Bearing3_2": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Mixed"},
    "Bearing3_3": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Inner race"},
    "Bearing3_4": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Inner race"},
    "Bearing3_5": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Outer race"},
}
CONDITION_DIRS = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
FAULT_LABELS = {0: "非常健康", 1: "退化预警", 2: "故障退化"}


@dataclass
class PipelineConfig:
    data_root: Path = Path("D:/Desktop/数据集实验/西交数据集/data/XJTU-SY_Bearing_Datasets")
    output_dir: Path = Path("output_v2")
    sampling_rate_hz: float = SAMPLE_RATE_HZ
    csv_rows: int = CSV_ROWS
    p: float = 1.0
    wavelet: str = "db4"
    wavelet_level: int = 4
    healthy_quantile: float = 0.95
    failure_quantile: float = 0.05
    hi_lambda: float = 3.0
    hi_epsilon: float = 1e-8
    hi_ema_alpha: float = 0.30
    rul_ema_alpha: float = 0.15
    hi_thresholds: Tuple[float, float, float] = (0.75, 0.50, 0.25)
    tau_fail: float = 0.25
    rul_max_steps: int = 200
    rul_recent_points: int = 8
    rul_ci_z: float = 1.96

    baseline_windows: int = 50
    forecast_horizon: int = 10
    # Conservative defaults from observed XJTU-SY RMS escalation: <1.6 healthy, 1.6~3.0 warning, >=3.0 degraded.
    stage_thresholds: Tuple[float, float] = (1.6, 3.0)
    lag_steps: Tuple[int, ...] = (1, 2, 3, 5, 10)
    calendar_periods: Tuple[int, ...] = (5, 10, 20, 60, 120)
    lag_feature_bases: Tuple[str, ...] = (
        "rms_combined",
        "kurt_max",
        "horizontal_crest",
        "vertical_crest",
        "horizontal_spectral_entropy",
        "vertical_spectral_entropy",
        "rms_over_baseline",
    )

    # number of top features to emit in diagnosis CSV for explainability
    shap_top_k: int = 8

    tabpfn_regressor_model_path: Path = Path("model/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt")
    tabpfn_classifier_model_path: Path = Path("model/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt")
    device: Optional[str] = None
    random_seed: int = 42
    degradation_directions: Dict[str, str] = field(default_factory=lambda: {
        "horizontal_rms": "positive", "vertical_rms": "positive", "rms_combined": "positive",
        "horizontal_kurt": "positive", "vertical_kurt": "positive", "kurt_max": "positive",
        "horizontal_crest": "positive", "vertical_crest": "positive", "horizontal_impulse": "positive", "vertical_impulse": "positive",
        "horizontal_peak": "positive", "vertical_peak": "positive", "energyratio_hv": "positive", "rho_hv": "reverse",
        "rms_over_baseline": "positive",
    })

    def window_size(self) -> int:
        if not (0 < self.p <= 1):
            raise ValueError("p must be in (0, 1]")
        return max(1, int(math.floor(self.csv_rows * self.p)))

    def ensure_dirs(self) -> None:
        for sub in ("features", "diagnosis", "hi", "rul", "figures", "logs"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)


def compute_fault_freqs(speed_hz: float) -> dict:
    nb = BEARING_GEOMETRY["n_b"]
    d = BEARING_GEOMETRY["d_mm"]
    D = BEARING_GEOMETRY["D_mm"]
    alpha = math.radians(BEARING_GEOMETRY["alpha_deg"])
    lam = (d / D) * math.cos(alpha)
    return {
        "BPFO": nb / 2 * speed_hz * (1 - lam),
        "BPFI": nb / 2 * speed_hz * (1 + lam),
        "BSF": D / (2 * d) * speed_hz * (1 - lam**2),
        "FTF": 0.5 * speed_hz * (1 - lam),
    }


def resolve_device(device: Optional[str] = None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
