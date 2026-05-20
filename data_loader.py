from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd

try:
    from .config import BEARING_CONFIG, CONDITION_DIRS, PipelineConfig
except ImportError:  # pragma: no cover
    from config import BEARING_CONFIG, CONDITION_DIRS, PipelineConfig


@dataclass
class WindowRecord:
    history: np.ndarray
    bearing_id: str
    window_id: int
    speed_hz: float
    load_kn: float
    condition: int
    fault: str
    start_sample: int
    end_sample: int


def bearing_dir(bearing_id: str, cfg: PipelineConfig) -> Path:
    meta = BEARING_CONFIG[bearing_id]
    return cfg.data_root / CONDITION_DIRS[int(meta["condition"])] / bearing_id


def sorted_csv_files(bearing_id: str, cfg: PipelineConfig) -> List[Path]:
    root = bearing_dir(bearing_id, cfg)
    files = [p for p in root.glob("*.csv") if p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


def _read_csv_two_channels(path: Path) -> np.ndarray:
    try:
        df = pd.read_csv(path, usecols=[0, 1])
    except Exception as exc:
        raise ValueError(f"{path} must contain at least 2 columns for horizontal/vertical channels") from exc
    df = df.apply(pd.to_numeric, errors="coerce")
    arr = df.dropna().values
    if arr.ndim == 1 and arr.size >= 2:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path} does not contain two numeric channels. Actual shape: {arr.shape}")
    return arr


def iter_windows(bearing_id: str, cfg: PipelineConfig) -> Iterator[WindowRecord]:
    """Causal non-overlapping windows; p<1 continues across CSV boundaries."""
    if bearing_id not in BEARING_CONFIG:
        raise KeyError(f"Unknown bearing_id: {bearing_id}")
    meta = BEARING_CONFIG[bearing_id]
    n = cfg.window_size()
    csv_files = sorted_csv_files(bearing_id, cfg)

    buf = np.empty((cfg.csv_rows + n, 2), dtype=float)
    buf_len = 0
    window_id = 0
    consumed = 0

    for csv_path in csv_files:
        data = _read_csv_two_channels(csv_path)
        dlen = len(data)
        needed = buf_len + dlen
        if needed > buf.shape[0]:
            new_buf = np.empty((needed + n, 2), dtype=float)
            new_buf[:buf_len] = buf[:buf_len]
            buf = new_buf
        buf[buf_len:buf_len + dlen] = data
        buf_len += dlen

        while buf_len >= n:
            window = buf[:n].copy()
            buf[:buf_len - n] = buf[n:buf_len]
            buf_len -= n
            window_id += 1
            start = consumed
            consumed += n
            yield WindowRecord(
                history=window,
                bearing_id=bearing_id,
                window_id=window_id,
                speed_hz=float(meta["speed_hz"]),
                load_kn=float(meta["load_kn"]),
                condition=int(meta["condition"]),
                fault=str(meta["fault"]),
                start_sample=start,
                end_sample=consumed,
            )
