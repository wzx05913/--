from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import BEARING_CONFIG, FAULT_LABELS, PipelineConfig, compute_fault_freqs, resolve_device
from .data_loader import iter_windows
from .feature_engineering import extract_features
from .fault_diagnosis import OnlineFaultDiagnoser
from .health_index import HealthIndexBuilder
from .rul import RULPredictor


def _write_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    extras = sorted({k for r in rows for k in r.keys()} - set(keys))
    keys.extend(extras)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("xjtu_sy_phm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def _feature_matrix_row(row: Dict[str, float]) -> tuple[np.ndarray, List[str]]:
    exclude = {"run_index", "speed_hz", "load_kn", "condition"}
    names = sorted([k for k, v in row.items() if k not in exclude and np.isscalar(v)])
    return np.array([float(row[k]) for k in names], dtype=float), names


def run_bearing(bearing_id: str, cfg: PipelineConfig, args: argparse.Namespace, logger: logging.Logger, stamp: str) -> None:
    diagnoser = OnlineFaultDiagnoser(cfg)
    hi_builder = HealthIndexBuilder(cfg)
    rul_predictor = RULPredictor(cfg)
    prev_base = None
    feature_rows = []
    diag_rows = []
    hi_rows = []
    rul_rows = []

    for record in iter_windows(bearing_id, cfg):
        if args.max_windows and record.window_id > args.max_windows:
            break
        feats, prev_base = extract_features(record, cfg, prev_base)
        vec, names = _feature_matrix_row(feats)
        peak = max(float(feats.get("horizontal_peak", 0.0)), float(feats.get("vertical_peak", 0.0)))
        diag = diagnoser.update(vec, names, record.window_id, record.fault, peak)
        hi = hi_builder.update(feats, diag.shap_top)
        rul = rul_predictor.update(float(hi["HI"]))

        base_meta = {"bearing_id": bearing_id, "window_id": record.window_id, "fault": record.fault}
        feature_rows.append({**base_meta, **feats})
        diag_rows.append({**base_meta, "label": diag.label, "label_name": FAULT_LABELS[diag.label], "weak_label": diag.weak_label, "weak_reason": diag.weak_reason,
                          **{f"prob_{i}": float(diag.probs[i]) for i in range(6)}, "shap_top": json.dumps(diag.shap_top, ensure_ascii=False), **diag.metrics,
                          "calibration": json.dumps(diag.calibration, ensure_ascii=False)})
        hi_rows.append({**base_meta, "HI": hi["HI"], "D": hi["D"], "level": hi["level"],
                        "weights": json.dumps(hi["weights"], ensure_ascii=False), "damages": json.dumps(hi["damages"], ensure_ascii=False)})
        rul_rows.append({**base_meta, "rul": rul["rul"], "ci_low": rul["ci_low"], "ci_high": rul["ci_high"], "method": rul["method"],
                         "future_hi": json.dumps(rul["future_hi"], ensure_ascii=False)})
        logger.info("%s #%d p=%s HI=%.4f D=%.4f level=%s RUL=%s CI=[%s,%s] method=%s probs=%s shap=%s metrics=%s",
                    bearing_id, record.window_id, cfg.p, hi["HI"], hi["D"], hi["level"], rul["rul"], rul["ci_low"], rul["ci_high"], rul["method"],
                    np.round(diag.probs, 4).tolist(), diag.shap_top, diag.metrics)

    if not feature_rows:
        logger.warning("%s 没有产生窗口，请检查数据路径。", bearing_id)
        return
    _write_rows_csv(feature_rows, cfg.output_dir / "features" / f"{stamp}_{bearing_id}_features.csv")
    _write_rows_csv(diag_rows, cfg.output_dir / "diagnosis" / f"{stamp}_{bearing_id}_diagnosis.csv")
    _write_rows_csv(hi_rows, cfg.output_dir / "hi" / f"{stamp}_{bearing_id}_hi.csv")
    _write_rows_csv(rul_rows, cfg.output_dir / "rul" / f"{stamp}_{bearing_id}_rul.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XJTU-SY 轴承端到端在线 PHM 流水线")
    p.add_argument("--bearings", nargs="+", default=["Bearing1_1"], choices=sorted(BEARING_CONFIG))
    p.add_argument("--p", type=float, default=1.0, help="窗口分钟比例，(0,1]")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--lambda-hi", type=float, default=3.0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--data-root", default=r"D:\Desktop\数据集实验\西交数据集\data\XJTU-SY_Bearing_Datasets")
    p.add_argument("--output-dir", default="output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(data_root=Path(args.data_root), output_dir=Path(args.output_dir), p=args.p, wavelet=args.wavelet, hi_lambda=args.lambda_hi, device=resolve_device(args.device))
    cfg.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(cfg.output_dir / "logs" / f"{stamp}_run.log")
    logger.info("启动 XJTU-SY PHM 在线流水线: bearings=%s p=%s wavelet=%s device=%s lambda_hi=%s", args.bearings, cfg.p, cfg.wavelet, cfg.device, cfg.hi_lambda)
    logger.info("故障特征频率示例: %s", {b: compute_fault_freqs(float(BEARING_CONFIG[b]["speed_hz"])) for b in args.bearings})
    for b in args.bearings:
        run_bearing(b, cfg, args, logger, stamp)
    logger.info("全部完成，结果已保存到 %s", cfg.output_dir)


if __name__ == "__main__":
    main()
