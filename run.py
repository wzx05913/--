from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .config import BEARING_CONFIG, FAULT_LABELS, PipelineConfig, resolve_device
    from .fault_diagnosis import OfflineFaultDiagnoser, classification_metrics, top_feature_effects
    from .health_index import HealthIndexBuilder
    from .preprocess import build_all_bearings
    from .rul import RULPredictor
except ImportError:  # pragma: no cover
    from config import BEARING_CONFIG, FAULT_LABELS, PipelineConfig, resolve_device
    from fault_diagnosis import OfflineFaultDiagnoser, classification_metrics, top_feature_effects
    from health_index import HealthIndexBuilder
    from preprocess import build_all_bearings
    from rul import RULPredictor


def _write_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
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
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _dashboard_plot(
    bearing_id: str,
    diag_df: pd.DataFrame,
    hi_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    save_path: Path,
    fpt_index: int | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = diag_df["file_index"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, diag_df["true_future_health_stage"], label="true_future_stage", linewidth=1.2)
    axes[0].plot(x, diag_df["pred_future_health_stage"], label="pred_future_stage", linewidth=1.2)
    if fpt_index is not None:
        axes[0].axvline(fpt_index, color="r", linestyle="--", linewidth=1.0, label="first_pred_degraded")
    axes[0].set_ylabel("future stage")
    axes[0].set_title(f"{bearing_id} diagnosis (A2 forecast)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(hi_df["file_index"], hi_df["HI"], color="tab:green", linewidth=1.3)
    if fpt_index is not None:
        axes[1].axvline(fpt_index, color="r", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("HI")
    axes[1].set_title("Health Index")
    axes[1].grid(alpha=0.25)

    axes[2].plot(rul_df["file_index"], rul_df["rul"], color="tab:purple", linewidth=1.3)
    if fpt_index is not None:
        axes[2].axvline(fpt_index, color="r", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("RUL")
    axes[2].set_xlabel("file_index")
    axes[2].set_title("RUL trajectory")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "bearing_id",
        "fault",
        "file_index",
        "health_stage",
        "future_health_stage",
        "is_fault",
    }
    return [c for c in df.columns if c not in excluded and np.issubdtype(df[c].dtype, np.number)]


def run_lobo(cfg: PipelineConfig, args: argparse.Namespace, logger: logging.Logger) -> None:
    bearings = [b for b in args.bearings if b in BEARING_CONFIG]
    all_df = build_all_bearings(cfg, bearings=bearings, max_windows=args.max_windows)

    summary_rows: List[Dict[str, object]] = []
    for target in bearings:
        test_df = all_df.get(target, pd.DataFrame())
        train_parts = [all_df[b] for b in bearings if b != target and not all_df[b].empty]
        train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()

        if test_df.empty or train_df.empty:
            logger.warning("skip %s: empty train/test split", target)
            continue

        feature_cols = _feature_columns(test_df)
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["future_health_stage"].to_numpy(dtype=int)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["future_health_stage"].to_numpy(dtype=int)

        diagnoser = OfflineFaultDiagnoser(cfg)
        diagnoser.fit(x_train, y_train)
        probs = diagnoser.predict_proba(x_test, n_classes=3)
        y_pred = np.argmax(probs, axis=1).astype(int)
        metrics = classification_metrics(y_test, y_pred, n_classes=3)

        hi_builder = HealthIndexBuilder(cfg)
        rul_predictor = RULPredictor(cfg)

        diag_rows: List[Dict[str, object]] = []
        hi_rows: List[Dict[str, object]] = []
        rul_rows: List[Dict[str, object]] = []

        pred_degraded = np.where(y_pred > 0)[0]
        fpt_index = int(test_df.loc[int(pred_degraded[0]), "file_index"]) if len(pred_degraded) else None

        for i, (_, row) in enumerate(test_df.iterrows()):
            feat_row = {k: float(row[k]) for k in cfg.degradation_directions if k in row}
            hi = hi_builder.update(feat_row)
            rul = rul_predictor.update(float(hi["HI"]))
            shap_top = top_feature_effects(x_test[i], feature_cols, top_k=cfg.shap_top_k)

            base_meta = {
                "bearing_id": target,
                "file_index": int(row["file_index"]),
                "fault": row.get("fault", ""),
            }
            diag_rows.append({
                **base_meta,
                "true_future_health_stage": int(y_test[i]),
                "pred_future_health_stage": int(y_pred[i]),
                "true_future_stage_name": FAULT_LABELS[int(y_test[i])],
                "pred_future_stage_name": FAULT_LABELS[int(y_pred[i])],
                "model_kind": diagnoser.model_kind,
                "shap_top": json.dumps(shap_top, ensure_ascii=False),
                **{f"prob_{j}": float(probs[i, j]) for j in range(probs.shape[1])},
            })
            hi_rows.append({
                **base_meta,
                "HI": float(hi["HI"]),
                "D": float(hi["D"]),
                "level": int(hi["level"]),
                "weights": json.dumps(hi["weights"], ensure_ascii=False),
                "damages": json.dumps(hi["damages"], ensure_ascii=False),
            })
            rul_rows.append({
                **base_meta,
                "rul": int(rul["rul"]),
                "ci_low": int(rul["ci_low"]),
                "ci_high": int(rul["ci_high"]),
                "method": rul["method"],
                "future_hi": json.dumps(rul["future_hi"], ensure_ascii=False),
            })

        _write_rows_csv(diag_rows, cfg.output_dir / "diagnosis" / f"{target}_diagnosis.csv")
        _write_rows_csv(hi_rows, cfg.output_dir / "hi" / f"{target}_hi.csv")
        _write_rows_csv(rul_rows, cfg.output_dir / "rul" / f"{target}_rul.csv")

        diag_df = pd.DataFrame(diag_rows)
        hi_df = pd.DataFrame(hi_rows)
        rul_df = pd.DataFrame(rul_rows)
        _dashboard_plot(
            bearing_id=target,
            diag_df=diag_df,
            hi_df=hi_df,
            rul_df=rul_df,
            save_path=cfg.output_dir / "figures" / f"{target}_dashboard.png",
            fpt_index=fpt_index,
        )

        summary_rows.append({
            "bearing_id": target,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "support": int(metrics["support"]),
            "fpt_index": fpt_index,
            "model_kind": diagnoser.model_kind,
            "confusion_matrix": json.dumps(metrics["confusion_matrix"], ensure_ascii=False),
        })

        logger.info(
            "%s LOBO done | acc=%.4f macro_f1=%.4f weighted_f1=%.4f n_test=%d model=%s fpt=%s",
            target,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
            len(test_df),
            diagnoser.model_kind,
            fpt_index,
        )

    _write_rows_csv(summary_rows, cfg.output_dir / "GLOBAL_SUMMARY.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XJTU-SY LOBO offline diagnosis→HI→RUL pipeline")
    p.add_argument("--bearings", nargs="+", default=sorted(BEARING_CONFIG), choices=sorted(BEARING_CONFIG))
    p.add_argument("--p", type=float, default=1.0, help="窗口分钟比例，(0,1]")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--lambda-hi", type=float, default=3.0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--data-root", default=r"D:\Desktop\数据集实验\西交数据集\data\XJTU-SY_Bearing_Datasets")
    p.add_argument("--output-dir", default="output_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        p=args.p,
        wavelet=args.wavelet,
        hi_lambda=args.lambda_hi,
        device=resolve_device(args.device),
    )
    cfg.ensure_dirs()
    logger = _setup_logger(cfg.output_dir / "logs" / "run.log")
    logger.info("启动 XJTU-SY LOBO 离线流水线: bearings=%s p=%s wavelet=%s device=%s lambda_hi=%s", args.bearings, cfg.p, cfg.wavelet, cfg.device, cfg.hi_lambda)
    run_lobo(cfg, args, logger)
    logger.info("全部完成，结果已保存到 %s", cfg.output_dir)


if __name__ == "__main__":
    main()
