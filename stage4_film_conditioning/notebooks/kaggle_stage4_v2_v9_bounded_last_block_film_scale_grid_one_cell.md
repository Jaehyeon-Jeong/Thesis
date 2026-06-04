# Kaggle Stage 4 v2 Priority 10 Bounded Last-Block FiLM Scale Grid One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Fear & Greed Kaggle dataset:
  `ashishpatel8736/historical-and-fear-greed-index-datasets`

This runner executes Stage 4 v2 priority 10:

```text
I60 / R20 / ohlc_ma_vb / context_window=60
F&G-only context x film_full_bounded_last_block x 3 scales x 5 seeds = 15 runs

features: fg_value, fg_mean_60, fg_delta_60, fg_std_60
method: film_full_bounded_last_block
scales: 0.02, 0.05, 0.10
seeds: 42, 43, 44, 45, 46
```

Purpose:
- keep the strong Stage 2 visual baseline fixed;
- keep the FiLM architecture fixed;
- test whether bounded FiLM modulation strength is the reason for P8 seed
  collapse;
- record validation/test predicted-positive-rate collapse without changing the
  checkpoint rule yet.

Default settings skip Grad-CAM during the scale grid to save disk and time. Set
`RUN_GRADCAM=True` only after selecting a scale worth interpreting.

```python
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import yaml
from IPython.display import display, Markdown

# ============================================================
# User settings
# ============================================================
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning")
STAGE2_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage4_saved_outputs")

# Leave empty to auto-detect under /kaggle/input.
SOURCE_FILE = ""
FEAR_GREED_FILE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_FEATURE_SET_NAME = "fg_only"
PRIMARY_CONTEXT_FEATURES = ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"]
CONTEXT_METHOD = "film_full_bounded_last_block"
BOUNDED_FILM_SCALES = [0.02, 0.05, 0.10]
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLITS = ["validation", "test"]
TRADING_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
SAVE_BACKUP_ZIPS = False
RESUME_EXISTING_PROJECT = False
DELETE_EXISTING_BACKUP_ZIPS_ON_START = True
RUN_GRADCAM = False

# Smoke check only. For the real 4-V9 run, keep False.
SMOKE_TEST = False

# Strict Stage 2-style comparison settings.
BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

STAGE2_OHLC_MA_VB_BASELINE_ACCURACY_MEAN = 0.579320
STAGE2_OHLC_MA_VB_BASELINE_ROC_AUC_MEAN = 0.584862
P7_FG_ONLY_FILM_FULL_ACCURACY_MEAN = 0.5524
P7_FG_ONLY_FILM_FULL_ROC_AUC_MEAN = 0.5465
P8_FG_ONLY_BOUNDED_SCALE_010_ACCURACY_MEAN = 0.5425
P8_FG_ONLY_BOUNDED_SCALE_010_ROC_AUC_MEAN = 0.5763

COLLAPSE_DOWN_THRESHOLD = 0.15
COLLAPSE_UP_THRESHOLD = 0.85


def copy_or_extract_input(src: Path, dst: Path, expected_child: str | None = None):
    """Copy a Kaggle input folder or extract a Kaggle input zip file."""

    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        candidate = src / expected_child if expected_child and (src / expected_child).exists() else src
        shutil.copytree(candidate, dst)
        return
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = dst.parent / f"{dst.name}_unzipped"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as archive:
            archive.extractall(tmp)
        candidate = tmp / expected_child if expected_child and (tmp / expected_child).exists() else tmp
        if expected_child and not (candidate / "scripts").exists():
            nested = next(tmp.rglob(expected_child), None)
            if nested is not None:
                candidate = nested
        shutil.copytree(candidate, dst)
        shutil.rmtree(tmp)
        return
    raise FileNotFoundError(f"Input must be a folder or .zip file: {src}")


def run(cmd, cwd=PROJECT_ROOT, capture=False, check=True):
    """Run one command. Use capture=True only for quiet completion probes."""

    cmd = [str(item) for item in cmd]
    if not capture:
        print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def scale_label(scale: float) -> str:
    """Return a stable filename-safe label for a modulation scale."""

    return f"{float(scale):.2f}".replace(".", "p")


def experiment_suffix(scale: float) -> str:
    """Scale-specific suffix to prevent overwriting P8 or other V9 scales."""

    return f"{CONTEXT_FEATURE_SET_NAME}_scale_{scale_label(scale)}"


def experiment_name(context_method: str, scale: float) -> str:
    return (
        f"stage4_{context_method}_i{IMAGE_WINDOW}_"
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{experiment_suffix(scale)}"
    )


def context_name() -> str:
    return (
        f"stage4_context_i{IMAGE_WINDOW}_"
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{CONTEXT_FEATURE_SET_NAME}"
    )


def assert_required_code():
    """Fail early if an uploaded code snapshot is stale or incomplete."""

    required_stage4 = [
        "scripts/audit_stage4_context_sources.py",
        "scripts/build_stage4_context_features.py",
        "scripts/run_stage4_context_model.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/evaluate_stage4_trading.py",
        "scripts/generate_stage4_gradcam_context.py",
        "src/stage4_film/models/film_stock_cnn.py",
        "src/stage4_film/layers/film.py",
    ]
    missing_stage4 = [path for path in required_stage4 if not (PROJECT_ROOT / path).exists()]
    if missing_stage4:
        raise RuntimeError("Stage 4 code snapshot is incomplete: " + ", ".join(missing_stage4))

    required_stage2 = [
        "src/stage2_btc/data",
        "src/stage2_btc/evaluation",
        "src/stage2_btc/models/stock_cnn.py",
    ]
    missing_stage2 = [path for path in required_stage2 if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing_stage2:
        raise RuntimeError("Stage 2 dependency snapshot is incomplete: " + ", ".join(missing_stage2))

    marker_checks = [
        ("src/stage4_film/config.py", "experiment_suffix"),
        ("src/stage4_film/context/features.py", "context_suffix"),
        ("src/stage4_film/models/film_stock_cnn.py", "BoundedLastBlockFilmContextStockCNN"),
        ("src/stage4_film/models/film_stock_cnn.py", "modulation_scale"),
        ("src/stage4_film/runners/context_experiment.py", "film_full_bounded_last_block"),
    ]
    stale = []
    for rel_path, marker in marker_checks:
        path = PROJECT_ROOT / rel_path
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            stale.append(f"{rel_path} missing marker {marker!r}")
    if stale:
        raise RuntimeError(
            "Stage 4 code snapshot is stale. Upload the latest "
            "stage4_film_conditioning folder/zip, then rerun. Problems: "
            + "; ".join(stale)
        )


def patch_kaggle_config(scale: float):
    """Patch Stage 4 Kaggle config for one bounded-FiLM scale."""

    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cfg["paths"]["project_root"] = str(PROJECT_ROOT)
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
    cfg["paths"]["fear_greed_file"] = FEAR_GREED_FILE
    cfg["paths"]["output_root"] = str(PROJECT_ROOT / "outputs/stage4")
    cfg["paths"]["checkpoint_root"] = str(PROJECT_ROOT / "outputs/stage4/checkpoints")
    cfg["paths"]["metrics_root"] = str(PROJECT_ROOT / "outputs/stage4/metrics")
    cfg["paths"]["predictions_root"] = str(PROJECT_ROOT / "outputs/stage4/predictions")
    cfg["paths"]["figures_root"] = str(PROJECT_ROOT / "outputs/stage4/figures")
    cfg["paths"]["context_root"] = str(PROJECT_ROOT / "outputs/stage4/context")
    cfg["paths"]["run_manifest_root"] = str(PROJECT_ROOT / "outputs/stage4/run_manifests")
    cfg["paths"]["reports_root"] = str(PROJECT_ROOT / "reports")
    cfg["paths"]["tables_root"] = str(PROJECT_ROOT / "reports/tables")

    cfg["stage2_dependency"]["project_root"] = str(STAGE2_PROJECT_ROOT)
    cfg["stage2_dependency"]["src_path"] = str(STAGE2_PROJECT_ROOT / "src")
    cfg["stage2_dependency"]["baseline_output_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2")

    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["feature_set_name"] = CONTEXT_FEATURE_SET_NAME
    cfg["context"]["primary_features"] = list(PRIMARY_CONTEXT_FEATURES)
    cfg["context"]["fear_greed"]["kaggle_file"] = FEAR_GREED_FILE

    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(PRIMARY_CONTEXT_FEATURES)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = experiment_suffix(scale)
    cfg["stage4_model"].setdefault("film_full_bounded_last_block", {})
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = float(scale)

    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    cfg["evaluation"]["batch_size"] = BATCH_SIZE

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        f"Config patched: {config_path} scale={scale:.2f} "
        f"suffix={experiment_suffix(scale)}",
        flush=True,
    )


def expected_output_paths(context_method: str, run_seed: int, scale: float) -> dict[str, Path]:
    exp = experiment_name(context_method, scale)
    seed_dir = f"seed_{run_seed}"
    outputs = PROJECT_ROOT / "outputs/stage4"
    gradcam_dir = outputs / "figures" / exp / seed_dir / "gradcam" / TRADING_SPLIT
    context_dir = outputs / "context" / context_name() / seed_dir
    metrics_dir = outputs / "metrics" / exp / seed_dir
    predictions_dir = outputs / "predictions" / exp / seed_dir
    return {
        "best_checkpoint": outputs / "checkpoints" / exp / seed_dir / "best.pt",
        "validation_predictions": predictions_dir / "validation_predictions.csv",
        "validation_metrics": metrics_dir / "validation_metrics.json",
        "test_predictions": predictions_dir / "test_predictions.csv",
        "test_metrics": metrics_dir / "test_metrics.json",
        "test_trading_metrics": metrics_dir / "test_trading_metrics.json",
        "gradcam": gradcam_dir / f"btc_context_gradcam_test_{GRADCAM_SAMPLES_PER_CLASS}perclass.png",
        "context_features": context_dir / "context_features.csv",
        "run_manifest": outputs / "run_manifests" / exp / seed_dir / "run_manifest.json",
    }


def backup_outputs(label: str, phase: str, context_method: str | None = None, run_seed: int | None = None, scale: float | None = None):
    """Zip current Stage 4 outputs outside PROJECT_ROOT so later reruns cannot erase them."""

    if not SAVE_BACKUP_ZIPS:
        return None
    outputs_root = PROJECT_ROOT / "outputs/stage4"
    if not outputs_root.exists():
        print(f"[backup:{label}:{phase}] skip: no outputs yet", flush=True)
        return None

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_label = f"seed{run_seed}" if run_seed is not None else "allseeds"
    scale_part = f"_scale{scale_label(scale)}" if scale is not None else ""
    archive_base = BACKUP_ROOT / f"{label}_{seed_label}{scale_part}_{phase}_{timestamp}_outputs"
    try:
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", outputs_root))
    except OSError as exc:
        print(f"[backup:{label}:{phase}] skipped after OSError: {exc}", flush=True)
        return None
    receipt = {
        "label": label,
        "context_method": context_method,
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "context_window": CONTEXT_WINDOW,
        "context_feature_set_name": CONTEXT_FEATURE_SET_NAME,
        "bounded_film_scale": scale,
        "run_seed": run_seed,
        "phase": phase,
        "created_utc": timestamp,
        "archive_path": str(archive_path),
        "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 3),
    }
    receipt_path = BACKUP_ROOT / f"{label}_{seed_label}{scale_part}_{phase}_{timestamp}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        f"[backup:{label}:{phase}] saved {archive_path} "
        f"({receipt['archive_size_mb']} MB), receipt={receipt_path}",
        flush=True,
    )
    return archive_path


def delete_existing_backup_zips():
    """Free Kaggle disk space by deleting old Stage 4 backup zips/receipts."""

    if not DELETE_EXISTING_BACKUP_ZIPS_ON_START or not BACKUP_ROOT.exists():
        return
    removed = 0
    removed_bytes = 0
    for pattern in ["*.zip", "*receipt.json"]:
        for path in BACKUP_ROOT.glob(pattern):
            if path.is_file():
                size = path.stat().st_size
                path.unlink()
                removed += 1
                removed_bytes += size
    print(
        f"[cleanup] removed {removed} backup files "
        f"({removed_bytes / (1024 * 1024):.1f} MB) from {BACKUP_ROOT}",
        flush=True,
    )


def run_step(label: str, phase: str, cmd, context_method: str | None = None, run_seed: int | None = None, scale: float | None = None):
    """Run a stage and optionally back up current outputs even if the stage fails."""

    try:
        return run(cmd)
    finally:
        backup_outputs(label, phase, context_method=context_method, run_seed=run_seed, scale=scale)


def metric_output_check(context_method: str, run_seed: int, scale: float) -> tuple[bool, list[str]]:
    """Check V9 metric-grid outputs without requiring Grad-CAM by default."""

    paths = expected_output_paths(context_method, run_seed, scale)
    required = [
        "best_checkpoint",
        "validation_predictions",
        "validation_metrics",
        "test_predictions",
        "test_metrics",
        "test_trading_metrics",
        "context_features",
        "run_manifest",
    ]
    if RUN_GRADCAM:
        required.append("gradcam")

    missing = [name for name in required if not paths[name].exists()]
    if missing:
        return False, missing

    try:
        test_rows = len(pd.read_csv(paths["test_predictions"], usecols=["pred_class"]))
    except Exception as exc:
        return False, [f"test_predictions_parse_failed:{exc}"]
    if test_rows < MIN_PREDICTIONS:
        return False, [f"test_predictions_rows<{MIN_PREDICTIONS}:{test_rows}"]

    try:
        validation_rows = len(pd.read_csv(paths["validation_predictions"], usecols=["pred_class"]))
    except Exception as exc:
        return False, [f"validation_predictions_parse_failed:{exc}"]
    if validation_rows <= 0:
        return False, ["validation_predictions_empty"]
    return True, []


def is_completed(context_method: str, run_seed: int, scale: float) -> bool:
    ok, missing = metric_output_check(context_method, run_seed, scale)
    if not ok:
        print(
            f"[not-complete] {experiment_name(context_method, scale)}, "
            f"seed={run_seed}: {missing}",
            flush=True,
        )
    return ok


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collapse_label(predicted_positive_rate) -> str:
    if predicted_positive_rate is None:
        return "unknown"
    value = float(predicted_positive_rate)
    if value <= COLLAPSE_DOWN_THRESHOLD:
        return "mostly_down"
    if value >= COLLAPSE_UP_THRESHOLD:
        return "mostly_up"
    return "not_collapsed"


def add_metric_fields(row: dict, metrics: dict, prefix: str) -> None:
    for key in [
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "accuracy_minus_majority_class_accuracy",
        "accuracy_minus_majority",
        "roc_auc",
        "average_precision",
        "f1",
        "brier_score",
        "positive_rate",
        "predicted_positive_rate",
    ]:
        if key in metrics:
            row[f"{prefix}_{key}"] = metrics.get(key)
    if row.get(f"{prefix}_accuracy_minus_majority") is None:
        value = row.get(f"{prefix}_accuracy_minus_majority_class_accuracy")
        if value is not None:
            row[f"{prefix}_accuracy_minus_majority"] = value
    row[f"{prefix}_collapse_label"] = collapse_label(row.get(f"{prefix}_predicted_positive_rate"))


def read_result_row(context_method: str, run_seed: int, scale: float, status: str, error: str = "") -> dict:
    exp = experiment_name(context_method, scale)
    paths = expected_output_paths(context_method, run_seed, scale)
    validation_metrics = load_json_if_exists(paths["validation_metrics"])
    test_metrics = load_json_if_exists(paths["test_metrics"])
    trading_metrics = load_json_if_exists(paths["test_trading_metrics"])

    row = {
        "priority": "4-V9",
        "experiment_name": exp,
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "context_window": CONTEXT_WINDOW,
        "context_feature_set_name": CONTEXT_FEATURE_SET_NAME,
        "context_scope": "fear_greed_only",
        "context_features": ",".join(PRIMARY_CONTEXT_FEATURES),
        "context_method": context_method,
        "bounded_film_scale": float(scale),
        "experiment_suffix": experiment_suffix(scale),
        "run_seed": run_seed,
        "status": status,
        "error": error,
        "validation_available": bool(validation_metrics),
        "test_available": bool(test_metrics),
        "trading_available": bool(trading_metrics),
        "stage2_ohlc_ma_vb_baseline_accuracy_mean": STAGE2_OHLC_MA_VB_BASELINE_ACCURACY_MEAN,
        "stage2_ohlc_ma_vb_baseline_roc_auc_mean": STAGE2_OHLC_MA_VB_BASELINE_ROC_AUC_MEAN,
        "p7_fg_only_film_full_accuracy_mean": P7_FG_ONLY_FILM_FULL_ACCURACY_MEAN,
        "p7_fg_only_film_full_roc_auc_mean": P7_FG_ONLY_FILM_FULL_ROC_AUC_MEAN,
        "p8_fg_only_bounded_scale_010_accuracy_mean": P8_FG_ONLY_BOUNDED_SCALE_010_ACCURACY_MEAN,
        "p8_fg_only_bounded_scale_010_roc_auc_mean": P8_FG_ONLY_BOUNDED_SCALE_010_ROC_AUC_MEAN,
    }
    add_metric_fields(row, validation_metrics, "validation")
    add_metric_fields(row, test_metrics, "test")

    if trading_metrics:
        for strategy_name in ["long_flat", "long_short"]:
            values = trading_metrics.get(strategy_name, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                row[f"test_{strategy_name}_{key}"] = values.get(key)

    if row.get("test_accuracy") is not None:
        row["test_accuracy_minus_stage2_ohlc_ma_vb_mean"] = (
            float(row["test_accuracy"]) - STAGE2_OHLC_MA_VB_BASELINE_ACCURACY_MEAN
        )
        row["test_accuracy_minus_p7_film_full_mean"] = (
            float(row["test_accuracy"]) - P7_FG_ONLY_FILM_FULL_ACCURACY_MEAN
        )
    if row.get("test_roc_auc") is not None:
        row["test_roc_auc_minus_stage2_ohlc_ma_vb_mean"] = (
            float(row["test_roc_auc"]) - STAGE2_OHLC_MA_VB_BASELINE_ROC_AUC_MEAN
        )
        row["test_roc_auc_minus_p7_film_full_mean"] = (
            float(row["test_roc_auc"]) - P7_FG_ONLY_FILM_FULL_ROC_AUC_MEAN
        )
    return row


def summarize_seed_results(seed_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(seed_rows)
    if df.empty:
        return pd.DataFrame()

    numeric = [
        "validation_accuracy",
        "validation_roc_auc",
        "validation_average_precision",
        "validation_f1",
        "validation_brier_score",
        "validation_predicted_positive_rate",
        "test_num_samples",
        "test_accuracy",
        "test_majority_class_accuracy",
        "test_accuracy_minus_majority",
        "test_roc_auc",
        "test_average_precision",
        "test_f1",
        "test_brier_score",
        "test_positive_rate",
        "test_predicted_positive_rate",
        "test_long_flat_sharpe_net",
        "test_long_short_sharpe_net",
        "test_long_flat_annualized_return_net",
        "test_long_short_annualized_return_net",
    ]
    available = [column for column in numeric if column in df.columns]
    rows = []
    grouped = df.groupby(["context_method", "bounded_film_scale"], dropna=False)
    for (method, scale), frame in grouped:
        row = {
            "priority": "4-V9",
            "context_method": method,
            "bounded_film_scale": scale,
            "experiment_suffix": experiment_suffix(float(scale)),
            "seed_count": int(frame["run_seed"].nunique()),
            "ok_seed_count": int((frame["status"].astype(str).isin(["ok", "skipped_completed"])).sum()),
            "validation_collapse_count": int((frame.get("validation_collapse_label", pd.Series(dtype=str)) != "not_collapsed").sum()),
            "test_collapse_count": int((frame.get("test_collapse_label", pd.Series(dtype=str)) != "not_collapsed").sum()),
        }
        for column in available:
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)
            row[f"{column}_count"] = int(values.notna().sum())
        if row.get("test_accuracy_mean") is not None:
            row["test_accuracy_mean_minus_stage2_ohlc_ma_vb_mean"] = (
                row["test_accuracy_mean"] - STAGE2_OHLC_MA_VB_BASELINE_ACCURACY_MEAN
            )
            row["test_accuracy_mean_minus_p7_film_full_mean"] = (
                row["test_accuracy_mean"] - P7_FG_ONLY_FILM_FULL_ACCURACY_MEAN
            )
        if row.get("test_roc_auc_mean") is not None:
            row["test_roc_auc_mean_minus_stage2_ohlc_ma_vb_mean"] = (
                row["test_roc_auc_mean"] - STAGE2_OHLC_MA_VB_BASELINE_ROC_AUC_MEAN
            )
            row["test_roc_auc_mean_minus_p7_film_full_mean"] = (
                row["test_roc_auc_mean"] - P7_FG_ONLY_FILM_FULL_ROC_AUC_MEAN
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_collapse(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    rows = []
    for scale, frame in seed_df.groupby("bounded_film_scale", dropna=False):
        row = {
            "bounded_film_scale": scale,
            "seed_count": int(frame["run_seed"].nunique()),
        }
        for split in ["validation", "test"]:
            labels = frame.get(f"{split}_collapse_label", pd.Series(dtype=str)).astype(str)
            row[f"{split}_mostly_down_count"] = int((labels == "mostly_down").sum())
            row[f"{split}_mostly_up_count"] = int((labels == "mostly_up").sum())
            row[f"{split}_not_collapsed_count"] = int((labels == "not_collapsed").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def create_result_bundle(paths: list[Path]) -> Path:
    bundle_path = Path("/kaggle/working/stage4_v2_v9_bounded_last_block_film_scale_grid_result_bundle.zip")
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.exists():
                archive.write(path, arcname=f"reports/tables/{path.name}")
    print(f"Result bundle: {bundle_path}", flush=True)
    return bundle_path


# ============================================================
# 1. Copy code snapshots and patch config
# ============================================================
if RESUME_EXISTING_PROJECT:
    if not PROJECT_ROOT.exists() or not STAGE2_PROJECT_ROOT.exists():
        raise FileNotFoundError(
            "RESUME_EXISTING_PROJECT=True requires existing PROJECT_ROOT and "
            "STAGE2_PROJECT_ROOT in /kaggle/working."
        )
    print(f"Resuming existing Stage 4 project: {PROJECT_ROOT}", flush=True)
    print(f"Resuming existing Stage 2 dependency: {STAGE2_PROJECT_ROOT}", flush=True)
else:
    copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
    copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_required_code()
patch_kaggle_config(BOUNDED_FILM_SCALES[0])
delete_existing_backup_zips()
print(f"Stage 4 project ready at: {PROJECT_ROOT}", flush=True)
print(f"Stage 2 dependency ready at: {STAGE2_PROJECT_ROOT}", flush=True)


# ============================================================
# 2. Source audit
# ============================================================
run([
    sys.executable, "-u",
    "scripts/audit_stage4_context_sources.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--return-horizon", str(RETURN_HORIZON),
    "--output", "reports/tables/stage4_context_source_audit.json",
])


# ============================================================
# 3. Train/evaluate/check bounded last-block FiLM scale grid
# ============================================================
smoke_train_args = []
smoke_data_args = []
if SMOKE_TEST:
    smoke_train_args = [
        "--max-epochs", "2",
        "--max-train-rows", "128",
        "--max-validation-rows", "64",
        "--max-test-rows", "64",
    ]
    smoke_data_args = [
        "--max-train-rows", "128",
        "--max-validation-rows", "64",
        "--max-test-rows", "64",
    ]

summary_rows = []
total_runs = len(BOUNDED_FILM_SCALES) * len(RUN_SEEDS)
run_index = 0

for run_seed in RUN_SEEDS:
    print("\n" + "#" * 80, flush=True)
    print(f"Seed {run_seed}: build shared F&G-only context features", flush=True)
    print("#" * 80, flush=True)
    patch_kaggle_config(BOUNDED_FILM_SCALES[0])
    run_step(f"stage4_context_seed{run_seed}", "after_context_build", [
        sys.executable, "-u",
        "scripts/build_stage4_context_features.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(run_seed),
        "--write-report-copy",
    ], run_seed=run_seed)

    for scale in BOUNDED_FILM_SCALES:
        patch_kaggle_config(scale)
        run_index += 1
        exp = experiment_name(CONTEXT_METHOD, scale)
        print("\n" + "=" * 80, flush=True)
        print(
            f"[{run_index}/{total_runs}] {exp}, seed={run_seed}, scale={scale:.2f}",
            flush=True,
        )
        print("=" * 80, flush=True)

        if SKIP_COMPLETED and is_completed(CONTEXT_METHOD, run_seed, scale):
            print(f"[skip] Metric output check already passes for {exp}, seed={run_seed}", flush=True)
            summary_rows.append(read_result_row(CONTEXT_METHOD, run_seed, scale, status="skipped_completed"))
            continue

        try:
            run_step(exp, "after_train", [
                sys.executable, "-u",
                "scripts/run_stage4_context_model.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", IMAGE_SPEC,
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", CONTEXT_METHOD,
                "--run-seed", str(run_seed),
            ] + smoke_train_args, context_method=CONTEXT_METHOD, run_seed=run_seed, scale=scale)

            for split in EVAL_SPLITS:
                run_step(exp, f"after_{split}_prediction_eval", [
                    sys.executable, "-u",
                    "scripts/evaluate_stage4_predictions.py",
                    "--config", "configs/env_kaggle.yaml",
                    "--image-window", str(IMAGE_WINDOW),
                    "--image-spec", IMAGE_SPEC,
                    "--return-horizon", str(RETURN_HORIZON),
                    "--context-method", CONTEXT_METHOD,
                    "--run-seed", str(run_seed),
                    "--split", split,
                ] + smoke_data_args, context_method=CONTEXT_METHOD, run_seed=run_seed, scale=scale)

            run_step(exp, "after_trading_eval", [
                sys.executable, "-u",
                "scripts/evaluate_stage4_trading.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", IMAGE_SPEC,
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", CONTEXT_METHOD,
                "--run-seed", str(run_seed),
                "--split", TRADING_SPLIT,
            ], context_method=CONTEXT_METHOD, run_seed=run_seed, scale=scale)

            if RUN_GRADCAM:
                run_step(exp, "after_gradcam", [
                    sys.executable, "-u",
                    "scripts/generate_stage4_gradcam_context.py",
                    "--config", "configs/env_kaggle.yaml",
                    "--image-window", str(IMAGE_WINDOW),
                    "--image-spec", IMAGE_SPEC,
                    "--return-horizon", str(RETURN_HORIZON),
                    "--context-method", CONTEXT_METHOD,
                    "--run-seed", str(run_seed),
                    "--split", TRADING_SPLIT,
                    "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
                    "--write-report-copy",
                ] + smoke_data_args, context_method=CONTEXT_METHOD, run_seed=run_seed, scale=scale)

            ok, missing = metric_output_check(CONTEXT_METHOD, run_seed, scale)
            if not ok:
                raise RuntimeError("Metric output check failed: " + ", ".join(missing))
            summary_rows.append(read_result_row(CONTEXT_METHOD, run_seed, scale, status="ok"))
        except Exception as exc:
            print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
            summary_rows.append(read_result_row(CONTEXT_METHOD, run_seed, scale, status="failed", error=str(exc)))
            if not CONTINUE_ON_ERROR:
                raise


# ============================================================
# 4. Write seed-level and mean/std summary tables
# ============================================================
tables_root = PROJECT_ROOT / "reports/tables"
tables_root.mkdir(parents=True, exist_ok=True)
seed_df = pd.DataFrame(summary_rows)
mean_std_df = summarize_seed_results(summary_rows)
collapse_df = summarize_collapse(seed_df)

seed_csv = tables_root / "stage4_v2_v9_bounded_last_block_film_scale_grid_seed_results.csv"
mean_std_csv = tables_root / "stage4_v2_v9_bounded_last_block_film_scale_grid_mean_std_results.csv"
collapse_csv = tables_root / "stage4_v2_v9_bounded_last_block_film_scale_grid_collapse_summary.csv"
run_summary_json = tables_root / "stage4_v2_v9_bounded_last_block_film_scale_grid_run_summary.json"
seed_df.to_csv(seed_csv, index=False)
mean_std_df.to_csv(mean_std_csv, index=False)
collapse_df.to_csv(collapse_csv, index=False)
run_summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

bundle_path = create_result_bundle([seed_csv, mean_std_csv, collapse_csv, run_summary_json])
backup_outputs("stage4_v2_v9_bounded_last_block_film_scale_grid", "after_summary")

display(Markdown("# Stage 4 v2 Priority 10 Bounded Last-Block FiLM Scale Grid: Seed-Level Results"))
display(seed_df.sort_values(["bounded_film_scale", "run_seed"]))

display(Markdown("# Stage 4 v2 Priority 10 Mean/Std Summary"))
if not mean_std_df.empty and "test_accuracy_mean" in mean_std_df.columns:
    display(mean_std_df.sort_values(["test_accuracy_mean", "test_roc_auc_mean"], ascending=False))
else:
    display(mean_std_df)

display(Markdown("# Stage 4 v2 Priority 10 Collapse Summary"))
display(collapse_df)

print("\nDONE", flush=True)
print("Outputs:", PROJECT_ROOT / "outputs/stage4", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Collapse CSV:", collapse_csv, flush=True)
print("Run summary JSON:", run_summary_json, flush=True)
print("Result bundle:", bundle_path, flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```
