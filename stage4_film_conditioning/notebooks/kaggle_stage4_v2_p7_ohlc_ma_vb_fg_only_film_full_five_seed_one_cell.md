# Kaggle Stage 4 v2 Priority 7 OHLC_MA_VB F&G-Only FiLM-Full Five-Seed One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Fear & Greed Kaggle dataset:
  `ashishpatel8736/historical-and-fear-greed-index-datasets`

This runner executes Stage 4 v2 priority 7:

```text
I60 / R20 / ohlc_ma_vb / context_window=60
F&G-only context x film_full x 5 seeds = 5 runs

features: fg_value, fg_mean_60, fg_delta_60, fg_std_60
method: film_full
seeds: 42, 43, 44, 45, 46
```

It supports resume-style execution with `SKIP_COMPLETED=True`. Completion is
checked using `check_stage4_outputs.py` plus `MIN_PREDICTIONS=1000`, so old
smoke outputs will not be accepted as completed full runs.

Disk-space note:
- Five-seed runs can fill `/kaggle/working` if every intermediate backup zip is
  kept.
- Default settings below use `SAVE_BACKUP_ZIPS=False` and
  `DELETE_EXISTING_BACKUP_ZIPS_ON_START=True`.
- If a previous run stopped with `No space left on device`, set
  `RESUME_EXISTING_PROJECT=True` and rerun this cell. It will keep the existing
  project outputs, delete old backup zips, skip completed method/seed runs, and
  continue.

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
CONTEXT_METHODS = ["film_full"]
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
SAVE_BACKUP_ZIPS = False
RESUME_EXISTING_PROJECT = False
DELETE_EXISTING_BACKUP_ZIPS_ON_START = True

# Smoke check only. For the real 4-V6 run, keep False.
SMOKE_TEST = False

# Strict Stage 2-style comparison settings.
BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20


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


def experiment_name(context_method: str) -> str:
    return (
        f"stage4_{context_method}_i{IMAGE_WINDOW}_"
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{CONTEXT_FEATURE_SET_NAME}"
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
        "scripts/check_stage4_outputs.py",
        "src/stage4_film/models/context_stock_cnn.py",
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

    marker_checks = {
        "src/stage4_film/config.py": "experiment_suffix",
        "src/stage4_film/context/features.py": "context_suffix",
    }
    stale = []
    for rel_path, marker in marker_checks.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            stale.append(f"{rel_path} missing marker {marker!r}")
    if stale:
        raise RuntimeError(
            "Stage 4 code snapshot is stale. Upload the latest "
            "stage4_film_conditioning folder/zip, then rerun. Problems: "
            + "; ".join(stale)
        )


def patch_kaggle_config():
    """Patch Stage 4 Kaggle config with the current notebook paths/settings."""

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
    cfg["stage4_model"]["experiment_suffix"] = CONTEXT_FEATURE_SET_NAME
    cfg["stage4_model"]["context_methods"] = list(CONTEXT_METHODS)

    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    cfg["evaluation"]["batch_size"] = BATCH_SIZE

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Config patched:", config_path, flush=True)


def expected_output_paths(context_method: str, run_seed: int) -> dict[str, str]:
    exp = experiment_name(context_method)
    seed_dir = f"seed_{run_seed}"
    outputs = PROJECT_ROOT / "outputs/stage4"
    gradcam_dir = outputs / "figures" / exp / seed_dir / "gradcam" / EVAL_SPLIT
    context_dir = outputs / "context" / context_name() / seed_dir
    return {
        "best_checkpoint": str(outputs / "checkpoints" / exp / seed_dir / "best.pt"),
        "predictions": str(outputs / "predictions" / exp / seed_dir / f"{EVAL_SPLIT}_predictions.csv"),
        "classification_metrics": str(outputs / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json"),
        "trading_metrics": str(outputs / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json"),
        "gradcam": str(gradcam_dir / f"btc_context_gradcam_{EVAL_SPLIT}_{GRADCAM_SAMPLES_PER_CLASS}perclass.png"),
        "context_features": str(context_dir / "context_features.csv"),
        "run_manifest": str(outputs / "run_manifests" / exp / seed_dir / "run_manifest.json"),
    }


def backup_outputs(label: str, phase: str, context_method: str | None = None, run_seed: int | None = None):
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
    archive_base = BACKUP_ROOT / f"{label}_{seed_label}_{phase}_{timestamp}_outputs"
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
        "run_seed": run_seed,
        "phase": phase,
        "created_utc": timestamp,
        "archive_path": str(archive_path),
        "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 3),
        "project_root": str(PROJECT_ROOT),
        "outputs_root": str(outputs_root),
        "expected_paths": (
            expected_output_paths(context_method, int(run_seed))
            if context_method and run_seed is not None
            else {}
        ),
    }
    receipt_path = BACKUP_ROOT / f"{label}_{seed_label}_{phase}_{timestamp}_receipt.json"
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


def run_step(label: str, phase: str, cmd, context_method: str | None = None, run_seed: int | None = None):
    """Run a stage and backup current outputs even if the stage fails."""

    try:
        return run(cmd)
    finally:
        backup_outputs(label, phase, context_method=context_method, run_seed=run_seed)


def output_check_cmd(context_method: str, run_seed: int):
    return [
        sys.executable, "-u",
        "scripts/check_stage4_outputs.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--context-method", context_method,
        "--run-seed", str(run_seed),
        "--split", EVAL_SPLIT,
        "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
        "--min-predictions", str(MIN_PREDICTIONS),
    ]


def is_completed(context_method: str, run_seed: int) -> bool:
    result = run(output_check_cmd(context_method, run_seed), capture=True, check=False)
    return result.returncode == 0


def read_result_row(context_method: str, run_seed: int, status: str, error: str = "") -> dict:
    exp = experiment_name(context_method)
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_trading_metrics.json"
    row = {
        "priority": "4-V6",
        "experiment_name": exp,
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "context_window": CONTEXT_WINDOW,
        "context_feature_set_name": CONTEXT_FEATURE_SET_NAME,
        "context_scope": "fear_greed_only",
        "context_features": ",".join(PRIMARY_CONTEXT_FEATURES),
        "context_method": context_method,
        "run_seed": run_seed,
        "status": status,
        "error": error,
        "classification_available": metrics_path.exists(),
        "trading_available": trading_path.exists(),
        "stage2_ohlc_baseline_accuracy_mean": 0.558085,
        "stage2_ohlc_baseline_roc_auc_mean": 0.560218,
        "stage2_ohlc_ma_vb_baseline_accuracy_mean": 0.579320,
        "stage2_ohlc_ma_vb_baseline_roc_auc_mean": 0.584862,
        "stage4_v2_all_context_seed42_accuracy": 0.572519,
        "stage4_v2_all_context_seed42_roc_auc": 0.557328,
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in [
            "num_samples",
            "accuracy",
            "majority_class_accuracy",
            "accuracy_minus_majority",
            "roc_auc",
            "average_precision",
            "f1",
            "brier_score",
            "positive_rate",
            "predicted_positive_rate",
        ]:
            row[key] = metrics.get(key)
        if row.get("accuracy_minus_majority") is None and row.get("accuracy") is not None:
            majority = row.get("majority_class_accuracy")
            if majority is not None:
                row["accuracy_minus_majority"] = float(row["accuracy"]) - float(majority)
    if trading_path.exists():
        trading = json.loads(trading_path.read_text(encoding="utf-8"))
        for strategy_name in ["long_flat", "long_short"]:
            values = trading.get(strategy_name, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                row[f"{strategy_name}_{key}"] = values.get(key)
    return row


def summarize_seed_results(seed_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(seed_rows)
    numeric = [
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "accuracy_minus_majority",
        "roc_auc",
        "average_precision",
        "f1",
        "brier_score",
        "positive_rate",
        "predicted_positive_rate",
        "long_flat_sharpe_net",
        "long_short_sharpe_net",
        "long_flat_annualized_return_net",
        "long_short_annualized_return_net",
    ]
    available = [column for column in numeric if column in df.columns]
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("context_method", dropna=False)
    rows = []
    for method, frame in grouped:
        row = {"context_method": method, "seed_count": int(frame["run_seed"].nunique())}
        for column in available:
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)
            row[f"{column}_count"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


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
patch_kaggle_config()
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
# 3. Train/evaluate/check F&G-only FiLM-full over five seeds
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
total_runs = len(CONTEXT_METHODS) * len(RUN_SEEDS)
run_index = 0
for run_seed in RUN_SEEDS:
    print("\n" + "#" * 80, flush=True)
    print(f"Seed {run_seed}", flush=True)
    print("#" * 80, flush=True)

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

    for context_method in CONTEXT_METHODS:
        run_index += 1
        exp = experiment_name(context_method)
        print("\n" + "=" * 80, flush=True)
        print(f"[{run_index}/{total_runs}] {exp}, seed={run_seed}", flush=True)
        print("=" * 80, flush=True)

        if SKIP_COMPLETED and is_completed(context_method, run_seed):
            print(f"[skip] Output check already passes for {exp}, seed={run_seed}", flush=True)
            summary_rows.append(read_result_row(context_method, run_seed, status="skipped_completed"))
            continue

        try:
            run_step(exp, "after_train", [
                sys.executable, "-u",
                "scripts/run_stage4_context_model.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", IMAGE_SPEC,
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", context_method,
                "--run-seed", str(run_seed),
            ] + smoke_train_args, context_method=context_method, run_seed=run_seed)

            run_step(exp, "after_prediction_eval", [
                sys.executable, "-u",
                "scripts/evaluate_stage4_predictions.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", IMAGE_SPEC,
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", context_method,
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
            ] + smoke_data_args, context_method=context_method, run_seed=run_seed)

            run_step(exp, "after_trading_eval", [
                sys.executable, "-u",
                "scripts/evaluate_stage4_trading.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", IMAGE_SPEC,
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", context_method,
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
            ], context_method=context_method, run_seed=run_seed)

            run_step(exp, "after_gradcam", [
                sys.executable, "-u",
                "scripts/generate_stage4_gradcam_context.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", IMAGE_SPEC,
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", context_method,
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
                "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
                "--write-report-copy",
            ] + smoke_data_args, context_method=context_method, run_seed=run_seed)

            run_step(
                exp,
                "after_output_check",
                output_check_cmd(context_method, run_seed),
                context_method=context_method,
                run_seed=run_seed,
            )
            summary_rows.append(read_result_row(context_method, run_seed, status="ok"))
        except Exception as exc:
            print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
            summary_rows.append(read_result_row(context_method, run_seed, status="failed", error=str(exc)))
            if not CONTINUE_ON_ERROR:
                raise


# ============================================================
# 4. Write seed-level and mean/std summary tables
# ============================================================
tables_root = PROJECT_ROOT / "reports/tables"
tables_root.mkdir(parents=True, exist_ok=True)
seed_df = pd.DataFrame(summary_rows)
mean_std_df = summarize_seed_results(summary_rows)
if not seed_df.empty and "accuracy" in seed_df.columns:
    seed_df["accuracy_minus_stage2_ohlc_mean"] = (
        pd.to_numeric(seed_df["accuracy"], errors="coerce") - 0.558085
    )
    seed_df["roc_auc_minus_stage2_ohlc_mean"] = (
        pd.to_numeric(seed_df["roc_auc"], errors="coerce") - 0.560218
    )
if not mean_std_df.empty and "accuracy_mean" in mean_std_df.columns:
    mean_std_df["accuracy_mean_minus_stage2_ohlc_mean"] = (
        pd.to_numeric(mean_std_df["accuracy_mean"], errors="coerce") - 0.558085
    )
    mean_std_df["roc_auc_mean_minus_stage2_ohlc_mean"] = (
        pd.to_numeric(mean_std_df["roc_auc_mean"], errors="coerce") - 0.560218
    )

seed_csv = tables_root / "stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_seed_results.csv"
mean_std_csv = tables_root / "stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_mean_std_results.csv"
run_summary_json = tables_root / "stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed_run_summary.json"
seed_df.to_csv(seed_csv, index=False)
mean_std_df.to_csv(mean_std_csv, index=False)
run_summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

backup_outputs("stage4_v2_p7_ohlc_ma_vb_fg_only_film_full_five_seed", "after_summary", context_method=None, run_seed=None)

display(Markdown("# Stage 4 v2 Priority 7 OHLC_MA_VB F&G-Only FiLM-Full Five-Seed Seed-Level Results"))
display(seed_df.sort_values(["context_method", "run_seed"]))

display(Markdown("# Stage 4 v2 Priority 7 OHLC_MA_VB F&G-Only FiLM-Full Five-Seed Mean/Std Summary"))
if not mean_std_df.empty and "accuracy_mean" in mean_std_df.columns:
    display(mean_std_df.sort_values("accuracy_mean", ascending=False))
else:
    display(mean_std_df)

print("\nDONE", flush=True)
print("Outputs:", PROJECT_ROOT / "outputs/stage4", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```
