# Kaggle Stage 4 v2 Priority 3 OHLC All-Context FiLM-Full One Cell

Copy the Python cell below into a Kaggle notebook after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Fear & Greed Kaggle dataset:
  `ashishpatel8736/historical-and-fear-greed-index-datasets`

This runner executes Stage 4 v2 priority 3:

```text
I60 / R20 / ohlc / context_window=60 / all structured context / film_full / seed=42
```

The run executes:

```text
context build -> train -> prediction metrics -> trading metrics
-> Grad-CAM/context/modulation export -> output check -> compact summary
```

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
IMAGE_SPEC = "ohlc"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHODS = ["film_full"]
RUN_SEED = 42
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
SAVE_BACKUP_ZIPS = False

# Smoke check only. For the real 4-V2 run, keep False.
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
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}"
    )


def context_name() -> str:
    return (
        f"stage4_context_i{IMAGE_WINDOW}_"
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}"
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
    cfg["context"]["fear_greed"]["kaggle_file"] = FEAR_GREED_FILE

    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON

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


def expected_output_paths(context_method: str) -> dict[str, str]:
    exp = experiment_name(context_method)
    seed_dir = f"seed_{RUN_SEED}"
    outputs = PROJECT_ROOT / "outputs/stage4"
    gradcam_dir = outputs / "figures" / exp / seed_dir / "gradcam" / EVAL_SPLIT
    context_dir = outputs / "context" / context_name() / seed_dir
    return {
        "best_checkpoint": str(outputs / "checkpoints" / exp / seed_dir / "best.pt"),
        "last_checkpoint": str(outputs / "checkpoints" / exp / seed_dir / "last.pt"),
        "train_history": str(outputs / "metrics" / exp / seed_dir / "train_history.csv"),
        "train_metadata": str(outputs / "metrics" / exp / seed_dir / "train_metadata.json"),
        "predictions": str(outputs / "predictions" / exp / seed_dir / f"{EVAL_SPLIT}_predictions.csv"),
        "classification_metrics": str(outputs / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json"),
        "trading_metrics": str(outputs / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json"),
        "gradcam": str(gradcam_dir / f"btc_context_gradcam_{EVAL_SPLIT}_{GRADCAM_SAMPLES_PER_CLASS}perclass.png"),
        "gradcam_samples": str(gradcam_dir / "samples.csv"),
        "modulation_summary": str(gradcam_dir / "modulation_summary.csv"),
        "modulation_values": str(gradcam_dir / "modulation_values.json"),
        "context_features": str(context_dir / "context_features.csv"),
        "context_scaler": str(context_dir / "context_scaler.json"),
        "context_feature_audit": str(context_dir / "context_feature_audit.json"),
        "context_feature_summary": str(context_dir / "context_feature_summary.csv"),
        "run_manifest": str(outputs / "run_manifests" / exp / seed_dir / "run_manifest.json"),
    }


def backup_outputs(label: str, phase: str, context_method: str | None = None):
    """Zip current Stage 4 outputs outside PROJECT_ROOT so later reruns cannot erase them."""

    if not SAVE_BACKUP_ZIPS:
        return None
    outputs_root = PROJECT_ROOT / "outputs/stage4"
    if not outputs_root.exists():
        print(f"[backup:{label}:{phase}] skip: no outputs yet", flush=True)
        return None

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_base = BACKUP_ROOT / f"{label}_seed{RUN_SEED}_{phase}_{timestamp}_outputs"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", outputs_root))
    receipt = {
        "label": label,
        "context_method": context_method,
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "context_window": CONTEXT_WINDOW,
        "run_seed": RUN_SEED,
        "phase": phase,
        "created_utc": timestamp,
        "archive_path": str(archive_path),
        "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 3),
        "project_root": str(PROJECT_ROOT),
        "outputs_root": str(outputs_root),
        "expected_paths": expected_output_paths(context_method) if context_method else {},
    }
    receipt_path = BACKUP_ROOT / f"{label}_seed{RUN_SEED}_{phase}_{timestamp}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        f"[backup:{label}:{phase}] saved {archive_path} "
        f"({receipt['archive_size_mb']} MB), receipt={receipt_path}",
        flush=True,
    )
    return archive_path


def run_step(label: str, phase: str, cmd, context_method: str | None = None):
    """Run a stage and backup current outputs even if the stage fails."""

    try:
        return run(cmd)
    finally:
        backup_outputs(label, phase, context_method=context_method)


def output_check_cmd(context_method: str):
    return [
        sys.executable, "-u",
        "scripts/check_stage4_outputs.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--context-method", context_method,
        "--run-seed", str(RUN_SEED),
        "--split", EVAL_SPLIT,
        "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
        "--min-predictions", str(MIN_PREDICTIONS),
    ]


def is_completed(context_method: str) -> bool:
    result = run(output_check_cmd(context_method), capture=True, check=False)
    return result.returncode == 0


def read_result_row(context_method: str, status: str, error: str = "") -> dict:
    exp = experiment_name(context_method)
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{RUN_SEED}" / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{RUN_SEED}" / f"{EVAL_SPLIT}_trading_metrics.json"
    row = {
        "priority": "4-V2",
        "experiment_name": exp,
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "context_window": CONTEXT_WINDOW,
        "context_scope": "all_structured_context",
        "context_method": context_method,
        "run_seed": RUN_SEED,
        "status": status,
        "error": error,
        "classification_available": metrics_path.exists(),
        "trading_available": trading_path.exists(),
        "stage2_ohlc_baseline_accuracy_mean": 0.558085,
        "stage2_ohlc_baseline_roc_auc_mean": 0.560218,
        "stage2_ohlc_ma_vb_baseline_accuracy_mean": 0.579320,
        "stage2_ohlc_ma_vb_baseline_roc_auc_mean": 0.584862,
        "stage4_v1_film_full_ohlc_ma_vb_accuracy_mean": 0.551006,
        "stage4_v1_film_full_ohlc_ma_vb_roc_auc_mean": 0.567679,
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


# ============================================================
# 1. Copy code snapshots and patch config
# ============================================================
copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_required_code()
patch_kaggle_config()
print(f"Stage 4 code copied to: {PROJECT_ROOT}", flush=True)
print(f"Stage 2 dependency copied to: {STAGE2_PROJECT_ROOT}", flush=True)


# ============================================================
# 2. Source audit and context build
# ============================================================
run([
    sys.executable, "-u",
    "scripts/audit_stage4_context_sources.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--return-horizon", str(RETURN_HORIZON),
    "--output", "reports/tables/stage4_context_source_audit.json",
])

run_step("stage4_context", "after_context_build", [
    sys.executable, "-u",
    "scripts/build_stage4_context_features.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--write-report-copy",
])


# ============================================================
# 3. Train/evaluate/check priority-3 FiLM-full run
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
for index, context_method in enumerate(CONTEXT_METHODS, start=1):
    exp = experiment_name(context_method)
    print("\n" + "=" * 80, flush=True)
    print(f"[{index}/{len(CONTEXT_METHODS)}] {exp}", flush=True)
    print("=" * 80, flush=True)

    if SKIP_COMPLETED and is_completed(context_method):
        print(f"[skip] Output check already passes for {exp}", flush=True)
        summary_rows.append(read_result_row(context_method, status="skipped_completed"))
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
            "--run-seed", str(RUN_SEED),
        ] + smoke_train_args, context_method=context_method)

        run_step(exp, "after_prediction_eval", [
            sys.executable, "-u",
            "scripts/evaluate_stage4_predictions.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--context-method", context_method,
            "--run-seed", str(RUN_SEED),
            "--split", EVAL_SPLIT,
        ] + smoke_data_args, context_method=context_method)

        run_step(exp, "after_trading_eval", [
            sys.executable, "-u",
            "scripts/evaluate_stage4_trading.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--context-method", context_method,
            "--run-seed", str(RUN_SEED),
            "--split", EVAL_SPLIT,
        ], context_method=context_method)

        run_step(exp, "after_gradcam", [
            sys.executable, "-u",
            "scripts/generate_stage4_gradcam_context.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--context-method", context_method,
            "--run-seed", str(RUN_SEED),
            "--split", EVAL_SPLIT,
            "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
            "--write-report-copy",
        ] + smoke_data_args, context_method=context_method)

        run_step(exp, "after_output_check", output_check_cmd(context_method), context_method=context_method)
        summary_rows.append(read_result_row(context_method, status="ok"))
    except Exception as exc:
        print(f"[error] {exp}: {exc}", flush=True)
        summary_rows.append(read_result_row(context_method, status="failed", error=str(exc)))
        if not CONTINUE_ON_ERROR:
            raise


# ============================================================
# 4. Write compact priority-3 summary table and optional final backup
# ============================================================
tables_root = PROJECT_ROOT / "reports/tables"
tables_root.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty and "accuracy" in summary_df.columns:
    summary_df["accuracy_minus_stage2_ohlc_mean"] = (
        pd.to_numeric(summary_df["accuracy"], errors="coerce")
        - summary_df["stage2_ohlc_baseline_accuracy_mean"]
    )
    summary_df["roc_auc_minus_stage2_ohlc_mean"] = (
        pd.to_numeric(summary_df["roc_auc"], errors="coerce")
        - summary_df["stage2_ohlc_baseline_roc_auc_mean"]
    )
    summary_df["accuracy_minus_4v0_seed42"] = (
        pd.to_numeric(summary_df["accuracy"], errors="coerce") - 0.603053
    )
    summary_df["roc_auc_minus_4v0_seed42"] = (
        pd.to_numeric(summary_df["roc_auc"], errors="coerce") - 0.616950
    )
summary_csv = tables_root / "stage4_v2_p3_ohlc_all_context_film_full_seed42_run_summary.csv"
summary_json = tables_root / "stage4_v2_p3_ohlc_all_context_film_full_seed42_run_summary.json"
summary_df.to_csv(summary_csv, index=False)
summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

backup_outputs("stage4_v2_p3_ohlc_all_context_film_full", "after_summary", context_method=None)

display(Markdown("# Stage 4 v2 Priority 3 OHLC All-Context FiLM-Full Seed 42 Summary"))
display(summary_df)

if not summary_df.empty and "accuracy" in summary_df.columns:
    display(Markdown("## Key Comparison"))
    comparison_cols = [
        "priority",
        "experiment_name",
        "accuracy",
        "accuracy_minus_stage2_ohlc_mean",
        "roc_auc",
        "roc_auc_minus_stage2_ohlc_mean",
        "predicted_positive_rate",
        "f1",
        "brier_score",
        "long_flat_sharpe_net",
        "long_short_sharpe_net",
    ]
    display(summary_df[[col for col in comparison_cols if col in summary_df.columns]])

print("\nDONE", flush=True)
print("Outputs:", PROJECT_ROOT / "outputs/stage4", flush=True)
print("Summary CSV:", summary_csv, flush=True)
print("Backup zips (disabled by default):", BACKUP_ROOT, flush=True)
```
