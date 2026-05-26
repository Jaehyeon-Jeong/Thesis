# Kaggle Stage 4 v2 Priority 1 Visual-Only Same-Split One Cell

Copy the Python cell below into a Kaggle notebook after attaching:

- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`

This cell is Stage 4 v2 priority 1:

```text
I60 / R20 / ohlc_ma_vb / no context / visual-only CNN
```

It deliberately uses the Stage 2 BTC runner because this control has no context
branch. The goal is not to introduce a new Stage 4 model, but to confirm the
visual-only baseline under the same selected `I60/R20/ohlc_ma_vb` setup before
changing FiLM.

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
STAGE2_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")
REPORT_ROOT = Path("/kaggle/working/stage4_v2_visual_only_reports")

# Leave empty to auto-detect btc_1d_data_2018_to_2025.csv under /kaggle/input.
SOURCE_FILE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEEDS = [42]
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
SAVE_FINAL_OUTPUT_ZIP = True

# Smoke check only. For the real priority-1 run, keep False.
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


def run(cmd, cwd=STAGE2_PROJECT_ROOT, capture=False, check=True):
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


def experiment_name() -> str:
    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def assert_required_code():
    """Fail early if the uploaded Stage 2 code snapshot is stale or incomplete."""

    required = [
        "scripts/audit_btc_ohlcv.py",
        "scripts/run_stage2_btc_baseline.py",
        "scripts/evaluate_stage2_predictions.py",
        "scripts/evaluate_stage2_trading.py",
        "scripts/generate_stage2_gradcam.py",
        "scripts/check_stage2_outputs.py",
        "src/stage2_btc/data",
        "src/stage2_btc/evaluation",
        "src/stage2_btc/models/stock_cnn.py",
    ]
    missing = [path for path in required if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing:
        raise RuntimeError("Stage 2 code snapshot is incomplete: " + ", ".join(missing))


def patch_stage2_kaggle_config():
    """Patch Stage 2 Kaggle config with current notebook paths/settings."""

    config_path = STAGE2_PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cfg["paths"]["project_root"] = str(STAGE2_PROJECT_ROOT)
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
    cfg["paths"]["output_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2")
    cfg["paths"]["checkpoint_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/checkpoints")
    cfg["paths"]["metrics_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/metrics")
    cfg["paths"]["predictions_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/predictions")
    cfg["paths"]["figures_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/figures")
    cfg["paths"]["run_manifest_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/run_manifests")
    cfg["paths"]["reports_root"] = str(STAGE2_PROJECT_ROOT / "reports")
    cfg["paths"]["tables_root"] = str(STAGE2_PROJECT_ROOT / "reports/tables")

    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

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


def output_paths(run_seed: int) -> dict[str, Path]:
    exp = experiment_name()
    seed_dir = f"seed_{run_seed}"
    outputs = STAGE2_PROJECT_ROOT / "outputs/stage2"
    gradcam_dir = outputs / "figures" / exp / seed_dir / "gradcam" / EVAL_SPLIT
    return {
        "checkpoint": outputs / "checkpoints" / exp / seed_dir / "best.pt",
        "train_history": outputs / "metrics" / exp / seed_dir / "train_history.csv",
        "train_metadata": outputs / "metrics" / exp / seed_dir / "train_metadata.json",
        "predictions": outputs / "predictions" / exp / seed_dir / f"{EVAL_SPLIT}_predictions.csv",
        "metrics": outputs / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json",
        "trading": outputs / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json",
        "gradcam": gradcam_dir / f"btc_gradcam_{EVAL_SPLIT}_{GRADCAM_SAMPLES_PER_CLASS}perclass.png",
        "manifest": outputs / "run_manifests" / exp / seed_dir / "run_manifest.json",
    }


def check_cmd(run_seed: int):
    return [
        sys.executable, "-u",
        "scripts/check_stage2_outputs.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(run_seed),
        "--split", EVAL_SPLIT,
        "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
    ]


def is_completed(run_seed: int) -> bool:
    result = run(check_cmd(run_seed), capture=True, check=False)
    if result.returncode != 0:
        return False
    paths = output_paths(run_seed)
    if not paths["predictions"].exists():
        return False
    try:
        predictions = pd.read_csv(paths["predictions"])
    except Exception:
        return False
    return len(predictions) >= MIN_PREDICTIONS


def read_result_row(run_seed: int, status: str, error: str = "") -> dict:
    paths = output_paths(run_seed)
    row = {
        "priority": "4-V0",
        "experiment_name": experiment_name(),
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "context": "none",
        "model": "stage2_visual_only_stock_cnn",
        "run_seed": run_seed,
        "status": status,
        "error": error,
        "metrics_available": paths["metrics"].exists(),
        "trading_available": paths["trading"].exists(),
        "checkpoint_available": paths["checkpoint"].exists(),
        "gradcam_available": paths["gradcam"].exists(),
    }
    if paths["metrics"].exists():
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        for key in [
            "num_samples",
            "accuracy",
            "majority_class_accuracy",
            "accuracy_minus_majority",
            "roc_auc",
            "f1",
            "brier_score",
            "positive_rate",
            "predicted_positive_rate",
        ]:
            row[key] = metrics.get(key)
    if paths["trading"].exists():
        trading = json.loads(paths["trading"].read_text(encoding="utf-8"))
        for strategy_name in ["long_flat", "long_short"]:
            values = trading.get(strategy_name, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                row[f"{strategy_name}_{key}"] = values.get(key)
    return row


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [
        "accuracy",
        "majority_class_accuracy",
        "accuracy_minus_majority",
        "roc_auc",
        "f1",
        "brier_score",
        "long_flat_sharpe_net",
        "long_short_sharpe_net",
        "long_flat_annualized_return_net",
        "long_short_annualized_return_net",
    ]
    summary = {
        "priority": "4-V0",
        "experiment_name": experiment_name(),
        "seed_count": int(df["run_seed"].nunique()) if not df.empty else 0,
        "stage2_selected_baseline_accuracy_mean": 0.579320,
        "stage2_selected_baseline_roc_auc_mean": 0.584862,
        "stage4_v1_film_full_accuracy_mean": 0.551006,
        "stage4_v1_film_full_roc_auc_mean": 0.567679,
    }
    for column in numeric:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            summary[f"{column}_mean"] = values.mean()
            summary[f"{column}_std"] = values.std(ddof=1)
            summary[f"{column}_count"] = int(values.notna().sum())
    return pd.DataFrame([summary])


def save_final_output_zip():
    if not SAVE_FINAL_OUTPUT_ZIP:
        return None
    outputs_root = STAGE2_PROJECT_ROOT / "outputs/stage2"
    if not outputs_root.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_base = REPORT_ROOT / f"stage4_v2_p1_visual_only_{experiment_name()}_{timestamp}_outputs"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", outputs_root))
    print(f"[backup] saved {archive_path} ({archive_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return archive_path


# ============================================================
# 1. Copy code and patch config
# ============================================================
copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_required_code()
patch_stage2_kaggle_config()
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Stage 2 project ready at: {STAGE2_PROJECT_ROOT}", flush=True)
print(f"Report root: {REPORT_ROOT}", flush=True)


# ============================================================
# 2. Source audit
# ============================================================
run([
    sys.executable, "-u",
    "scripts/audit_btc_ohlcv.py",
    "--data-root", str(DATA_ROOT),
    "--output-dir", "reports/data_audit",
])


# ============================================================
# 3. Train/evaluate visual-only priority-1 baseline
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

rows = []
for run_seed in RUN_SEEDS:
    print("\n" + "=" * 80, flush=True)
    print(f"4-V0 visual-only same-split baseline: {experiment_name()}, seed={run_seed}", flush=True)
    print("=" * 80, flush=True)

    if SKIP_COMPLETED and is_completed(run_seed):
        print(f"[skip] Output check already passes for {experiment_name()}, seed={run_seed}")
        rows.append(read_result_row(run_seed, status="skipped_completed"))
        continue

    try:
        run([
            sys.executable, "-u",
            "scripts/run_stage2_btc_baseline.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(run_seed),
        ] + smoke_train_args)

        run([
            sys.executable, "-u",
            "scripts/evaluate_stage2_predictions.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(run_seed),
            "--split", EVAL_SPLIT,
        ] + smoke_data_args)

        run([
            sys.executable, "-u",
            "scripts/evaluate_stage2_trading.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(run_seed),
            "--split", EVAL_SPLIT,
        ])

        run([
            sys.executable, "-u",
            "scripts/generate_stage2_gradcam.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(run_seed),
            "--split", EVAL_SPLIT,
            "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
            "--write-report-copy",
        ] + smoke_data_args)

        run(check_cmd(run_seed))
        rows.append(read_result_row(run_seed, status="ok"))
    except Exception as exc:
        print(f"[error] seed={run_seed}: {exc}", flush=True)
        rows.append(read_result_row(run_seed, status="failed", error=str(exc)))
        if not CONTINUE_ON_ERROR:
            raise


# ============================================================
# 4. Save report tables
# ============================================================
seed_df = pd.DataFrame(rows)
summary_df = summarize(rows)

seed_csv = REPORT_ROOT / "stage4_v2_p1_visual_only_seed_results.csv"
summary_csv = REPORT_ROOT / "stage4_v2_p1_visual_only_summary.csv"
seed_df.to_csv(seed_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

archive_path = save_final_output_zip()

display(Markdown("# Stage 4 v2 Priority 1: Visual-Only Same-Split Seed Results"))
display(seed_df)
display(Markdown("# Stage 4 v2 Priority 1 Summary"))
display(summary_df)

print("\nDONE", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Summary CSV:", summary_csv, flush=True)
if archive_path is not None:
    print("Output zip:", archive_path, flush=True)
```
