# Kaggle Stage 4 N15-A - Rebuild I60/R20 Four-Image Stage 2 Checkpoints

## English

This cell rebuilds the Stage 2 BTC baselines needed for the Stage 4 N15
image-spec context-complement experiments.

It trains/evaluates:

- `I60/R20/ohlc`
- `I60/R20/ohlc_ma`
- `I60/R20/ohlc_vb`
- `I60/R20/ohlc_ma_vb`
- seeds `42, 43, 44, 45, 46`

It creates one downloadable bundle:

```text
/kaggle/working/stage2_i60_r20_four_image_specs_seed42_46_checkpoints_for_stage4_n15.zip
```

Use that bundle for N15-B/C so each image spec uses its own Stage 2 checkpoint.

## 한국어

이 cell은 Stage 4 N15 image-spec context-complement 실험에 필요한 Stage 2
BTC baseline checkpoint를 다시 만듭니다.

실행 대상:

- `I60/R20/ohlc`
- `I60/R20/ohlc_ma`
- `I60/R20/ohlc_vb`
- `I60/R20/ohlc_ma_vb`
- seed `42, 43, 44, 45, 46`

생성되는 bundle:

```text
/kaggle/working/stage2_i60_r20_four_image_specs_seed42_46_checkpoints_for_stage4_n15.zip
```

N15-B/C에서는 이 bundle을 Stage 2 pretrained checkpoint source로 사용합니다.

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
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

SOURCE_FILE = "/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv"

IMAGE_WINDOW = 60
IMAGE_SPECS = ["ohlc", "ohlc_ma", "ohlc_vb", "ohlc_ma_vb"]
RETURN_HORIZON = 20
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"

SUMMARY_NAME = "stage2_n15a_i60_r20_four_image_specs_five_seed"
BUNDLE_PATH = Path("/kaggle/working/stage2_i60_r20_four_image_specs_seed42_46_checkpoints_for_stage4_n15.zip")

# Strict Stage 2 comparison settings.
BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = False

# Set True only for a quick code-path check. For the real checkpoint rebuild, keep False.
SMOKE_TEST = False


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


def run(cmd, cwd=PROJECT_ROOT):
    """Run one command and stream output."""

    cmd = [str(item) for item in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def assert_latest_stage2_code(project_root: Path):
    """Fail early if the uploaded Stage 2 snapshot is stale/incomplete."""

    required = [
        "scripts/run_stage2_btc_baseline.py",
        "scripts/evaluate_stage2_predictions.py",
        "scripts/evaluate_stage2_trading.py",
        "src/stage2_btc/models/stock_cnn.py",
        "src/stage2_btc/runners/btc_baseline.py",
        "src/stage2_btc/training/loop.py",
    ]
    missing = [path for path in required if not (project_root / path).exists()]
    if missing:
        raise RuntimeError(
            "Stage 2 code snapshot is stale/incomplete. "
            "Upload the latest stage2_btc_extension dataset. Missing: "
            + ", ".join(missing)
        )


def experiment_name(image_spec: str) -> str:
    return f"stage2_i{IMAGE_WINDOW}_{image_spec}_r{RETURN_HORIZON}"


def patch_kaggle_config():
    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES
    cfg["evaluation"]["batch_size"] = BATCH_SIZE
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Config patched:", config_path, flush=True)


def output_paths(image_spec: str, run_seed: int) -> dict[str, Path]:
    exp = experiment_name(image_spec)
    seed_dir = f"seed_{run_seed}"
    root = PROJECT_ROOT / "outputs/stage2"
    return {
        "best_checkpoint": root / "checkpoints" / exp / seed_dir / "best.pt",
        "last_checkpoint": root / "checkpoints" / exp / seed_dir / "last.pt",
        "train_history": root / "metrics" / exp / seed_dir / "train_history.csv",
        "train_metadata": root / "metrics" / exp / seed_dir / "train_metadata.json",
        "predictions": root / "predictions" / exp / seed_dir / f"{EVAL_SPLIT}_predictions.csv",
        "classification_metrics": root / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json",
        "trading_metrics": root / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json",
        "run_manifest": root / "run_manifests" / exp / seed_dir / "run_manifest.json",
    }


def is_seed_complete(image_spec: str, run_seed: int) -> bool:
    required = output_paths(image_spec, run_seed)
    needed = [
        "best_checkpoint",
        "train_history",
        "train_metadata",
        "predictions",
        "classification_metrics",
        "trading_metrics",
        "run_manifest",
    ]
    return all(required[label].exists() for label in needed)


def read_metric_row(image_spec: str, run_seed: int, status: str = "ok", error: str = "") -> dict:
    paths = output_paths(image_spec, run_seed)
    row = {
        "experiment_name": experiment_name(image_spec),
        "image_window": IMAGE_WINDOW,
        "image_spec": image_spec,
        "return_horizon": RETURN_HORIZON,
        "run_seed": run_seed,
        "status": status,
        "error": error,
        "checkpoint_available": paths["best_checkpoint"].exists(),
        "metrics_available": paths["classification_metrics"].exists(),
        "trading_available": paths["trading_metrics"].exists(),
    }
    if paths["classification_metrics"].exists():
        metrics = json.loads(paths["classification_metrics"].read_text(encoding="utf-8"))
        for key in [
            "num_samples",
            "accuracy",
            "majority_class_accuracy",
            "accuracy_minus_majority_class_accuracy",
            "roc_auc",
            "average_precision",
            "f1",
            "brier_score",
            "positive_rate",
            "predicted_positive_rate",
        ]:
            row[key] = metrics.get(key)
    if paths["trading_metrics"].exists():
        trading = json.loads(paths["trading_metrics"].read_text(encoding="utf-8"))
        for strategy in ["long_flat", "long_short"]:
            values = trading.get(strategy, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                row[f"{strategy}_{key}"] = values.get(key)
    return row


def summarize_seed_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "accuracy_minus_majority_class_accuracy",
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
    rows = []
    for image_spec, frame in seed_df.groupby("image_spec", sort=False):
        row = {
            "experiment_name": experiment_name(str(image_spec)),
            "image_window": IMAGE_WINDOW,
            "image_spec": str(image_spec),
            "return_horizon": RETURN_HORIZON,
            "seed_count": int(frame["run_seed"].nunique()),
        }
        for column in numeric:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                row[f"{column}_mean"] = values.mean()
                row[f"{column}_std"] = values.std(ddof=1)
                row[f"{column}_count"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def make_checkpoint_bundle(seed_df: pd.DataFrame, mean_std_df: pd.DataFrame) -> Path:
    """Create one downloadable zip containing Stage 2 checkpoints and metrics."""

    tables_root = PROJECT_ROOT / "reports/tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    seed_csv = tables_root / f"{SUMMARY_NAME}_seed_results.csv"
    mean_std_csv = tables_root / f"{SUMMARY_NAME}_mean_std_results.csv"
    manifest_path = tables_root / f"{SUMMARY_NAME}_checkpoint_bundle_manifest.json"
    seed_df.to_csv(seed_csv, index=False)
    mean_std_df.to_csv(mean_std_csv, index=False)

    files: list[Path] = [PROJECT_ROOT / "configs/env_kaggle.yaml", seed_csv, mean_std_csv]
    bundle_labels = [
        "best_checkpoint",
        "train_history",
        "train_metadata",
        "predictions",
        "classification_metrics",
        "trading_metrics",
        "run_manifest",
    ]
    for image_spec in IMAGE_SPECS:
        for run_seed in RUN_SEEDS:
            paths = output_paths(image_spec, run_seed)
            files.extend(paths[label] for label in bundle_labels)
    files = [path for path in files if path.exists()]

    manifest = {
        "status": "ok",
        "purpose": "Stage 2 I60/R20 four-image checkpoints for Stage 4 N15",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_window": IMAGE_WINDOW,
        "image_specs": IMAGE_SPECS,
        "return_horizon": RETURN_HORIZON,
        "run_seeds": RUN_SEEDS,
        "source_file": SOURCE_FILE,
        "project_root": str(PROJECT_ROOT),
        "bundle_path": str(BUNDLE_PATH),
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in files if path.is_relative_to(PROJECT_ROOT)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)

    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if not path.exists() or not path.is_file():
                continue
            try:
                arcname = path.relative_to(PROJECT_ROOT)
            except ValueError:
                arcname = Path(path.name)
            archive.write(path, arcname=str(arcname))

    print(
        "Bundle written:",
        BUNDLE_PATH,
        f"({BUNDLE_PATH.stat().st_size / (1024 * 1024):.3f} MB)",
        flush=True,
    )
    return BUNDLE_PATH


# ============================================================
# 1. Copy code and patch config
# ============================================================
copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_latest_stage2_code(PROJECT_ROOT)
patch_kaggle_config()
print("Stage 2 project ready:", PROJECT_ROOT, flush=True)
print(
    "N15-A checkpoint rebuild:",
    {
        "image_window": IMAGE_WINDOW,
        "image_specs": IMAGE_SPECS,
        "return_horizon": RETURN_HORIZON,
        "run_seeds": RUN_SEEDS,
        "smoke_test": SMOKE_TEST,
    },
    flush=True,
)


# ============================================================
# 2. Data audit
# ============================================================
run([
    sys.executable, "-u",
    "scripts/audit_btc_ohlcv.py",
    "--data-root", str(DATA_ROOT),
    "--output-dir", "reports/data_audit",
])


# ============================================================
# 3. Train/evaluate selected Stage 2 checkpoints
# ============================================================
smoke_train_args = []
smoke_eval_args = []
if SMOKE_TEST:
    smoke_train_args = [
        "--max-epochs", "2",
        "--max-train-rows", "128",
        "--max-validation-rows", "64",
        "--max-test-rows", "64",
    ]
    smoke_eval_args = [
        "--max-train-rows", "128",
        "--max-validation-rows", "64",
        "--max-test-rows", "64",
    ]

summary_rows = []
total_runs = len(IMAGE_SPECS) * len(RUN_SEEDS)
run_index = 0

for image_spec in IMAGE_SPECS:
    for run_seed in RUN_SEEDS:
        run_index += 1
        exp = experiment_name(image_spec)
        print("\n" + "=" * 96, flush=True)
        print(f"[{run_index}/{total_runs}] Stage 2 N15-A rebuild: {exp}, seed={run_seed}", flush=True)
        print("=" * 96, flush=True)

        if SKIP_COMPLETED and is_seed_complete(image_spec, run_seed):
            print(f"[skip] outputs already complete for {exp}, seed={run_seed}", flush=True)
            summary_rows.append(read_metric_row(image_spec, run_seed, status="skipped_completed"))
            continue

        try:
            run([
                sys.executable, "-u",
                "scripts/run_stage2_btc_baseline.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", image_spec,
                "--return-horizon", str(RETURN_HORIZON),
                "--run-seed", str(run_seed),
            ] + smoke_train_args)

            run([
                sys.executable, "-u",
                "scripts/evaluate_stage2_predictions.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", image_spec,
                "--return-horizon", str(RETURN_HORIZON),
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
            ] + smoke_eval_args)

            run([
                sys.executable, "-u",
                "scripts/evaluate_stage2_trading.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", image_spec,
                "--return-horizon", str(RETURN_HORIZON),
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
            ])

            summary_rows.append(read_metric_row(image_spec, run_seed, status="ok"))
        except Exception as exc:
            print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
            summary_rows.append(read_metric_row(image_spec, run_seed, status="failed", error=str(exc)))
            if not CONTINUE_ON_ERROR:
                raise


# ============================================================
# 4. Verify outputs and bundle checkpoint artifacts
# ============================================================
missing = []
for image_spec in IMAGE_SPECS:
    for run_seed in RUN_SEEDS:
        required = output_paths(image_spec, run_seed)
        for label in [
            "best_checkpoint",
            "train_history",
            "train_metadata",
            "predictions",
            "classification_metrics",
            "trading_metrics",
            "run_manifest",
        ]:
            path = required[label]
            if not path.exists():
                missing.append({"image_spec": image_spec, "run_seed": run_seed, "label": label, "path": str(path)})
if missing:
    raise RuntimeError("Missing required Stage 2 N15-A artifact(s): " + json.dumps(missing, indent=2))

seed_df = pd.DataFrame(summary_rows)
mean_std_df = summarize_seed_rows(seed_df)
bundle_path = make_checkpoint_bundle(seed_df, mean_std_df)

display(Markdown("# Stage 2 N15-A Seed-Level Results"))
display(seed_df.sort_values(["image_spec", "run_seed"]))

display(Markdown("# Stage 2 N15-A Mean/Std Summary"))
display(mean_std_df.sort_values("accuracy_mean", ascending=False))

display(Markdown("# Stage 2 N15-A Accuracy Pivot"))
try:
    display(mean_std_df.pivot_table(index=["image_window", "return_horizon"], columns="image_spec", values="accuracy_mean"))
except Exception as exc:
    print("pivot failed:", exc)

print("\nDONE", flush=True)
print("Seed CSV:", PROJECT_ROOT / f"reports/tables/{SUMMARY_NAME}_seed_results.csv", flush=True)
print("Mean/std CSV:", PROJECT_ROOT / f"reports/tables/{SUMMARY_NAME}_mean_std_results.csv", flush=True)
print("Bundle:", bundle_path, flush=True)
```
