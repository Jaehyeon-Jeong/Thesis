# Kaggle Stage 4 N8-A0 - Rebuild Stage 2 Selected Checkpoints

## English

This cell rebuilds only the Stage 2 selected BTC baseline checkpoints needed
for Stage 4 N8 pretrained/frozen FiLM:

- `I60/R20/ohlc_ma_vb`
- seeds `42, 43, 44, 45, 46`

It does **not** rerun the full Stage 2 grid. It trains/evaluates five models,
writes seed-level and mean/std summary tables, verifies checkpoint artifacts,
and creates one downloadable checkpoint bundle for Stage 4 N8-A.

## 한국어

이 cell은 Stage 4 N8 pretrained/frozen FiLM에 필요한 Stage 2 selected BTC
baseline checkpoint만 다시 만듭니다.

- `I60/R20/ohlc_ma_vb`
- seed `42, 43, 44, 45, 46`

전체 Stage 2 grid를 다시 돌리지 않습니다. 5개 모델만 학습/평가하고, seed-level
및 mean/std summary table을 만든 뒤, Stage 4 N8-A에서 사용할 checkpoint bundle
zip을 하나 생성합니다.

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
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"

SUMMARY_NAME = "stage2_n8a0_i60_ohlc_ma_vb_r20_five_seed"
BUNDLE_PATH = Path("/kaggle/working/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8.zip")

# Strict Stage 2 comparison settings.
BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False

# Set True only for a quick code-path check. For real checkpoint rebuilding, keep False.
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
        "scripts/summarize_stage2_grid_results.py",
        "src/stage2_btc/models/stock_cnn.py",
        "src/stage2_btc/training/loop.py",
    ]
    missing = [path for path in required if not (project_root / path).exists()]
    if missing:
        raise RuntimeError(
            "Stage 2 code snapshot is stale/incomplete. "
            "Upload the latest stage2_btc_extension dataset. Missing: "
            + ", ".join(missing)
        )


def experiment_name() -> str:
    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def patch_kaggle_config():
    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["evaluation"]["batch_size"] = BATCH_SIZE
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Config patched:", config_path, flush=True)


def output_paths(run_seed: int) -> dict[str, Path]:
    exp = experiment_name()
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


def is_seed_complete(run_seed: int) -> bool:
    required = output_paths(run_seed)
    return all(path.exists() for path in required.values())


def read_metric_row(run_seed: int) -> dict:
    paths = output_paths(run_seed)
    row = {
        "experiment_name": experiment_name(),
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "run_seed": run_seed,
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
    row = {
        "experiment_name": experiment_name(),
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "seed_count": int(seed_df["run_seed"].nunique()),
    }
    for column in numeric:
        if column in seed_df.columns:
            values = pd.to_numeric(seed_df[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)
            row[f"{column}_count"] = int(values.notna().sum())
    return pd.DataFrame([row])


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
    for run_seed in RUN_SEEDS:
        paths = output_paths(run_seed)
        files.extend(paths[label] for label in bundle_labels)
    files = [path for path in files if path.exists()]

    manifest = {
        "status": "ok",
        "purpose": "Stage 2 selected checkpoints for Stage 4 N8 pretrained/frozen FiLM",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_name": experiment_name(),
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
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
    "Selected checkpoint rebuild:",
    {
        "experiment_name": experiment_name(),
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

for run_seed in RUN_SEEDS:
    print("\n" + "=" * 88, flush=True)
    print(f"Stage 2 selected checkpoint rebuild: {experiment_name()}, seed={run_seed}", flush=True)
    print("=" * 88, flush=True)

    if is_seed_complete(run_seed):
        print(f"[skip] outputs already complete for seed={run_seed}", flush=True)
        continue

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
    ] + smoke_eval_args)

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


# ============================================================
# 4. Verify outputs and bundle checkpoint artifacts
# ============================================================
missing = []
for run_seed in RUN_SEEDS:
    for label, path in output_paths(run_seed).items():
        if not path.exists():
            missing.append({"run_seed": run_seed, "label": label, "path": str(path)})
if missing:
    raise RuntimeError("Missing required Stage 2 N8-A0 artifact(s): " + json.dumps(missing, indent=2))

seed_df = pd.DataFrame([read_metric_row(run_seed) for run_seed in RUN_SEEDS])
mean_std_df = summarize_seed_rows(seed_df)
bundle_path = make_checkpoint_bundle(seed_df, mean_std_df)

display(Markdown("# Stage 2 N8-A0 Seed-Level Results"))
display(seed_df.sort_values("run_seed"))

display(Markdown("# Stage 2 N8-A0 Mean/Std Summary"))
display(mean_std_df)

print("\nDONE", flush=True)
print("Experiment:", experiment_name(), flush=True)
print("Bundle:", bundle_path, flush=True)
```
