# Kaggle Stage 4 N15-C F&G-Only Across Image Specs One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage22/stage2_btc_extension_latest_for_n13_6/stage2_btc_extension`
- N15-A Stage 2 checkpoint/prediction bundle:
  `/kaggle/input/datasets/moskow/stage22/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- F&G Kaggle dataset:
  `ashishpatel8736/historical-and-fear-greed-index-datasets`

This runner executes `4-N15-C`:

```text
for each image spec in ohlc / ohlc_ma / ohlc_vb / ohlc_ma_vb:
    load the matching Stage 2 I60/R20 checkpoint
    freeze visual CNN and classifier
    train only context encoder + bounded final-block FiLM heads
    use the same F&G-only external market-regime vector
```

Feature set:

```text
all image specs -> fg_only: fg_value, fg_mean_60, fg_delta_60, fg_std_60
```

Main question:

```text
Does external market-regime information help regardless of how much chart
information is already drawn into the image?
```

```python
from pathlib import Path
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
CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n15c_snapshot.zip"),
]
STAGE2_CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage22/stage2_btc_extension_latest_for_n13_6/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage22/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage4/stage2_btc_extension"),
]
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect under /kaggle/input.
SOURCE_FILE = ""
FEAR_GREED_FILE = ""

# N15-A Stage 2 checkpoint/prediction bundle. Keep this explicit because Kaggle
# resets /kaggle/working and the dataset is nested under the stage22 dataset.
# This folder contains:
# outputs/stage2/checkpoints/stage2_i60_ohlc_r20/seed_42/best.pt
# outputs/stage2/checkpoints/stage2_i60_ohlc_ma_r20/seed_42/best.pt
# outputs/stage2/checkpoints/stage2_i60_ohlc_vb_r20/seed_42/best.pt
# outputs/stage2/checkpoints/stage2_i60_ohlc_ma_vb_r20/seed_42/best.pt
STAGE2_CHECKPOINT_BUNDLE = (
    "/kaggle/input/datasets/moskow/stage22/"
    "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/"
    "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"
)

IMAGE_WINDOW = 60
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
MODULATION_SCALE = 0.02
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
MIN_PREDICTIONS = 1000

SPEC_FEATURE_ROWS = [
    {
        "image_spec": "ohlc",
        "feature_set": "fg_only",
        "features": ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"],
        "rationale": "External F&G regime vector on the weakest OHLC-only visual baseline.",
    },
    {
        "image_spec": "ohlc_ma",
        "feature_set": "fg_only",
        "features": ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"],
        "rationale": "External F&G regime vector on OHLC+MA visual baseline.",
    },
    {
        "image_spec": "ohlc_vb",
        "feature_set": "fg_only",
        "features": ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"],
        "rationale": "External F&G regime vector on OHLC+volume-bar visual baseline.",
    },
    {
        "image_spec": "ohlc_ma_vb",
        "feature_set": "fg_only",
        "features": ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"],
        "rationale": "External F&G regime vector on the strongest full visual baseline.",
    },
]
SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
RUN_GRADCAM = False
GRADCAM_SAMPLES_PER_CLASS = 2
RESUME_EXISTING_PROJECT = False

# Smoke check only. For the real N15-C run, keep False.
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

OUTPUT_PREFIX = "stage4_n15c_fg_only_across_image_specs"
BUNDLE_PATH = Path("/kaggle/working/stage4_n15c_fg_only_across_image_specs_result_bundle.zip")


def first_existing(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in candidates))


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
    """Run one command."""

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
    text = f"{float(scale):.3f}".rstrip("0").rstrip(".")
    return "s" + text.replace(".", "p")


def context_feature_set_name(row: dict) -> str:
    return f"n15c_{row['feature_set']}"


def context_name(row: dict) -> str:
    return (
        f"stage4_context_i{IMAGE_WINDOW}_{row['image_spec']}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{context_feature_set_name(row)}"
    )


def experiment_suffix(row: dict) -> str:
    return f"n15c_{row['feature_set']}_pretrained_frozen_{scale_label(MODULATION_SCALE)}"


def experiment_name(row: dict) -> str:
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{row['image_spec']}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{experiment_suffix(row)}"
    )


def stage2_experiment_name(image_spec: str) -> str:
    return f"stage2_i{IMAGE_WINDOW}_{image_spec}_r{RETURN_HORIZON}"


def required_stage2_checkpoint(image_spec: str, run_seed: int = 42) -> Path:
    return (
        Path("outputs/stage2/checkpoints")
        / stage2_experiment_name(image_spec)
        / f"seed_{int(run_seed)}"
        / "best.pt"
    )


def find_stage2_checkpoint_bundle() -> Path:
    """Find a folder usable as Stage 2 output root parent."""

    if str(STAGE2_CHECKPOINT_BUNDLE).strip():
        root = Path(str(STAGE2_CHECKPOINT_BUNDLE)).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Configured checkpoint bundle missing: {root}")
        return resolve_checkpoint_candidate(root)

    if has_all_stage2_checkpoints(STAGE2_PROJECT_ROOT):
        return STAGE2_PROJECT_ROOT

    candidates = [
        Path("/kaggle/working/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
        Path("/kaggle/working/stage2_i60_r20_four_image_specs_seed42_46_checkpoints_for_stage4_n15.zip"),
        Path("/kaggle/input/datasets/moskow/stage4/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
        Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
    ]
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            candidates.extend(sorted(root.glob("*stage2*i60*r20*n15*")))
            candidates.extend(sorted(root.rglob("stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15")))
            candidates.extend(sorted(root.rglob("stage2_i60_r20_four_image_specs_seed42_46_checkpoints_for_stage4_n15.zip")))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            resolved = resolve_checkpoint_candidate(candidate)
        except (FileNotFoundError, ValueError, zipfile.BadZipFile):
            continue
        print("Using Stage 2 checkpoint bundle:", resolved, flush=True)
        return resolved

    raise FileNotFoundError(
        "Could not find the N15-A Stage 2 checkpoint bundle. Attach the bundle "
        "dataset or set STAGE2_CHECKPOINT_BUNDLE manually."
    )


def resolve_checkpoint_candidate(candidate: Path) -> Path:
    """Resolve a folder or zip containing all four Stage 2 image-spec checkpoints."""

    candidate = candidate.expanduser()
    if candidate.is_file():
        if candidate.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported checkpoint candidate file: {candidate}")
        extract_root = Path("/kaggle/working/stage2_n15_checkpoint_bundle_extracted") / candidate.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(extract_root)
        return resolve_checkpoint_candidate(extract_root)

    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint candidate missing: {candidate}")
    if has_all_stage2_checkpoints(candidate):
        return candidate

    nested_best = next(candidate.rglob(str(required_stage2_checkpoint("ohlc"))), None)
    if nested_best is not None:
        # expected layout: <root>/outputs/stage2/checkpoints/<exp>/seed_42/best.pt
        root = nested_best.parents[5]
        if has_all_stage2_checkpoints(root):
            return root

    raise FileNotFoundError(f"No complete N15-A checkpoint bundle under: {candidate}")


def has_all_stage2_checkpoints(root: Path) -> bool:
    for row in SPEC_FEATURE_ROWS:
        for seed in RUN_SEEDS:
            if not (root / required_stage2_checkpoint(row["image_spec"], seed)).exists():
                return False
    return True


def assert_required_code():
    """Fail early if uploaded code snapshots are stale or incomplete."""

    required_stage4 = [
        "scripts/build_stage4_context_features.py",
        "scripts/run_stage4_context_model.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/evaluate_stage4_trading.py",
        "scripts/check_stage4_outputs.py",
        "scripts/analyze_stage4_stage2_context_corrections.py",
        "src/stage4_film/training/loop.py",
        "src/stage4_film/runners/context_experiment.py",
        "src/stage4_film/models/film_stock_cnn.py",
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
        ("src/stage4_film/runners/context_experiment.py", "configure_stage2_pretrained_context_model"),
        ("src/stage4_film/runners/context_experiment.py", "freeze_visual_backbone"),
        ("src/stage4_film/training/loop.py", "initialize_weights"),
        ("src/stage4_film/models/film_stock_cnn.py", "BoundedLastBlockFilmContextStockCNN"),
    ]
    stale = []
    for rel_path, marker in marker_checks:
        path = PROJECT_ROOT / rel_path
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            stale.append(f"{rel_path} missing marker {marker!r}")
    if stale:
        raise RuntimeError(
            "Stage 4 code snapshot is stale/incomplete. Upload the latest "
            "stage4_film_conditioning folder/zip, then rerun. Problems: "
            + "; ".join(stale)
        )


def patch_config(row: dict, checkpoint_bundle: Path):
    """Patch Stage 4 Kaggle config for one N15-C row."""

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
    cfg["stage2_dependency"]["baseline_output_root"] = str(checkpoint_bundle / "outputs/stage2")

    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["context"]["source"] = "structured"
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["primary_features"] = list(row["features"])
    cfg["context"]["feature_set_name"] = context_feature_set_name(row)
    cfg["context"]["fear_greed"]["kaggle_file"] = FEAR_GREED_FILE

    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = row["image_spec"]
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(row["features"])
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = experiment_suffix(row)
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = float(MODULATION_SCALE)
    cfg["stage4_model"]["pretrained_stage2"] = {
        "enabled": True,
        "checkpoint_output_root": str(checkpoint_bundle),
        "freeze_visual_backbone": True,
        "freeze_classifier": True,
        "initialize_new_context_modules": True,
        "strict_load": True,
    }

    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    cfg["evaluation"]["batch_size"] = BATCH_SIZE

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def output_check_cmd(row: dict, run_seed: int):
    cmd = [
        sys.executable, "-u",
        "scripts/check_stage4_outputs.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", row["image_spec"],
        "--return-horizon", str(RETURN_HORIZON),
        "--context-method", CONTEXT_METHOD,
        "--run-seed", str(run_seed),
        "--split", EVAL_SPLIT,
        "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
        "--min-predictions", str(MIN_PREDICTIONS),
    ]
    if not RUN_GRADCAM:
        cmd.append("--skip-gradcam")
    return cmd


def is_completed(row: dict, run_seed: int, checkpoint_bundle: Path) -> bool:
    patch_config(row, checkpoint_bundle)
    result = run(output_check_cmd(row, run_seed), capture=True, check=False)
    return result.returncode == 0


def read_result_row(row: dict, run_seed: int, status: str, error: str = "") -> dict:
    exp = experiment_name(row)
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_trading_metrics.json"
    manifest_path = PROJECT_ROOT / "outputs/stage4/run_manifests" / exp / f"seed_{run_seed}" / "run_manifest.json"
    result = {
        "experiment_name": exp,
        "image_window": IMAGE_WINDOW,
        "return_horizon": RETURN_HORIZON,
        "image_spec": row["image_spec"],
        "context_method": CONTEXT_METHOD,
        "context_feature_set": row["feature_set"],
        "context_features": "|".join(row["features"]),
        "modulation_scale": float(MODULATION_SCALE),
        "run_seed": int(run_seed),
        "status": status,
        "error": error,
        "classification_available": metrics_path.exists(),
        "trading_available": trading_path.exists(),
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in [
            "num_samples",
            "accuracy",
            "majority_class_accuracy",
            "roc_auc",
            "average_precision",
            "f1",
            "brier_score",
            "predicted_positive_rate",
        ]:
            result[key] = metrics.get(key)
        if result.get("accuracy") is not None and result.get("majority_class_accuracy") is not None:
            result["accuracy_minus_majority_class_accuracy"] = (
                float(result["accuracy"]) - float(result["majority_class_accuracy"])
            )
    if trading_path.exists():
        trading = json.loads(trading_path.read_text(encoding="utf-8"))
        for strategy_name in ["long_flat", "long_short"]:
            values = trading.get(strategy_name, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                result[f"{strategy_name}_{key}"] = values.get(key)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pretrained = manifest.get("run_context", {}).get("pretrained_stage2", {})
        result["pretrained_loaded"] = pretrained.get("enabled")
        result["freeze_visual_backbone"] = pretrained.get("freeze_visual_backbone")
        result["freeze_classifier"] = pretrained.get("freeze_classifier")
        result["num_trainable_parameters"] = pretrained.get("num_trainable_parameters")
        result["num_frozen_parameters"] = pretrained.get("num_frozen_parameters")
    return result


def summarize_seed_results(seed_rows: list[dict], checkpoint_bundle: Path) -> pd.DataFrame:
    df = pd.DataFrame(seed_rows)
    if df.empty:
        return pd.DataFrame()

    numeric = [
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "accuracy_minus_majority_class_accuracy",
        "roc_auc",
        "average_precision",
        "f1",
        "brier_score",
        "predicted_positive_rate",
        "long_flat_sharpe_net",
        "long_short_sharpe_net",
        "long_flat_annualized_return_net",
        "long_short_annualized_return_net",
        "num_trainable_parameters",
        "num_frozen_parameters",
    ]
    rows = []
    group_cols = ["image_spec", "context_feature_set", "modulation_scale"]
    for keys, frame in df.groupby(group_cols, dropna=False):
        image_spec, feature_set, scale = keys
        row = {
            "image_spec": image_spec,
            "context_feature_set": feature_set,
            "modulation_scale": scale,
            "seed_count": int(frame["run_seed"].nunique()),
        }
        for column in numeric:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                row[f"{column}_mean"] = values.mean()
                row[f"{column}_std"] = values.std(ddof=1)
                row[f"{column}_count"] = int(values.notna().sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    return add_stage2_reference_columns(summary, checkpoint_bundle)


def add_stage2_reference_columns(summary: pd.DataFrame, checkpoint_bundle: Path) -> pd.DataFrame:
    reference = read_stage2_reference(checkpoint_bundle)
    if summary.empty or reference.empty:
        return summary
    merged = summary.merge(reference, on="image_spec", how="left")
    for metric in ["accuracy", "roc_auc", "f1", "brier_score"]:
        stage4_col = f"{metric}_mean"
        stage2_col = f"stage2_{metric}_mean"
        if stage4_col in merged.columns and stage2_col in merged.columns:
            merged[f"{metric}_minus_same_image_stage2"] = merged[stage4_col] - merged[stage2_col]
    full_row = reference[reference["image_spec"] == "ohlc_ma_vb"]
    if not full_row.empty and "accuracy_mean" in summary.columns:
        full_accuracy = float(full_row.iloc[0]["stage2_accuracy_mean"])
        merged["accuracy_gap_to_stage2_ohlc_ma_vb"] = merged["accuracy_mean"] - full_accuracy
    return merged


def read_stage2_reference(checkpoint_bundle: Path) -> pd.DataFrame:
    candidates = [
        checkpoint_bundle
        / "reports/tables/stage2_n15a_i60_r20_four_image_specs_five_seed_mean_std_results.csv",
        checkpoint_bundle
        / "reports/tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[(frame["image_window"] == IMAGE_WINDOW) & (frame["return_horizon"] == RETURN_HORIZON)].copy()
            keep = ["image_spec"]
            renames = {}
            for metric in ["accuracy", "roc_auc", "f1", "brier_score"]:
                column = f"{metric}_mean"
                if column in frame.columns:
                    keep.append(column)
                    renames[column] = f"stage2_{metric}_mean"
            return frame[keep].rename(columns=renames)
    return pd.DataFrame()


def stage2_output_root_for_analysis(checkpoint_bundle: Path, image_spec: str) -> Path:
    expected_prediction = (
        Path("predictions")
        / stage2_experiment_name(image_spec)
        / "seed_42"
        / f"{EVAL_SPLIT}_predictions.csv"
    )
    candidates = [
        checkpoint_bundle / "outputs/stage2",
        checkpoint_bundle / "stage2",
        checkpoint_bundle,
        STAGE2_PROJECT_ROOT / "outputs/stage2",
    ]
    for candidate in candidates:
        if (candidate / expected_prediction).exists():
            return candidate
    raise FileNotFoundError(
        "Stage 2 baseline predictions are required for correction analysis. "
        f"Missing {expected_prediction} under {checkpoint_bundle}"
    )


def run_correction_analysis(row: dict, checkpoint_bundle: Path):
    stage2_output_root = stage2_output_root_for_analysis(checkpoint_bundle, row["image_spec"])
    prefix = f"{OUTPUT_PREFIX}_{row['image_spec']}_{row['feature_set']}_stage2_vs_film_correction"
    run([
        sys.executable, "-u",
        "scripts/analyze_stage4_stage2_context_corrections.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", row["image_spec"],
        "--return-horizon", str(RETURN_HORIZON),
        "--context-window", str(CONTEXT_WINDOW),
        "--context-method", CONTEXT_METHOD,
        "--stage4-experiment-suffix", experiment_suffix(row),
        "--context-name", context_name(row),
        "--stage2-output-root", str(stage2_output_root),
        "--stage4-output-root", str(PROJECT_ROOT / "outputs/stage4"),
        "--run-seeds", *map(str, RUN_SEEDS),
        "--split", EVAL_SPLIT,
        "--top-k", "20",
        "--analysis-name", f"N15-C {row['image_spec']} {row['feature_set']}",
        "--output-prefix", prefix,
    ])
    return {
        "seed_summary": PROJECT_ROOT / "reports/tables" / f"{prefix}_seed_summary.csv",
        "transition_summary": PROJECT_ROOT / "reports/tables" / f"{prefix}_transition_summary.csv",
        "report": PROJECT_ROOT / "reports/tables" / f"{prefix}_report.md",
    }


def read_correction_outputs(paths_by_row: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_frames = []
    transition_frames = []
    for item in paths_by_row:
        row = item["row"]
        paths = item["paths"]
        if paths["seed_summary"].exists():
            frame = pd.read_csv(paths["seed_summary"])
            frame.insert(0, "image_spec", row["image_spec"])
            frame.insert(1, "context_feature_set", row["feature_set"])
            seed_frames.append(frame)
        if paths["transition_summary"].exists():
            frame = pd.read_csv(paths["transition_summary"])
            frame.insert(0, "image_spec", row["image_spec"])
            frame.insert(1, "context_feature_set", row["feature_set"])
            transition_frames.append(frame)
    seed_df = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    transition_df = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    return seed_df, transition_df


def make_result_bundle():
    """Zip N15-C outputs for local download and future Kaggle resume."""

    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "reports/figures",
        PROJECT_ROOT / "outputs/stage4/checkpoints",
        PROJECT_ROOT / "outputs/stage4/metrics",
        PROJECT_ROOT / "outputs/stage4/predictions",
        PROJECT_ROOT / "outputs/stage4/run_manifests",
        PROJECT_ROOT / "outputs/stage4/context",
        PROJECT_ROOT / "outputs/stage4/figures",
    ]
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in include_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(PROJECT_ROOT))
    print(f"Result bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / (1024 * 1024):.2f} MB)", flush=True)


# ============================================================
# 1. Copy code snapshots and find checkpoint bundle
# ============================================================
if RESUME_EXISTING_PROJECT:
    if not PROJECT_ROOT.exists() or not STAGE2_PROJECT_ROOT.exists():
        raise FileNotFoundError("RESUME_EXISTING_PROJECT=True requires existing working folders.")
    print(f"Resuming existing Stage 4 project: {PROJECT_ROOT}", flush=True)
else:
    copy_or_extract_input(first_existing(CODE_INPUT_CANDIDATES), PROJECT_ROOT, expected_child="stage4_film_conditioning")
    copy_or_extract_input(first_existing(STAGE2_CODE_INPUT_CANDIDATES), STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")

assert_required_code()
checkpoint_bundle = find_stage2_checkpoint_bundle()
print("Stage 4 project ready:", PROJECT_ROOT, flush=True)
print("Stage 2 dependency ready:", STAGE2_PROJECT_ROOT, flush=True)
print("Stage 2 checkpoint bundle:", checkpoint_bundle, flush=True)


# ============================================================
# 2. Train/evaluate/check N15-C rows
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
total_runs = len(SPEC_FEATURE_ROWS) * len(RUN_SEEDS)
run_index = 0

for row in SPEC_FEATURE_ROWS:
    print("\n" + "#" * 100, flush=True)
    print(f"N15-C row: image_spec={row['image_spec']} feature_set={row['feature_set']}", flush=True)
    print("features:", row["features"], flush=True)
    print("rationale:", row["rationale"], flush=True)
    print("#" * 100, flush=True)

    patch_config(row, checkpoint_bundle)

    for run_seed in RUN_SEEDS:
        print("\n" + "-" * 90, flush=True)
        print(f"Build context: {context_name(row)}, seed={run_seed}", flush=True)
        print("-" * 90, flush=True)
        run([
            sys.executable, "-u",
            "scripts/build_stage4_context_features.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", row["image_spec"],
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(run_seed),
            "--write-report-copy",
        ])

        run_index += 1
        exp = experiment_name(row)
        print("\n" + "=" * 100, flush=True)
        print(f"[{run_index}/{total_runs}] {exp}, seed={run_seed}", flush=True)
        print("=" * 100, flush=True)

        if SKIP_COMPLETED and is_completed(row, run_seed, checkpoint_bundle):
            print(f"[skip] Output check already passes for {exp}, seed={run_seed}", flush=True)
            summary_rows.append(read_result_row(row, run_seed, status="skipped_completed"))
            continue

        try:
            run([
                sys.executable, "-u",
                "scripts/run_stage4_context_model.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", row["image_spec"],
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", CONTEXT_METHOD,
                "--run-seed", str(run_seed),
                "--experiment-suffix", experiment_suffix(row),
                "--context-feature-set-name", context_feature_set_name(row),
                "--context-features", *row["features"],
                "--modulation-scale", str(MODULATION_SCALE),
                "--enable-stage2-pretrained",
                "--stage2-pretrained-bundle-root", str(checkpoint_bundle),
            ] + smoke_train_args)

            run([
                sys.executable, "-u",
                "scripts/evaluate_stage4_predictions.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", row["image_spec"],
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", CONTEXT_METHOD,
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
            ] + smoke_data_args)

            run([
                sys.executable, "-u",
                "scripts/evaluate_stage4_trading.py",
                "--config", "configs/env_kaggle.yaml",
                "--image-window", str(IMAGE_WINDOW),
                "--image-spec", row["image_spec"],
                "--return-horizon", str(RETURN_HORIZON),
                "--context-method", CONTEXT_METHOD,
                "--run-seed", str(run_seed),
                "--split", EVAL_SPLIT,
            ])

            if RUN_GRADCAM:
                run([
                    sys.executable, "-u",
                    "scripts/generate_stage4_gradcam_context.py",
                    "--config", "configs/env_kaggle.yaml",
                    "--image-window", str(IMAGE_WINDOW),
                    "--image-spec", row["image_spec"],
                    "--return-horizon", str(RETURN_HORIZON),
                    "--context-method", CONTEXT_METHOD,
                    "--run-seed", str(run_seed),
                    "--split", EVAL_SPLIT,
                    "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
                    "--write-report-copy",
                ] + smoke_data_args)

            run(output_check_cmd(row, run_seed))
            summary_rows.append(read_result_row(row, run_seed, status="ok"))
        except Exception as exc:
            print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
            summary_rows.append(read_result_row(row, run_seed, status="failed", error=str(exc)))
            if not CONTINUE_ON_ERROR:
                raise


# ============================================================
# 3. Stage2-vs-FiLM correction analysis
# ============================================================
correction_paths = []
for row in SPEC_FEATURE_ROWS:
    patch_config(row, checkpoint_bundle)
    try:
        paths = run_correction_analysis(row, checkpoint_bundle)
        correction_paths.append({"row": row, "paths": paths})
    except Exception as exc:
        print(f"[warning] correction analysis failed for {row['image_spec']} {row['feature_set']}: {exc}", flush=True)
        if not CONTINUE_ON_ERROR:
            raise

correction_seed_df, correction_transition_df = read_correction_outputs(correction_paths)


# ============================================================
# 4. Write summary tables and bundle
# ============================================================
tables_root = PROJECT_ROOT / "reports/tables"
tables_root.mkdir(parents=True, exist_ok=True)
seed_df = pd.DataFrame(summary_rows)
mean_std_df = summarize_seed_results(summary_rows, checkpoint_bundle)

seed_csv = tables_root / f"{OUTPUT_PREFIX}_seed_results.csv"
mean_std_csv = tables_root / f"{OUTPUT_PREFIX}_mean_std_results.csv"
correction_seed_csv = tables_root / f"{OUTPUT_PREFIX}_correction_seed_summary.csv"
correction_transition_csv = tables_root / f"{OUTPUT_PREFIX}_correction_transition_summary.csv"
run_summary_json = tables_root / f"{OUTPUT_PREFIX}_run_summary.json"

seed_df.to_csv(seed_csv, index=False)
mean_std_df.to_csv(mean_std_csv, index=False)
correction_seed_df.to_csv(correction_seed_csv, index=False)
correction_transition_df.to_csv(correction_transition_csv, index=False)
run_summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

make_result_bundle()

display(Markdown("# Stage 4 N15-C Seed-Level Results"))
if not seed_df.empty:
    display(seed_df.sort_values(["image_spec", "context_feature_set", "run_seed"]))
else:
    display(seed_df)

display(Markdown("# Stage 4 N15-C Mean/Std Summary"))
if not mean_std_df.empty and "accuracy_mean" in mean_std_df.columns:
    display(mean_std_df.sort_values("accuracy_mean", ascending=False))
else:
    display(mean_std_df)

display(Markdown("# Stage 4 N15-C Correction Seed Summary"))
display(correction_seed_df)

display(Markdown("# Stage 4 N15-C Transition Summary"))
display(correction_transition_df)

print("\nDONE", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Correction seed CSV:", correction_seed_csv, flush=True)
print("Correction transition CSV:", correction_transition_csv, flush=True)
print("Bundle:", BUNDLE_PATH, flush=True)
```
