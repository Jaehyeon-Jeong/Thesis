# Kaggle Stage 4 N13-2 FSI-Only Pretrained Frozen Bounded FiLM One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- Stage 2 selected checkpoint bundle from N8-A0:
  `stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`

This runner executes `4-N13-2` with a compact FSI feature-set grid:

```text
OFR FSI source CSV
-> source-level 20/60-observation FSI rolling features
-> prebuilt FSI context per seed
-> filtered feature-set contexts: fsi_2 / fsi_3 / fsi_all
-> Stage 2 I60/R20/ohlc_ma_vb checkpoint loaded/frozen
-> train only FSI context encoder + bounded final-block FiLM heads
-> five seeds: 42, 43, 44, 45, 46
```

The default scale is intentionally conservative: `0.02`.

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
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n13_2_with_stage2_bundle.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n13_1_latest.zip"),
]
STAGE2_CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage4/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension"),
]
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect under /kaggle/input.
SOURCE_FILE = ""

# Leave empty to use the FSI CSV bundled in the Stage 4 snapshot.
FSI_CSV = ""

# Leave empty to auto-detect Stage 2 checkpoints.
# If auto-detect fails, set this to a folder/zip that contains:
# outputs/stage2/checkpoints/stage2_i60_ohlc_ma_vb_r20/seed_42/best.pt
STAGE2_CHECKPOINT_BUNDLE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
SOURCE_CONTEXT_FEATURE_SET = "ofr_fsi_lag1_w20_60"
SOURCE_CONTEXT_NAME = (
    f"stage4_fsi_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
    f"r{RETURN_HORIZON}_{SOURCE_CONTEXT_FEATURE_SET}"
)
ALL_FSI_FEATURES = [
    "ofr_fsi_value",
    "ofr_fsi_mean_20",
    "ofr_fsi_mean_60",
    "ofr_fsi_delta_20",
    "ofr_fsi_delta_60",
    "ofr_fsi_std_60",
]
FSI_FEATURE_SETS = {
    # Screening result: the 60-observation level and change signals are the
    # least redundant and most stable train/validation candidates.
    "fsi_2": ["ofr_fsi_mean_60", "ofr_fsi_delta_60"],
    "fsi_3": ["ofr_fsi_mean_60", "ofr_fsi_delta_60", "ofr_fsi_std_60"],
    "fsi_all": list(ALL_FSI_FEATURES),
}
CONTEXT_FEATURE_SET_KEYS = ["fsi_2", "fsi_3", "fsi_all"]

MODULATION_SCALES = [0.02]
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True

# N13-6 handles targeted interpretation. Keep False for the first metric run.
RUN_GRADCAM = False
GRADCAM_SAMPLES_PER_CLASS = 2

# Smoke check only. For the real N13-2 run, keep False.
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

OUTPUT_PREFIX = "stage4_n13_2_fsi_only_pretrained_frozen_bounded_film"
BUNDLE_PATH = Path("/kaggle/working/stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_result_bundle.zip")


def first_existing(candidates):
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in candidates))


def copy_or_extract_input(src: Path, dst: Path, expected_child: str | None = None):
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


def selected_context_features(feature_set_key: str) -> list[str]:
    if feature_set_key not in FSI_FEATURE_SETS:
        raise KeyError(f"Unknown FSI feature set: {feature_set_key}")
    return list(FSI_FEATURE_SETS[feature_set_key])


def context_feature_set_name(feature_set_key: str) -> str:
    return f"{SOURCE_CONTEXT_FEATURE_SET}_{feature_set_key}"


def prebuilt_context_name(feature_set_key: str) -> str:
    return (
        f"stage4_fsi_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_{context_feature_set_name(feature_set_key)}"
    )


def experiment_suffix(scale: float, feature_set_key: str) -> str:
    return f"{feature_set_key}_ofr_fsi_pretrained_frozen_{scale_label(scale)}"


def experiment_name(scale: float, feature_set_key: str) -> str:
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{experiment_suffix(scale, feature_set_key)}"
    )


def source_context_seed_dir(run_seed: int) -> Path:
    return PROJECT_ROOT / "outputs/stage4/context" / SOURCE_CONTEXT_NAME / f"seed_{run_seed}"


def context_seed_dir(feature_set_key: str, run_seed: int) -> Path:
    return PROJECT_ROOT / "outputs/stage4/context" / prebuilt_context_name(feature_set_key) / f"seed_{run_seed}"


def find_stage2_checkpoint_bundle() -> Path:
    if str(STAGE2_CHECKPOINT_BUNDLE).strip():
        root = Path(str(STAGE2_CHECKPOINT_BUNDLE)).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Configured checkpoint bundle missing: {root}")
        return resolve_checkpoint_candidate(root)

    expected_exp = f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"
    working_checkpoint = (
        STAGE2_PROJECT_ROOT
        / "outputs/stage2/checkpoints"
        / expected_exp
        / "seed_42/best.pt"
    )
    if working_checkpoint.exists():
        return STAGE2_PROJECT_ROOT

    embedded_checkpoint_bundle = (
        PROJECT_ROOT
        / "stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"
    )
    if embedded_checkpoint_bundle.exists():
        return resolve_checkpoint_candidate(embedded_checkpoint_bundle)

    for search_root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if not search_root.exists():
            continue
        for candidate in search_root.rglob("*"):
            if candidate == PROJECT_ROOT or PROJECT_ROOT in candidate.parents:
                continue
            if candidate.is_file() and (
                candidate.suffix.lower() != ".zip"
                or ("stage2" not in candidate.name.lower() and "checkpoint" not in candidate.name.lower())
            ):
                continue
            try:
                resolved = resolve_checkpoint_candidate(candidate)
            except (FileNotFoundError, ValueError, zipfile.BadZipFile):
                continue
            print("Auto-detected Stage 2 checkpoint bundle:", resolved, flush=True)
            return resolved
    raise FileNotFoundError(
        "Could not auto-detect Stage 2 checkpoint bundle. Attach the N8-A0 "
        "bundle dataset or set STAGE2_CHECKPOINT_BUNDLE manually."
    )


def resolve_checkpoint_candidate(candidate: Path) -> Path:
    expected_exp = f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"
    candidate = candidate.expanduser()
    if candidate.is_file():
        if candidate.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported checkpoint candidate file: {candidate}")
        extract_root = Path("/kaggle/working/stage2_checkpoint_bundle_extracted") / candidate.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(extract_root)
        return resolve_checkpoint_candidate(extract_root)

    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint candidate missing: {candidate}")

    direct = candidate / "outputs/stage2/checkpoints" / expected_exp / "seed_42/best.pt"
    if direct.exists():
        return candidate

    output_root = candidate / "stage2/checkpoints" / expected_exp / "seed_42/best.pt"
    if output_root.exists():
        return candidate / "stage2"

    checkpoint_root = candidate / "checkpoints" / expected_exp / "seed_42/best.pt"
    if checkpoint_root.exists():
        return candidate

    nested = next(candidate.rglob(f"outputs/stage2/checkpoints/{expected_exp}/seed_42/best.pt"), None)
    if nested is not None:
        return nested.parents[5]

    raise FileNotFoundError(f"No Stage 2 checkpoint found under: {candidate}")


def assert_required_code():
    required_stage4 = [
        "scripts/build_stage4_fsi_context_features.py",
        "scripts/run_stage4_context_model.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/evaluate_stage4_trading.py",
        "scripts/check_stage4_outputs.py",
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
        ("scripts/build_stage4_fsi_context_features.py", "source-level feature engineering"),
        ("scripts/build_stage4_fsi_context_features.py", "add_source_level_fsi_features"),
        ("src/stage4_film/runners/context_experiment.py", "prebuilt_context_name"),
        ("src/stage4_film/runners/context_experiment.py", "configure_stage2_pretrained_context_model"),
        ("src/stage4_film/models/film_stock_cnn.py", "BoundedLastBlockFilmContextStockCNN"),
    ]
    stale = []
    for rel_path, marker in marker_checks:
        path = PROJECT_ROOT / rel_path
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            stale.append(f"{rel_path} missing marker {marker!r}")
    if stale:
        raise RuntimeError(
            "Stage 4 code snapshot is stale/incomplete. Upload latest Stage 4. Problems: "
            + "; ".join(stale)
        )


def resolve_fsi_csv() -> Path:
    if str(FSI_CSV).strip():
        path = Path(str(FSI_CSV)).expanduser()
    else:
        path = PROJECT_ROOT / "data_inventory/ofr_fsi/fsi.csv"
    if not path.exists():
        raise FileNotFoundError(f"OFR FSI CSV missing: {path}")
    return path


def patch_base_config(checkpoint_bundle: Path):
    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cfg["paths"]["project_root"] = str(PROJECT_ROOT)
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
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
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["primary_features"] = list(ALL_FSI_FEATURES)
    cfg["context"]["normalized_feature_columns"] = [f"{feature}_normalized" for feature in ALL_FSI_FEATURES]
    cfg["context"]["feature_set_name"] = SOURCE_CONTEXT_FEATURE_SET

    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(ALL_FSI_FEATURES)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]

    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    cfg["evaluation"]["batch_size"] = BATCH_SIZE

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def patch_prebuilt_fsi_config(scale: float, feature_set_key: str, checkpoint_bundle: Path):
    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    features = selected_context_features(feature_set_key)

    cfg["stage2_dependency"]["baseline_output_root"] = str(checkpoint_bundle / "outputs/stage2")
    cfg["context"]["source"] = "prebuilt"
    cfg["context"]["feature_set_name"] = context_feature_set_name(feature_set_key)
    cfg["context"]["prebuilt_context_name"] = prebuilt_context_name(feature_set_key)
    cfg["context"]["primary_features"] = features
    cfg["context"]["normalized_feature_columns"] = [f"{feature}_normalized" for feature in features]

    cfg["stage4_model"]["context_dim"] = len(features)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = experiment_suffix(scale, feature_set_key)
    cfg["stage4_model"].setdefault("film_full_bounded_last_block", {})
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = float(scale)
    cfg["stage4_model"]["pretrained_stage2"] = {
        "enabled": True,
        "checkpoint_output_root": str(checkpoint_bundle),
        "freeze_visual_backbone": True,
        "freeze_classifier": True,
        "initialize_new_context_modules": True,
        "strict_load": True,
    }

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def build_fsi_context_for_seed(run_seed: int, fsi_csv: Path):
    scaler = source_context_seed_dir(run_seed) / "context_scaler.json"
    features = source_context_seed_dir(run_seed) / "context_features.csv"
    if scaler.exists() and features.exists():
        print(f"Source FSI context exists for seed {run_seed}: {source_context_seed_dir(run_seed)}", flush=True)
        return
    run([
        sys.executable, "-u",
        "scripts/build_stage4_fsi_context_features.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(run_seed),
        "--fsi-csv", str(fsi_csv),
        "--asof-lag-days", "1",
        "--output-prefix", "stage4_fsi_context",
        "--feature-set-name", SOURCE_CONTEXT_FEATURE_SET,
        "--write-report-copy",
    ])


def prepare_fsi_feature_subset_context(run_seed: int, feature_set_key: str):
    """Create a prebuilt context folder whose scaler exposes only one FSI subset."""

    src_dir = source_context_seed_dir(run_seed)
    dst_dir = context_seed_dir(feature_set_key, run_seed)
    dst_dir.mkdir(parents=True, exist_ok=True)
    features = selected_context_features(feature_set_key)
    normalized = [f"{feature}_normalized" for feature in features]

    source_context = src_dir / "context_features.csv"
    source_scaler = src_dir / "context_scaler.json"
    if not source_context.exists() or not source_scaler.exists():
        raise FileNotFoundError(f"Source FSI context is missing for seed {run_seed}: {src_dir}")

    table = pd.read_csv(source_context)
    required = {"split", "sample_index", *features, *normalized}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Source FSI context missing subset column(s): {missing}")
    table.to_csv(dst_dir / "context_features.csv", index=False)

    scaler = json.loads(source_scaler.read_text(encoding="utf-8"))
    scaler["feature_order"] = features
    scaler["normalized_feature_columns"] = normalized
    if "feature_groups" in scaler:
        scaler["feature_groups"] = {"ofr_fsi_selected": features}
    for key in ["transforms", "medians", "q01", "q99", "means", "stds", "missing_rates"]:
        if isinstance(scaler.get(key), dict):
            scaler[key] = {feature: scaler[key][feature] for feature in features if feature in scaler[key]}
    (dst_dir / "context_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    audit = {
        "status": "ok",
        "source_context_name": SOURCE_CONTEXT_NAME,
        "context_name": prebuilt_context_name(feature_set_key),
        "feature_set_key": feature_set_key,
        "primary_features": features,
        "normalized_feature_columns": normalized,
        "num_rows": int(len(table)),
        "split_counts": table["split"].astype(str).value_counts().sort_index().to_dict(),
    }
    (dst_dir / "context_feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    summary_rows = []
    for split_name, frame in table.groupby(table["split"].astype(str)):
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            norm_values = pd.to_numeric(frame[f"{feature}_normalized"], errors="coerce")
            summary_rows.append({
                "split": split_name,
                "feature": feature,
                "num_rows": int(len(frame)),
                "raw_missing_rate": float(values.isna().mean()),
                "raw_mean": float(values.mean()),
                "raw_std": float(values.std(ddof=1)),
                "raw_min": float(values.min()),
                "raw_max": float(values.max()),
                "normalized_mean": float(norm_values.mean()),
                "normalized_std": float(norm_values.std(ddof=1)),
                "normalized_min": float(norm_values.min()),
                "normalized_max": float(norm_values.max()),
            })
    pd.DataFrame(summary_rows).to_csv(dst_dir / "context_feature_summary.csv", index=False)
    print(f"Prepared {feature_set_key} context for seed {run_seed}: {dst_dir}", flush=True)


def minimal_completed(scale: float, feature_set_key: str, run_seed: int) -> bool:
    exp = experiment_name(scale, feature_set_key)
    seed_dir = f"seed_{run_seed}"
    base = PROJECT_ROOT / "outputs/stage4"
    paths = [
        base / "checkpoints" / exp / seed_dir / "best.pt",
        base / "predictions" / exp / seed_dir / f"{EVAL_SPLIT}_predictions.csv",
        base / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json",
        base / "metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json",
        base / "run_manifests" / exp / seed_dir / "run_manifest.json",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        predictions = pd.read_csv(paths[1])
    except Exception:
        return False
    return len(predictions) >= MIN_PREDICTIONS


def output_check_cmd(run_seed: int):
    return [
        sys.executable, "-u",
        "scripts/check_stage4_outputs.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--context-method", CONTEXT_METHOD,
        "--run-seed", str(run_seed),
        "--split", EVAL_SPLIT,
        "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
        "--min-predictions", str(MIN_PREDICTIONS),
    ]


def stage2_prediction_path(checkpoint_bundle: Path, run_seed: int) -> Path:
    exp = f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"
    return checkpoint_bundle / "outputs/stage2/predictions" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_predictions.csv"


def correction_counts(stage4_prediction_path: Path, checkpoint_bundle: Path, run_seed: int) -> dict:
    stage2_path = stage2_prediction_path(checkpoint_bundle, run_seed)
    empty = {
        "stage2_predictions_available": stage2_path.exists(),
        "stage2_correct_context_correct": None,
        "stage2_correct_context_wrong": None,
        "stage2_wrong_context_correct": None,
        "stage2_wrong_context_wrong": None,
        "net_correction": None,
    }
    if not stage2_path.exists() or not stage4_prediction_path.exists():
        return empty

    s2 = pd.read_csv(stage2_path)
    s4 = pd.read_csv(stage4_prediction_path)
    merged = s2[["sample_index", "correct"]].rename(columns={"correct": "stage2_correct"}).merge(
        s4[["sample_index", "correct"]].rename(columns={"correct": "context_correct"}),
        on="sample_index",
        how="inner",
    )
    stage2_correct = merged["stage2_correct"].astype(bool)
    context_correct = merged["context_correct"].astype(bool)
    correction = int((~stage2_correct & context_correct).sum())
    regression = int((stage2_correct & ~context_correct).sum())
    return {
        "stage2_predictions_available": True,
        "stage2_context_overlap": int(len(merged)),
        "stage2_correct_context_correct": int((stage2_correct & context_correct).sum()),
        "stage2_correct_context_wrong": regression,
        "stage2_wrong_context_correct": correction,
        "stage2_wrong_context_wrong": int((~stage2_correct & ~context_correct).sum()),
        "net_correction": correction - regression,
    }


def read_result_row(
    scale: float,
    feature_set_key: str,
    run_seed: int,
    checkpoint_bundle: Path,
    status: str,
    error: str = "",
) -> dict:
    exp = experiment_name(scale, feature_set_key)
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_trading_metrics.json"
    manifest_path = PROJECT_ROOT / "outputs/stage4/run_manifests" / exp / f"seed_{run_seed}" / "run_manifest.json"
    prediction_path = PROJECT_ROOT / "outputs/stage4/predictions" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_predictions.csv"
    row = {
        "experiment_name": exp,
        "context_method": CONTEXT_METHOD,
        "context_feature_set": context_feature_set_name(feature_set_key),
        "context_feature_set_key": feature_set_key,
        "context_features": ",".join(selected_context_features(feature_set_key)),
        "context_dim": len(selected_context_features(feature_set_key)),
        "modulation_scale": float(scale),
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
            row[key] = metrics.get(key)
        ppr = row.get("predicted_positive_rate")
        row["collapse_warning"] = bool(ppr is not None and (float(ppr) < 0.10 or float(ppr) > 0.90))
    if trading_path.exists():
        trading = json.loads(trading_path.read_text(encoding="utf-8"))
        for strategy_name in ["long_flat", "long_short"]:
            values = trading.get(strategy_name, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                row[f"{strategy_name}_{key}"] = values.get(key)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pretrained = manifest.get("run_context", {}).get("pretrained_stage2", {})
        row["pretrained_loaded"] = pretrained.get("enabled")
        row["num_trainable_parameters"] = pretrained.get("num_trainable_parameters")
        row["num_frozen_parameters"] = pretrained.get("num_frozen_parameters")
    row.update(correction_counts(prediction_path, checkpoint_bundle, run_seed))
    return row


def summarize_seed_results(seed_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(seed_rows)
    if df.empty:
        return pd.DataFrame()
    numeric = [
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "roc_auc",
        "average_precision",
        "f1",
        "brier_score",
        "predicted_positive_rate",
        "stage2_correct_context_wrong",
        "stage2_wrong_context_correct",
        "net_correction",
        "long_flat_sharpe_net",
        "long_short_sharpe_net",
        "long_flat_annualized_return_net",
        "long_short_annualized_return_net",
        "num_trainable_parameters",
        "num_frozen_parameters",
    ]
    rows = []
    for (feature_set_key, scale, method), frame in df.groupby(
        ["context_feature_set_key", "modulation_scale", "context_method"],
        dropna=False,
    ):
        row = {
            "context_feature_set_key": feature_set_key,
            "context_feature_set": str(frame["context_feature_set"].iloc[0]),
            "context_dim": int(frame["context_dim"].iloc[0]) if "context_dim" in frame.columns else None,
            "context_features": str(frame["context_features"].iloc[0]) if "context_features" in frame.columns else "",
            "modulation_scale": scale,
            "context_method": method,
            "seed_count": int(frame["run_seed"].nunique()),
            "collapse_warning_count": int(frame.get("collapse_warning", pd.Series(dtype=bool)).fillna(False).sum()),
        }
        for column in numeric:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                row[f"{column}_mean"] = values.mean()
                row[f"{column}_std"] = values.std(ddof=1)
                row[f"{column}_count"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def make_result_bundle():
    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "reports/figures",
        PROJECT_ROOT / "outputs/stage4/metrics",
        PROJECT_ROOT / "outputs/stage4/run_manifests",
        PROJECT_ROOT / "outputs/stage4/predictions",
        PROJECT_ROOT / "outputs/stage4/context" / SOURCE_CONTEXT_NAME,
        PROJECT_ROOT / "outputs/stage4/figures",
    ]
    include_roots.extend(
        PROJECT_ROOT / "outputs/stage4/context" / prebuilt_context_name(feature_set_key)
        for feature_set_key in CONTEXT_FEATURE_SET_KEYS
    )
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
# 1. Copy code snapshots and locate inputs
# ============================================================
code_input = first_existing(CODE_INPUT_CANDIDATES)
stage2_code_input = first_existing(STAGE2_CODE_INPUT_CANDIDATES)
copy_or_extract_input(code_input, PROJECT_ROOT, expected_child="stage4_film_conditioning")
expected_stage2_checkpoint = (
    STAGE2_PROJECT_ROOT
    / "outputs/stage2/checkpoints"
    / f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"
    / "seed_42/best.pt"
)
if expected_stage2_checkpoint.exists():
    print("Keeping existing Stage 2 working folder with checkpoints:", STAGE2_PROJECT_ROOT, flush=True)
else:
    copy_or_extract_input(stage2_code_input, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")

assert_required_code()
checkpoint_bundle = find_stage2_checkpoint_bundle()
fsi_csv = resolve_fsi_csv()
print("Stage 2 checkpoint bundle:", checkpoint_bundle, flush=True)
print("OFR FSI CSV:", fsi_csv, flush=True)


# ============================================================
# 2. Build FSI context artifacts per seed
# ============================================================
patch_base_config(checkpoint_bundle)
for run_seed in RUN_SEEDS:
    print("\n" + "#" * 80, flush=True)
    print(f"Build OFR FSI context for seed {run_seed}", flush=True)
    print("#" * 80, flush=True)
    build_fsi_context_for_seed(run_seed, fsi_csv)
    for feature_set_key in CONTEXT_FEATURE_SET_KEYS:
        prepare_fsi_feature_subset_context(run_seed, feature_set_key)


# ============================================================
# 3. Train/evaluate FSI-only frozen bounded FiLM
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
for run_seed in RUN_SEEDS:
    for feature_set_key in CONTEXT_FEATURE_SET_KEYS:
        for scale in MODULATION_SCALES:
            patch_prebuilt_fsi_config(scale, feature_set_key, checkpoint_bundle)
            exp = experiment_name(scale, feature_set_key)
            print("\n" + "=" * 80, flush=True)
            print(f"{exp}, seed={run_seed}", flush=True)
            print("=" * 80, flush=True)

            if SKIP_COMPLETED and minimal_completed(scale, feature_set_key, run_seed):
                print(f"[skip] Metrics already exist for {exp}, seed={run_seed}", flush=True)
                summary_rows.append(
                    read_result_row(
                        scale,
                        feature_set_key,
                        run_seed,
                        checkpoint_bundle,
                        status="skipped_completed",
                    )
                )
                continue

            try:
                run([
                    sys.executable, "-u",
                    "scripts/run_stage4_context_model.py",
                    "--config", "configs/env_kaggle.yaml",
                    "--image-window", str(IMAGE_WINDOW),
                    "--image-spec", IMAGE_SPEC,
                    "--return-horizon", str(RETURN_HORIZON),
                    "--context-method", CONTEXT_METHOD,
                    "--run-seed", str(run_seed),
                ] + smoke_train_args)

                run([
                    sys.executable, "-u",
                    "scripts/evaluate_stage4_predictions.py",
                    "--config", "configs/env_kaggle.yaml",
                    "--image-window", str(IMAGE_WINDOW),
                    "--image-spec", IMAGE_SPEC,
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
                    "--image-spec", IMAGE_SPEC,
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
                        "--image-spec", IMAGE_SPEC,
                        "--return-horizon", str(RETURN_HORIZON),
                        "--context-method", CONTEXT_METHOD,
                        "--run-seed", str(run_seed),
                        "--split", EVAL_SPLIT,
                        "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
                        "--write-report-copy",
                    ] + smoke_data_args)
                    run(output_check_cmd(run_seed))
                elif not minimal_completed(scale, feature_set_key, run_seed):
                    raise RuntimeError(f"Minimal output check failed for {exp}, seed={run_seed}")

                summary_rows.append(
                    read_result_row(scale, feature_set_key, run_seed, checkpoint_bundle, status="ok")
                )
            except Exception as exc:
                print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
                summary_rows.append(
                    read_result_row(
                        scale,
                        feature_set_key,
                        run_seed,
                        checkpoint_bundle,
                        status="failed",
                        error=str(exc),
                    )
                )
                if not CONTINUE_ON_ERROR:
                    raise


# ============================================================
# 4. Summary tables and result bundle
# ============================================================
tables_root = PROJECT_ROOT / "reports/tables"
tables_root.mkdir(parents=True, exist_ok=True)
seed_df = pd.DataFrame(summary_rows)
mean_std_df = summarize_seed_results(summary_rows)

seed_csv = tables_root / f"{OUTPUT_PREFIX}_seed_results.csv"
mean_std_csv = tables_root / f"{OUTPUT_PREFIX}_mean_std_results.csv"
run_summary_json = tables_root / f"{OUTPUT_PREFIX}_run_summary.json"
seed_df.to_csv(seed_csv, index=False)
mean_std_df.to_csv(mean_std_csv, index=False)
run_summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
make_result_bundle()

display(Markdown("# Stage 4 N13-2 FSI-Only Seed-Level Results"))
display(seed_df.sort_values(["context_feature_set_key", "modulation_scale", "run_seed"]))

display(Markdown("# Stage 4 N13-2 FSI-Only Mean/Std Summary"))
if not mean_std_df.empty and "accuracy_mean" in mean_std_df.columns:
    display(mean_std_df.sort_values("accuracy_mean", ascending=False))
else:
    display(mean_std_df)

print("\nDONE", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Bundle:", BUNDLE_PATH, flush=True)
```
