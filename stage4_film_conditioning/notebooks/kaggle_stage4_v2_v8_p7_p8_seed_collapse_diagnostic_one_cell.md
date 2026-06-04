# Kaggle Stage 4 v2 V8 P7/P8 Seed-Collapse Diagnostic One Cell

Copy the Python cell below into Kaggle after P7 and P8 have been trained in the
same `/kaggle/working/stage4_film_conditioning` workspace, or after their
outputs/checkpoints have been restored.

This diagnostic does **not** train new models. It:

- keeps `I60/R20/ohlc_ma_vb`;
- compares P7 `film_full` and P8 `film_full_bounded_last_block`;
- uses F&G-only context: `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`;
- exports validation and test predictions if they are missing;
- calibrates a threshold on validation and applies it to test;
- writes seed-collapse tables and a compact Markdown report.

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
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning")
STAGE2_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect under /kaggle/input.
SOURCE_FILE = ""
FEAR_GREED_FILE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_FEATURE_SET_NAME = "fg_only"
PRIMARY_CONTEXT_FEATURES = ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"]
CONTEXT_METHODS = ["film_full", "film_full_bounded_last_block"]
BOUNDED_FILM_SCALE = 0.10
RUN_SEEDS = [42, 43, 44, 45, 46]
SPLITS_FOR_DIAGNOSTIC = ["validation", "test"]

# Set True only if you want to overwrite existing prediction CSVs.
FORCE_REEVALUATE_PREDICTIONS = False

# Keep True for this diagnostic so existing trained outputs are not deleted.
RESUME_EXISTING_PROJECT = True

# Keep True when you uploaded a newer Stage 4 code snapshot but want to preserve
# existing /kaggle/working/stage4_film_conditioning/outputs.
REFRESH_CODE_PRESERVING_OUTPUTS = True

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False

OUTPUT_PREFIX = "stage4_v2_v8_p7_p8_seed_collapse"


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


def extract_or_copy_to_temp(src: Path, expected_child: str | None = None) -> Path:
    """Return a temporary copied/extracted code snapshot path."""

    tmp = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}_code_refresh_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        candidate = src / expected_child if expected_child and (src / expected_child).exists() else src
        target = tmp / candidate.name
        shutil.copytree(candidate, target)
        return target
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as archive:
            archive.extractall(tmp)
        candidate = tmp / expected_child if expected_child and (tmp / expected_child).exists() else tmp
        if expected_child and not (candidate / "scripts").exists():
            nested = next(tmp.rglob(expected_child), None)
            if nested is not None:
                candidate = nested
        return candidate
    raise FileNotFoundError(f"Input must be a folder or .zip file: {src}")


def refresh_code_preserving_outputs(src: Path, dst: Path, expected_child: str | None = None):
    """Refresh code/docs/config folders without deleting existing outputs."""

    candidate = extract_or_copy_to_temp(src, expected_child=expected_child)
    refresh_paths = [
        "src",
        "scripts",
        "configs",
        "notebooks",
        "docs",
        "checklist_results",
        "README.md",
        "checklist.md",
        "workflow_diagram.md",
    ]
    for rel_path in refresh_paths:
        source = candidate / rel_path
        if not source.exists():
            continue
        target = dst / rel_path
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    tmp_root = candidate.parent
    if tmp_root.exists() and tmp_root.name.endswith("_code_refresh_tmp"):
        shutil.rmtree(tmp_root)


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
    """Fail early if the uploaded snapshot lacks V8 diagnostic support."""

    marker_checks = {
        "src/stage4_film/config.py": "film_full_bounded_last_block",
        "src/stage4_film/context/features.py": "context_suffix",
        "src/stage4_film/models/film_stock_cnn.py": "BoundedLastBlockFilmContextStockCNN",
        "src/stage4_film/runners/context_experiment.py": "film_full_bounded_last_block",
        "scripts/analyze_stage4_seed_collapse.py": "validation threshold calibration",
    }
    stale = []
    for rel_path, marker in marker_checks.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            stale.append(f"{rel_path} missing marker {marker!r}")
    if stale:
        raise RuntimeError(
            "Stage 4 code snapshot is stale/incomplete. Upload the latest "
            "stage4_film_conditioning folder/zip, then rerun. Problems: "
            + "; ".join(stale)
        )

    required_stage2 = [
        "src/stage2_btc/data",
        "src/stage2_btc/evaluation",
        "src/stage2_btc/models/stock_cnn.py",
    ]
    missing_stage2 = [path for path in required_stage2 if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing_stage2:
        raise RuntimeError("Stage 2 dependency snapshot is incomplete: " + ", ".join(missing_stage2))


def patch_kaggle_config():
    """Patch Stage 4 Kaggle config with diagnostic settings."""

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
    cfg["stage4_model"].setdefault("film_full_bounded_last_block", {})
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = BOUNDED_FILM_SCALE

    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["mixed_precision"] = MIXED_PRECISION
    cfg["training"]["data_parallel"] = DATA_PARALLEL
    cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
    cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
    cfg["evaluation"]["batch_size"] = BATCH_SIZE

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Config patched:", config_path, flush=True)


def ensure_context_features(run_seed: int):
    path = (
        PROJECT_ROOT
        / f"outputs/stage4/context/{context_name()}/seed_{run_seed}/context_features.csv"
    )
    if path.exists():
        return
    run([
        sys.executable, "-u",
        "scripts/build_stage4_context_features.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(run_seed),
        "--write-report-copy",
    ])


def ensure_predictions(context_method: str, run_seed: int, split: str):
    exp = experiment_name(context_method)
    checkpoint = PROJECT_ROOT / f"outputs/stage4/checkpoints/{exp}/seed_{run_seed}/best.pt"
    if not checkpoint.exists():
        print(f"[skip] missing checkpoint: {checkpoint}", flush=True)
        return
    prediction = PROJECT_ROOT / f"outputs/stage4/predictions/{exp}/seed_{run_seed}/{split}_predictions.csv"
    if prediction.exists() and not FORCE_REEVALUATE_PREDICTIONS:
        print(f"[ok] prediction exists: {prediction}", flush=True)
        return
    run([
        sys.executable, "-u",
        "scripts/evaluate_stage4_predictions.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--context-method", context_method,
        "--run-seed", str(run_seed),
        "--split", split,
    ])


# ============================================================
# 1. Prepare code/config without deleting existing outputs
# ============================================================
if RESUME_EXISTING_PROJECT and PROJECT_ROOT.exists() and STAGE2_PROJECT_ROOT.exists():
    print(f"Using existing Stage 4 project: {PROJECT_ROOT}", flush=True)
    print(f"Using existing Stage 2 dependency: {STAGE2_PROJECT_ROOT}", flush=True)
    if REFRESH_CODE_PRESERVING_OUTPUTS:
        refresh_code_preserving_outputs(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
        print("Stage 4 code refreshed while preserving existing outputs.", flush=True)
else:
    copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
    copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
    print(f"Stage 4 code copied to: {PROJECT_ROOT}", flush=True)
    print(f"Stage 2 dependency copied to: {STAGE2_PROJECT_ROOT}", flush=True)

assert_required_code()
patch_kaggle_config()


# ============================================================
# 2. Ensure validation/test predictions exist
# ============================================================
for run_seed in RUN_SEEDS:
    ensure_context_features(run_seed)
    for context_method in CONTEXT_METHODS:
        for split in SPLITS_FOR_DIAGNOSTIC:
            ensure_predictions(context_method, run_seed, split)


# ============================================================
# 3. Run seed-collapse diagnostic
# ============================================================
run([
    sys.executable, "-u",
    "scripts/analyze_stage4_seed_collapse.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--context-window", str(CONTEXT_WINDOW),
    "--context-suffix", CONTEXT_FEATURE_SET_NAME,
    "--methods", *CONTEXT_METHODS,
    "--run-seeds", *map(str, RUN_SEEDS),
    "--splits", *SPLITS_FOR_DIAGNOSTIC,
    "--calibration-metric", "balanced_accuracy",
    "--output-prefix", OUTPUT_PREFIX,
])

tables_root = PROJECT_ROOT / "reports/tables"
default_metrics = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_default_metrics.csv")
thresholds = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_threshold_calibration.csv")
pairwise = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_pairwise_comparison.csv")
quantiles = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_probability_quantiles.csv")
report_path = tables_root / f"{OUTPUT_PREFIX}_report.md"

display(Markdown("# 4-V8 Default Metrics"))
display(default_metrics.sort_values(["split", "run_seed", "context_method"]))

display(Markdown("# 4-V8 Validation Threshold Calibration"))
display(thresholds.sort_values(["run_seed", "context_method"]))

display(Markdown("# 4-V8 Pairwise P7/P8 Comparison"))
display(pairwise.sort_values(["split", "run_seed"]))

display(Markdown("# 4-V8 Probability Quantiles"))
display(quantiles.sort_values(["split", "run_seed", "context_method"]))

display(Markdown(report_path.read_text(encoding="utf-8")))

print("\nDONE", flush=True)
print("Default metrics:", tables_root / f"{OUTPUT_PREFIX}_default_metrics.csv", flush=True)
print("Threshold calibration:", tables_root / f"{OUTPUT_PREFIX}_threshold_calibration.csv", flush=True)
print("Pairwise comparison:", tables_root / f"{OUTPUT_PREFIX}_pairwise_comparison.csv", flush=True)
print("Probability quantiles:", tables_root / f"{OUTPUT_PREFIX}_probability_quantiles.csv", flush=True)
print("Report:", report_path, flush=True)
```
