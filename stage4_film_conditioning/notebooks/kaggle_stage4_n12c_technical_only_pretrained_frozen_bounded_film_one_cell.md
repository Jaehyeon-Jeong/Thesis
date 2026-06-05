# Kaggle Stage 4 N12-C Technical-Only Pretrained Frozen Bounded FiLM One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- Stage 2 selected checkpoint bundle from N8-A0:
  `stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- F&G Kaggle dataset (attached for source-audit compatibility; N12-C uses OHLCV-derived technical features):
  `ashishpatel8736/historical-and-fear-greed-index-datasets`

This runner executes 4-N12-C:

```text
Stage 2 I60/R20/ohlc_ma_vb checkpoint -> load learned weights
freeze visual CNN blocks + freeze classifier
train only technical context encoder + bounded final-block FiLM heads
context: bb_percent_b_60, bb_bandwidth_60, mfi_60, rv_60
scale grid: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
```

Purpose:
- Previous Stage 4 runs trained a Stage2-style CNN from scratch.
- N12-C tests the context-source question: do image-derived technical indicators add useful correction after the strong Stage 2 visual model is frozen?

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

# Leave empty to auto-detect Stage 2 checkpoints.
# Auto-detect checks /kaggle/working/stage2_btc_extension first, then
# /kaggle/working checkpoint bundle folders/zips, then /kaggle/input.
# If auto-detect fails, set this to a folder that contains
# outputs/stage2/checkpoints/stage2_i60_ohlc_ma_vb_r20/seed_42/best.pt.
STAGE2_CHECKPOINT_BUNDLE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
CONTEXT_FEATURES = ["bb_percent_b_60", "bb_bandwidth_60", "mfi_60", "rv_60"]
CONTEXT_FEATURE_SET = "technical_only"
MODULATION_SCALES = [0.02, 0.05]
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
RUN_GRADCAM = True
GRADCAM_SAMPLES_PER_CLASS = 2
RESUME_EXISTING_PROJECT = False

# Smoke check only. For the real N12-C run, keep False.
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

OUTPUT_PREFIX = "stage4_n12c_technical_only_pretrained_frozen_bounded_film"
BUNDLE_PATH = Path("/kaggle/working/stage4_n12c_technical_only_pretrained_frozen_bounded_film_result_bundle.zip")


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
    text = f"{float(scale):.3f}".rstrip("0").rstrip(".")
    return "s" + text.replace(".", "p")


def experiment_suffix(scale: float) -> str:
    return f"n12c_technical_pretrained_frozen_{scale_label(scale)}"


def experiment_name(scale: float) -> str:
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{experiment_suffix(scale)}"
    )


def context_name() -> str:
    return (
        f"stage4_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{CONTEXT_FEATURE_SET}"
    )


def find_stage2_checkpoint_bundle() -> Path:
    """Find Stage 2 checkpoints under /kaggle/working first, then /kaggle/input."""

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

    search_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    for search_root in search_roots:
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
    """Return a folder usable by Stage 4, extracting zip bundles if needed."""

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

    nested = next(
        candidate.rglob(f"outputs/stage2/checkpoints/{expected_exp}/seed_42/best.pt"),
        None,
    )
    if nested is not None:
        return nested.parents[5]

    raise FileNotFoundError(f"No Stage 2 checkpoint found under: {candidate}")


def assert_required_code():
    """Fail early if an uploaded code snapshot is stale or incomplete."""

    required_stage4 = [
        "scripts/build_stage4_context_features.py",
        "scripts/run_stage4_context_model.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/evaluate_stage4_trading.py",
        "scripts/generate_stage4_gradcam_context.py",
        "scripts/check_stage4_outputs.py",
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


def patch_config(scale: float, checkpoint_bundle: Path):
    """Patch Stage 4 Kaggle config for one N12-C scale."""

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
    cfg["context"]["primary_features"] = list(CONTEXT_FEATURES)
    cfg["context"]["feature_set_name"] = CONTEXT_FEATURE_SET
    cfg["context"]["fear_greed"]["kaggle_file"] = FEAR_GREED_FILE

    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(CONTEXT_FEATURES)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = experiment_suffix(scale)
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = float(scale)
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


def output_check_cmd(scale: float, run_seed: int):
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


def is_completed(scale: float, run_seed: int) -> bool:
    patch_config(scale, checkpoint_bundle)
    result = run(output_check_cmd(scale, run_seed), capture=True, check=False)
    return result.returncode == 0


def read_result_row(scale: float, run_seed: int, status: str, error: str = "") -> dict:
    exp = experiment_name(scale)
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_trading_metrics.json"
    manifest_path = PROJECT_ROOT / "outputs/stage4/run_manifests" / exp / f"seed_{run_seed}" / "run_manifest.json"
    row = {
        "experiment_name": exp,
        "context_method": CONTEXT_METHOD,
        "context_feature_set": CONTEXT_FEATURE_SET,
        "modulation_scale": float(scale),
        "run_seed": int(run_seed),
        "status": status,
        "error": error,
        "classification_available": metrics_path.exists(),
        "trading_available": trading_path.exists(),
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ["num_samples", "accuracy", "majority_class_accuracy", "roc_auc", "average_precision", "f1", "brier_score", "predicted_positive_rate"]:
            row[key] = metrics.get(key)
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
        "long_flat_sharpe_net",
        "long_short_sharpe_net",
        "long_flat_annualized_return_net",
        "long_short_annualized_return_net",
        "num_trainable_parameters",
        "num_frozen_parameters",
    ]
    rows = []
    for (scale, method), frame in df.groupby(["modulation_scale", "context_method"], dropna=False):
        row = {
            "modulation_scale": scale,
            "context_method": method,
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


def make_result_bundle():
    """Zip compact N12-C outputs for local download."""

    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "reports/figures",
        PROJECT_ROOT / "outputs/stage4/metrics",
        PROJECT_ROOT / "outputs/stage4/run_manifests",
        PROJECT_ROOT / "outputs/stage4/figures",
    ]
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in include_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(PROJECT_ROOT))
    print(f"Result bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / (1024 * 1024):.2f} MB)", flush=True)


# ============================================================
# 1. Copy code snapshots and locate checkpoint bundle
# ============================================================
if not RESUME_EXISTING_PROJECT:
    copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
    expected_stage2_checkpoint = (
        STAGE2_PROJECT_ROOT
        / "outputs/stage2/checkpoints"
        / f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"
        / "seed_42/best.pt"
    )
    if expected_stage2_checkpoint.exists():
        print(
            "Keeping existing Stage 2 working folder with checkpoints:",
            STAGE2_PROJECT_ROOT,
            flush=True,
        )
    else:
        copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_required_code()
checkpoint_bundle = find_stage2_checkpoint_bundle()
print("Stage 2 checkpoint bundle:", checkpoint_bundle, flush=True)


# ============================================================
# 2. Source audit and context feature build
# ============================================================
patch_config(MODULATION_SCALES[0], checkpoint_bundle)
run([
    sys.executable, "-u",
    "scripts/audit_stage4_context_sources.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--return-horizon", str(RETURN_HORIZON),
    "--output", "reports/tables/stage4_n12c_context_source_audit.json",
])

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
    print("\n" + "#" * 80, flush=True)
    print(f"Build technical context for seed {run_seed}", flush=True)
    print("#" * 80, flush=True)
    patch_config(MODULATION_SCALES[0], checkpoint_bundle)
    run([
        sys.executable, "-u",
        "scripts/build_stage4_context_features.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(run_seed),
        "--write-report-copy",
    ] + smoke_data_args)

    for scale in MODULATION_SCALES:
        patch_config(scale, checkpoint_bundle)
        exp = experiment_name(scale)
        print("\n" + "=" * 80, flush=True)
        print(f"{exp}, seed={run_seed}", flush=True)
        print("=" * 80, flush=True)

        if SKIP_COMPLETED and is_completed(scale, run_seed):
            print(f"[skip] Output check already passes for {exp}, seed={run_seed}", flush=True)
            summary_rows.append(read_result_row(scale, run_seed, status="skipped_completed"))
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

                run(output_check_cmd(scale, run_seed))

            summary_rows.append(read_result_row(scale, run_seed, status="ok"))
        except Exception as exc:
            print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
            summary_rows.append(read_result_row(scale, run_seed, status="failed", error=str(exc)))
            if not CONTINUE_ON_ERROR:
                raise


# ============================================================
# 3. Summary tables and result bundle
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

display(Markdown("# Stage 4 N12-C Seed-Level Results"))
display(seed_df.sort_values(["modulation_scale", "run_seed"]))

display(Markdown("# Stage 4 N12-C Mean/Std Summary"))
if not mean_std_df.empty and "accuracy_mean" in mean_std_df.columns:
    display(mean_std_df.sort_values("accuracy_mean", ascending=False))
else:
    display(mean_std_df)

print("\nDONE", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Bundle:", BUNDLE_PATH, flush=True)
```
