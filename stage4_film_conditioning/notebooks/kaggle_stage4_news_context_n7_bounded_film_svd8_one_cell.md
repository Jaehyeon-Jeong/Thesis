# Kaggle Stage 4 N7 News Bounded FiLM SVD8 One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot with bundled news CSV:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`

This runner executes the N7 news-conditioned bounded FiLM main test:

```text
I60 / R20 / ohlc_ma_vb
headline windows: 7d + 20d + 60d
model: CNN + news bounded last-block FiLM
SVD dim: 8
FiLM scale: 0.05
seeds: 42, 43, 44, 45, 46
```

Purpose:
- N6.1 showed SVD dim `8` keeps the strongest ROC-AUC signal among the tested
  news vectors, but still has seed-dependent class collapse.
- N7 tests whether bounded last-block FiLM can use the SVD8 news vector as
  conditional modulation while protecting the Stage 2 visual path.

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

SOURCE_FILE = ""
NEWS_CSV = PROJECT_ROOT / "news_data/BTC_match_title.csv"

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
NEWS_WINDOWS = [7, 20, 60]
SVD_DIMS = [8]
CONTEXT_METHOD = "film_full_bounded_last_block"
MODULATION_SCALE = 0.05
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
MIN_PREDICTIONS = 1000

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
RUN_GRADCAM = True
GRADCAM_SAMPLES_PER_CLASS = 2
RESUME_EXISTING_PROJECT = False
REBUILD_NEWS_FEATURES_IF_MISSING = True

# Smoke check only. For the real N7 run, keep False.
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

OUTPUT_PREFIX = "stage4_news_context_n7_bounded_film_svd8_scale0p05_five_seed"
BUNDLE_PATH = Path("/kaggle/working/stage4_news_context_n7_bounded_film_svd8_result_bundle.zip")


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


def feature_set_name(svd_dim: int) -> str:
    return f"tfidf_svd{int(svd_dim)}_w7_20_60"


def experiment_suffix(svd_dim: int) -> str:
    return f"news_tfidf_svd{int(svd_dim)}_w7_20_60_bounded_scale0p05"


def tfidf_output_prefix(svd_dim: int) -> str:
    return f"stage4_news_tfidf_svd{int(svd_dim)}"


def prebuilt_context_name(svd_dim: int) -> str:
    return (
        f"stage4_news_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_{feature_set_name(svd_dim)}"
    )


def experiment_name(svd_dim: int) -> str:
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_"
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{experiment_suffix(svd_dim)}"
    )


def context_seed_dir(svd_dim: int, run_seed: int) -> Path:
    return PROJECT_ROOT / "outputs/stage4/context" / prebuilt_context_name(svd_dim) / f"seed_{run_seed}"


def assert_required_code():
    """Fail early if an uploaded code snapshot is stale or incomplete."""

    required_stage4 = [
        "scripts/audit_stage4_news_source.py",
        "scripts/audit_stage4_news_alignment.py",
        "scripts/build_stage4_news_headline_windows.py",
        "scripts/build_stage4_news_tfidf_svd.py",
        "scripts/build_stage4_news_context_features.py",
        "scripts/run_stage4_context_model.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/evaluate_stage4_trading.py",
        "scripts/generate_stage4_gradcam_context.py",
        "scripts/check_stage4_outputs.py",
        "src/stage4_film/runners/context_experiment.py",
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

    marker_checks = [
        ("scripts/build_stage4_news_context_features.py", "sample-level news context feature builder"),
        ("src/stage4_film/runners/context_experiment.py", "LoadedContextScaler"),
        ("src/stage4_film/runners/context_experiment.py", "film_full_bounded_last_block"),
        ("src/stage4_film/models/film_stock_cnn.py", "BoundedLastBlockFilmContextStockCNN"),
        ("scripts/check_stage4_outputs.py", "_resolve_context_name"),
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


def patch_base_kaggle_config():
    """Patch Stage 4 config for Kaggle paths before N3/N4/N5 builds."""

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
    cfg["stage2_dependency"]["baseline_output_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2")

    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["context"]["context_window"] = CONTEXT_WINDOW

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
    print("Base config patched:", config_path, flush=True)


def patch_prebuilt_news_config(svd_dim: int, feature_order: list[str]):
    """Patch Stage 4 config so model runners read one prebuilt news context."""

    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    normalized = [f"{feature}_normalized" for feature in feature_order]

    cfg["context"]["source"] = "prebuilt_news"
    cfg["context"]["feature_set_name"] = feature_set_name(svd_dim)
    cfg["context"]["prebuilt_context_name"] = prebuilt_context_name(svd_dim)
    cfg["context"]["primary_features"] = list(feature_order)
    cfg["context"]["normalized_feature_columns"] = normalized

    cfg["stage4_model"]["context_dim"] = len(feature_order)
    cfg["stage4_model"]["experiment_suffix"] = experiment_suffix(svd_dim)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"].setdefault("film_full_bounded_last_block", {})
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = float(MODULATION_SCALE)

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        "Prebuilt news config patched:",
        config_path,
        "svd_dim=",
        svd_dim,
        "context_dim=",
        len(feature_order),
        "context_method=",
        CONTEXT_METHOD,
        "modulation_scale=",
        MODULATION_SCALE,
        flush=True,
    )


def ensure_headline_windows():
    """Build N3 headline-window documents if missing."""

    sample_windows = (
        PROJECT_ROOT
        / f"outputs/stage4/news/stage4_news_headline_windows_i{IMAGE_WINDOW}_r{RETURN_HORIZON}"
        / "sample_headline_windows.parquet"
    )
    if sample_windows.exists():
        print("Headline windows already exist:", sample_windows, flush=True)
        return
    if not REBUILD_NEWS_FEATURES_IF_MISSING:
        raise FileNotFoundError(f"Headline windows missing: {sample_windows}")

    news_csv_args = ["--news-csv", str(NEWS_CSV)] if NEWS_CSV and Path(NEWS_CSV).exists() else []
    if NEWS_CSV and not Path(NEWS_CSV).exists():
        print(f"[warning] local NEWS_CSV not found, trying Hugging Face download: {NEWS_CSV}", flush=True)

    run([
        sys.executable, "-u",
        "scripts/audit_stage4_news_source.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--return-horizon", str(RETURN_HORIZON),
        "--output-prefix", "stage4_news",
    ] + news_csv_args)
    run([
        sys.executable, "-u",
        "scripts/audit_stage4_news_alignment.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--return-horizon", str(RETURN_HORIZON),
        "--output-prefix", "stage4_news_alignment",
    ] + news_csv_args)
    run([
        sys.executable, "-u",
        "scripts/build_stage4_news_headline_windows.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--return-horizon", str(RETURN_HORIZON),
        "--window-days", *map(str, NEWS_WINDOWS),
        "--output-prefix", "stage4_news_headline_windows",
    ] + news_csv_args)


def build_news_vectors(svd_dim: int):
    """Build N4 TF-IDF/SVD vectors for one SVD dimension."""

    output_path = (
        PROJECT_ROOT
        / f"outputs/stage4/news/{tfidf_output_prefix(svd_dim)}_i{IMAGE_WINDOW}_r{RETURN_HORIZON}"
        / "news_tfidf_svd_features.parquet"
    )
    if output_path.exists():
        print(f"SVD dim {svd_dim} vectors already exist:", output_path, flush=True)
        return
    run([
        sys.executable, "-u",
        "scripts/build_stage4_news_tfidf_svd.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--return-horizon", str(RETURN_HORIZON),
        "--window-days", *map(str, NEWS_WINDOWS),
        "--svd-dim", str(svd_dim),
        "--output-prefix", tfidf_output_prefix(svd_dim),
    ])


def build_news_context_for_seed(svd_dim: int, run_seed: int):
    """Build N5 prebuilt news context artifact for one seed and SVD dimension."""

    scaler = context_seed_dir(svd_dim, run_seed) / "context_scaler.json"
    features = context_seed_dir(svd_dim, run_seed) / "context_features.csv"
    if scaler.exists() and features.exists():
        print(f"News context exists for dim {svd_dim}, seed {run_seed}: {context_seed_dir(svd_dim, run_seed)}", flush=True)
        return
    run([
        sys.executable, "-u",
        "scripts/build_stage4_news_context_features.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(run_seed),
        "--window-days", *map(str, NEWS_WINDOWS),
        "--input-prefix", tfidf_output_prefix(svd_dim),
        "--feature-set-name", feature_set_name(svd_dim),
        "--write-report-copy",
    ])


def load_feature_order(svd_dim: int) -> list[str]:
    scaler_path = context_seed_dir(svd_dim, RUN_SEEDS[0]) / "context_scaler.json"
    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_order = [str(feature) for feature in payload["feature_order"]]
    expected = int(svd_dim) * len(NEWS_WINDOWS) + 6
    if len(feature_order) != expected:
        print(f"[warning] expected {expected} features, got {len(feature_order)}", flush=True)
    return feature_order


def minimal_completed(svd_dim: int, run_seed: int) -> bool:
    """Check metric-level completion when Grad-CAM is disabled."""

    exp = experiment_name(svd_dim)
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


def output_check_cmd(svd_dim: int, run_seed: int):
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


def is_completed(svd_dim: int, run_seed: int) -> bool:
    if not RUN_GRADCAM:
        return minimal_completed(svd_dim, run_seed)
    result = run(output_check_cmd(svd_dim, run_seed), capture=True, check=False)
    return result.returncode == 0


def read_result_row(svd_dim: int, run_seed: int, status: str, error: str = "") -> dict:
    exp = experiment_name(svd_dim)
    seed_dir = f"seed_{run_seed}"
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json"
    row = {
        "svd_dim": svd_dim,
        "context_dim": int(svd_dim) * len(NEWS_WINDOWS) + 6,
        "modulation_scale": MODULATION_SCALE,
        "experiment_name": exp,
        "context_method": CONTEXT_METHOD,
        "run_seed": run_seed,
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
            "accuracy_minus_majority_class_accuracy",
            "roc_auc",
            "f1",
            "brier_score",
            "positive_rate",
            "predicted_positive_rate",
        ]:
            row[key] = metrics.get(key)
    if trading_path.exists():
        trading = json.loads(trading_path.read_text(encoding="utf-8"))
        for strategy_name in ["long_flat", "long_short"]:
            values = trading.get(strategy_name, {})
            for key in ["sharpe_net", "annualized_return_net"]:
                row[f"{strategy_name}_{key}"] = values.get(key)
    return row


def summarize_seed_results(seed_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(seed_rows)
    if df.empty:
        return pd.DataFrame()
    numeric = [
        "context_dim",
        "modulation_scale",
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "accuracy_minus_majority_class_accuracy",
        "roc_auc",
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
    for (svd_dim, method), frame in df.groupby(["svd_dim", "context_method"], dropna=False):
        row = {
            "svd_dim": int(svd_dim),
            "context_method": method,
            "seed_count": int(frame["run_seed"].nunique()),
        }
        for column in numeric:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)
            row[f"{column}_count"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_result_bundle(seed_csv: Path, mean_std_csv: Path, run_summary_json: Path):
    """Create one compact downloadable zip for local reporting."""

    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    include_paths = [seed_csv, mean_std_csv, run_summary_json]
    tables = PROJECT_ROOT / "reports/tables"
    include_paths.extend(sorted(tables.glob("stage4_news_context_n7_bounded_film_svd8*")))
    include_paths.extend(sorted(tables.glob("stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd*_seed*_news_context_feature_audit.json")))
    include_paths.extend(sorted(tables.glob("stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd*_seed*_news_context_feature_summary.csv")))
    include_paths.extend(sorted(tables.glob("stage4_news_tfidf_svd*_summary.csv")))
    include_paths.extend(sorted(tables.glob("stage4_news_tfidf_svd*_manifest.json")))
    include_paths.extend(sorted((PROJECT_ROOT / "reports/figures/gradcam").glob("*news_tfidf_svd8*w7_20_60*scale0p05*")))

    for svd_dim in SVD_DIMS:
        exp = experiment_name(svd_dim)
        for run_seed in RUN_SEEDS:
            seed_dir = f"seed_{run_seed}"
            gradcam_dir = PROJECT_ROOT / "outputs/stage4/figures" / exp / seed_dir / "gradcam" / EVAL_SPLIT
            include_paths.extend([
                PROJECT_ROOT / "outputs/stage4/metrics" / exp / seed_dir / "train_history.csv",
                PROJECT_ROOT / "outputs/stage4/metrics" / exp / seed_dir / "train_metadata.json",
                PROJECT_ROOT / "outputs/stage4/metrics" / exp / seed_dir / f"{EVAL_SPLIT}_metrics.json",
                PROJECT_ROOT / "outputs/stage4/metrics" / exp / seed_dir / f"{EVAL_SPLIT}_trading_metrics.json",
                PROJECT_ROOT / "outputs/stage4/run_manifests" / exp / seed_dir / "run_manifest.json",
                gradcam_dir / f"btc_context_gradcam_{EVAL_SPLIT}_{GRADCAM_SAMPLES_PER_CLASS}perclass.png",
                gradcam_dir / "samples.csv",
                gradcam_dir / "modulation_summary.csv",
                gradcam_dir / "modulation_values.json",
            ])

    written = []
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include_paths:
            if not path.exists() or not path.is_file():
                continue
            rel = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else Path(path.name)
            archive.write(path, arcname=str(rel))
            written.append(str(rel))
        manifest = {
            "bundle": str(BUNDLE_PATH),
            "svd_dims": SVD_DIMS,
            "run_seeds": RUN_SEEDS,
            "context_method": CONTEXT_METHOD,
            "modulation_scale": MODULATION_SCALE,
            "num_files": len(written),
            "files": written,
        }
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print("Bundle written:", BUNDLE_PATH, f"({BUNDLE_PATH.stat().st_size / (1024 * 1024):.2f} MB)", flush=True)


# ============================================================
# 1. Copy code snapshots and patch base config
# ============================================================
if RESUME_EXISTING_PROJECT:
    if not PROJECT_ROOT.exists() or not STAGE2_PROJECT_ROOT.exists():
        raise FileNotFoundError(
            "RESUME_EXISTING_PROJECT=True requires existing PROJECT_ROOT and STAGE2_PROJECT_ROOT."
        )
    print(f"Resuming existing Stage 4 project: {PROJECT_ROOT}", flush=True)
    print(f"Resuming existing Stage 2 dependency: {STAGE2_PROJECT_ROOT}", flush=True)
else:
    copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
    copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")

assert_required_code()
patch_base_kaggle_config()
print(f"Stage 4 project ready at: {PROJECT_ROOT}", flush=True)
print(f"Stage 2 dependency ready at: {STAGE2_PROJECT_ROOT}", flush=True)


# ============================================================
# 2. Build headline windows, SVD vectors, and context artifacts
# ============================================================
ensure_headline_windows()
for svd_dim in SVD_DIMS:
    print("\n" + "#" * 88, flush=True)
    print(f"Preparing SVD dim {svd_dim}", flush=True)
    print("#" * 88, flush=True)
    build_news_vectors(svd_dim)
    for run_seed in RUN_SEEDS:
        build_news_context_for_seed(svd_dim, run_seed)


# ============================================================
# 3. Train/evaluate news-conditioned bounded FiLM over SVD8 and five seeds
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
for svd_dim in SVD_DIMS:
    feature_order = load_feature_order(svd_dim)
    patch_prebuilt_news_config(svd_dim, feature_order)

    for run_seed in RUN_SEEDS:
        exp = experiment_name(svd_dim)
        print("\n" + "=" * 88, flush=True)
        print(f"{exp}, seed={run_seed}", flush=True)
        print("=" * 88, flush=True)

        if SKIP_COMPLETED and is_completed(svd_dim, run_seed):
            print(f"[skip] Metric output already exists for {exp}, seed={run_seed}", flush=True)
            summary_rows.append(read_result_row(svd_dim, run_seed, status="skipped_completed"))
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
                run(output_check_cmd(svd_dim, run_seed))

            summary_rows.append(read_result_row(svd_dim, run_seed, status="ok"))
        except Exception as exc:
            print(f"[error] {exp}, seed={run_seed}: {exc}", flush=True)
            summary_rows.append(read_result_row(svd_dim, run_seed, status="failed", error=str(exc)))
            if not CONTINUE_ON_ERROR:
                raise


# ============================================================
# 4. Write summaries and compact result bundle
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
write_result_bundle(seed_csv, mean_std_csv, run_summary_json)

display(Markdown("# Stage 4 N7 News Bounded FiLM SVD8 Seed-Level Results"))
display(seed_df.sort_values(["svd_dim", "run_seed"]))

display(Markdown("# Stage 4 N7 News Bounded FiLM SVD8 Mean/Std Summary"))
if not mean_std_df.empty and "accuracy_mean" in mean_std_df.columns:
    display(mean_std_df.sort_values(["accuracy_mean", "roc_auc_mean"], ascending=False))
else:
    display(mean_std_df)

display(Markdown("# Reference"))
display(pd.DataFrame([
    {
        "reference": "N6.1 SVD8 concat",
        "context_dim": 30,
        "accuracy_mean": 0.5407,
        "roc_auc_mean": 0.5817,
        "note": "Best news-vector ROC-AUC; seeds 45/46 collapsed Down.",
    },
    {
        "reference": "N6 dim32 concat",
        "context_dim": 102,
        "accuracy_mean": 0.5478,
        "roc_auc_mean": 0.5644,
        "note": "Useful signal in some seeds, but seeds 43/45 collapsed.",
    },
    {
        "reference": "Stage2 visual baseline",
        "context_dim": 0,
        "accuracy_mean": 0.5793,
        "roc_auc_mean": 0.5849,
        "note": "Target baseline before N7.",
    },
]))

print("\nDONE", flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Bundle:", BUNDLE_PATH, flush=True)
```
