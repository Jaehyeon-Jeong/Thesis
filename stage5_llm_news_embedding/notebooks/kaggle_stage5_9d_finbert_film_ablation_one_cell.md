# Kaggle Stage 5 5-9D FinBERT FiLM Ablation One Cell

Copy the Python cell below into Kaggle after attaching:

- latest `stage4_film_conditioning` code snapshot
- latest `stage2_btc_extension` code snapshot
- Stage 2 `I60/R20/ohlc_ma_vb` seed 42-46 checkpoint bundle
- Stage 5 FinBERT compact context bundle
- BTC OHLCV and news/F&G Kaggle datasets already used by Stage 4

This runner executes `5-9D`:

```text
image spec: ohlc_ma_vb
visual baseline: Stage2 I60/R20/ohlc_ma_vb checkpoints
context: FinBERT headline sentiment 7/20/60-day window features
feature source: FinBERT positive/negative/neutral probabilities, sentiment, counts, and news-FG proxy
model: Stage2 visual CNN + classifier frozen
context method: bounded final-block FiLM
scale: 0.02
seeds: 42,43,44,45,46
```

The cell rebuilds Stage 4 prebuilt context artifacts from the Stage 5 FinBERT
context table, then trains/evaluates the bounded FiLM correction model.

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
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n14b_snapshot/stage4_film_conditioning"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n16_snapshot/stage4_film_conditioning"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_stage5_snapshot/stage4_film_conditioning"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_stage5_snapshot.zip"),
    Path("/kaggle/input/stage4-film-conditioning/stage4_film_conditioning"),
    Path("/kaggle/input/stage4-film-conditioning-stage5-snapshot/stage4_film_conditioning"),
]
STAGE2_CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage22/stage2_btc_extension_latest_for_n13_6/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage22/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage4/stage2_btc_extension"),
]
STAGE5_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding_5_9d_finbert_context_bundle"),
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding_5_9d_finbert_context_bundle.zip"),
    Path("/kaggle/input/stage5-llm-news-embedding-5-9d-finbert-context-bundle/stage5_llm_news_embedding"),
    Path("/kaggle/input/stage5-llm-news-embedding-5-9d-finbert-context-bundle/stage5_llm_news_embedding_5_9d_finbert_context_bundle.zip"),
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding"),
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding_5_8_compact_context_bundle"),
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding_5_8_compact_context_bundle.zip"),
    Path("/kaggle/input/stage5-llm-news-embedding-5-8-compact-context-bundle/stage5_llm_news_embedding"),
    Path("/kaggle/input/stage5-llm-news-embedding-5-8-compact-context-bundle/stage5_llm_news_embedding_5_8_compact_context_bundle.zip"),
]

PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
STAGE5_PROJECT_ROOT = Path("/kaggle/working/stage5_llm_news_embedding")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect under /kaggle/input.
SOURCE_FILE = ""
FEAR_GREED_FILE = ""

# Stage 2 CHECKPOINT/RESULT bundle. This is not the Stage 2 source-code folder.
# It must contain:
# outputs/stage2/checkpoints/stage2_i60_ohlc_ma_vb_r20/seed_42/best.pt
# ...
# outputs/stage2/checkpoints/stage2_i60_ohlc_ma_vb_r20/seed_46/best.pt
#
# The N15-A four-image bundle is valid because it contains ohlc_ma_vb.
# Leave empty to auto-detect from Kaggle datasets.
STAGE2_CHECKPOINT_BUNDLE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
MODULATION_SCALE = 0.02
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
MIN_PREDICTIONS = 1000

FINBERT_RUN_ID = "finbert_prosusai_headline_v1"
FINBERT_FEATURE_LABEL = "finbert_sentiment_v1"


def scale_label(scale: float) -> str:
    text = f"{float(scale):.3f}".rstrip("0").rstrip(".")
    return "s" + text.replace(".", "p")


SCALE_LABEL = scale_label(MODULATION_SCALE)
FEATURE_SET_NAME = f"stage5_{FINBERT_FEATURE_LABEL}"
EXPERIMENT_SUFFIX = f"{FEATURE_SET_NAME}_pretrained_frozen_{SCALE_LABEL}"
CONTEXT_NAME = (
    f"stage5_finbert_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
    f"r{RETURN_HORIZON}_{FINBERT_FEATURE_LABEL}"
)

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
RUN_GRADCAM = False
GRADCAM_SAMPLES_PER_CLASS = 2

# Smoke check only. For the real 5-9D run, keep False.
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
LOG_EVERY_BATCHES = 20

OUTPUT_PREFIX = (
    f"stage5_9d_finbert_sentiment_"
    f"pretrained_frozen_bounded_film_{SCALE_LABEL}"
)
BUNDLE_PATH = Path(f"/kaggle/working/{OUTPUT_PREFIX}_result_bundle.zip")


def first_existing(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in candidates))


def looks_like_stage4_code_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "scripts/run_stage4_context_model.py").exists()
        and (path / "src/stage4_film").exists()
    )


def find_stage4_code_input() -> Path:
    for candidate in CODE_INPUT_CANDIDATES:
        candidate = Path(candidate)
        if candidate.exists():
            if candidate.is_file() and candidate.suffix.lower() == ".zip":
                return candidate
            if looks_like_stage4_code_root(candidate):
                return candidate
            nested = candidate / "stage4_film_conditioning"
            if looks_like_stage4_code_root(nested):
                return nested

    for script_path in Path("/kaggle/input").rglob("run_stage4_context_model.py"):
        root = script_path.parents[1]
        if looks_like_stage4_code_root(root):
            return root

    inspected = []
    for path in Path("/kaggle/input").rglob("*stage4*"):
        inspected.append(str(path))
        if len(inspected) >= 40:
            break
    raise FileNotFoundError(
        "Could not find Stage 4 code input. Expected a folder containing "
        "scripts/run_stage4_context_model.py and src/stage4_film. "
        "First stage4-like paths under /kaggle/input: "
        + json.dumps(inspected, indent=2)
    )


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


def stage2_experiment_name() -> str:
    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def stage4_experiment_name() -> str:
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{EXPERIMENT_SUFFIX}"
    )


def required_stage2_checkpoint(run_seed: int) -> Path:
    return (
        Path("outputs/stage2/checkpoints")
        / stage2_experiment_name()
        / f"seed_{int(run_seed)}"
        / "best.pt"
    )


def has_all_stage2_checkpoints(root: Path) -> bool:
    return all((root / required_stage2_checkpoint(seed)).exists() for seed in RUN_SEEDS)


def has_all_stage2_checkpoints_in_experiment_dir(experiment_dir: Path) -> bool:
    return all((experiment_dir / f"seed_{int(seed)}" / "best.pt").exists() for seed in RUN_SEEDS)


def resolve_checkpoint_candidate(candidate: Path) -> Path:
    candidate = candidate.expanduser()
    if candidate.is_file():
        if candidate.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported checkpoint candidate file: {candidate}")
        extract_root = Path("/kaggle/working/stage2_stage5_checkpoint_bundle_extracted") / candidate.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(extract_root)
        return resolve_checkpoint_candidate(extract_root)
    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint candidate missing: {candidate}")
    if has_all_stage2_checkpoints(candidate):
        return candidate
    if candidate.name == stage2_experiment_name() and has_all_stage2_checkpoints_in_experiment_dir(candidate):
        # candidate == <root>/outputs/stage2/checkpoints/<experiment>
        return candidate.parents[3]
    nested_best = next(candidate.rglob(str(required_stage2_checkpoint(42))), None)
    if nested_best is not None:
        root = nested_best.parents[5]
        if has_all_stage2_checkpoints(root):
            return root
    nested_exp_dir = next(candidate.rglob(stage2_experiment_name()), None)
    if nested_exp_dir is not None and has_all_stage2_checkpoints_in_experiment_dir(nested_exp_dir):
        return nested_exp_dir.parents[3]
    raise FileNotFoundError(f"No complete Stage 2 checkpoint bundle under: {candidate}")


def find_stage2_checkpoint_bundle() -> Path:
    if str(STAGE2_CHECKPOINT_BUNDLE).strip():
        return resolve_checkpoint_candidate(Path(str(STAGE2_CHECKPOINT_BUNDLE)).expanduser())
    candidates = [
        # Preferred current reusable Stage2 bundle: contains all four I60/R20 image specs.
        Path("/kaggle/input/datasets/moskow/stage22/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
        Path("/kaggle/input/datasets/moskow/stage22/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"),
        Path("/kaggle/input/datasets/moskow/stage22/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"),
        Path("/kaggle/input/stage2-i60-r20-four-image-full-seed-checkpoints-for-stage4-n15/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
        Path("/kaggle/input/stage2-i60-r20-four-image-full-seed-checkpoints-for-stage4-n15/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"),
        Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
        Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"),
        # Older N8-A0 ohlc_ma_vb-only bundle candidates.
        Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"),
        Path("/kaggle/input/datasets/moskow/stage22/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"),
        Path("/kaggle/input/datasets/moskow/stage22/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8.zip"),
        Path("/kaggle/input/stage2-i60-ohlc-ma-vb-r20-seed42-46-checkpoints-for-stage4-n8/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"),
        Path("/kaggle/working/stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"),
        Path("/kaggle/working/stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return resolve_checkpoint_candidate(candidate)
            except Exception as exc:
                print(f"[checkpoint search] rejected {candidate}: {exc}", flush=True)
    for pattern in [
        "*stage2*i60*r20*four*image*",
        "*stage2*i60*ohlc*ma*vb*r20*",
        stage2_experiment_name(),
    ]:
        for candidate in Path("/kaggle/input").rglob(pattern):
            try:
                return resolve_checkpoint_candidate(candidate)
            except Exception:
                pass
    for candidate in Path("/kaggle/working").rglob(stage2_experiment_name()):
        try:
            return resolve_checkpoint_candidate(candidate)
        except Exception:
            pass
    raise FileNotFoundError(
        "Could not find Stage 2 checkpoint bundle. Attach the N8-A0 bundle dataset "
        "or set STAGE2_CHECKPOINT_BUNDLE manually."
    )


def assert_required_code():
    required_stage4 = [
        "scripts/run_stage4_context_model.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/evaluate_stage4_trading.py",
        "scripts/check_stage4_outputs.py",
        "src/stage4_film/config.py",
        "src/stage4_film/models/film_stock_cnn.py",
        "src/stage4_film/runners/context_experiment.py",
    ]
    missing_stage4 = [path for path in required_stage4 if not (PROJECT_ROOT / path).exists()]
    if missing_stage4:
        raise RuntimeError("Stage 4 code snapshot is incomplete: " + ", ".join(missing_stage4))

    required_stage5 = [
        "scripts/build_stage5_stage4_prebuilt_context.py",
        "outputs/stage5/finbert_context/finbert_prosusai_headline_v1/stage5_finbert_context_features.csv",
    ]
    missing_stage5 = [path for path in required_stage5 if not (STAGE5_PROJECT_ROOT / path).exists()]
    if missing_stage5:
        raise RuntimeError("Stage 5 compact bundle is incomplete: " + ", ".join(missing_stage5))

    required_stage2 = [
        "src/stage2_btc/data",
        "src/stage2_btc/evaluation",
        "src/stage2_btc/models/stock_cnn.py",
    ]
    missing_stage2 = [path for path in required_stage2 if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing_stage2:
        raise RuntimeError("Stage 2 dependency snapshot is incomplete: " + ", ".join(missing_stage2))


def feature_table_path() -> Path:
    return (
        STAGE5_PROJECT_ROOT
        / "outputs/stage5/finbert_context"
        / FINBERT_RUN_ID
        / "stage5_finbert_context_features.csv"
    )


def read_feature_order_from_context() -> list[str]:
    scaler_path = (
        PROJECT_ROOT
        / "outputs/stage4/context"
        / CONTEXT_NAME
        / f"seed_{RUN_SEEDS[0]}"
        / "context_scaler.json"
    )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Context scaler missing after build: {scaler_path}")
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    return [str(feature) for feature in scaler["feature_order"]]


def build_prebuilt_stage4_context():
    run([
        sys.executable, "-u",
        "scripts/build_stage5_stage4_prebuilt_context.py",
        "--feature-table", str(feature_table_path()),
        "--stage4-root", str(PROJECT_ROOT),
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--context-window", str(CONTEXT_WINDOW),
        "--feature-source", "finbert",
        "--feature-prefix", "finbert_",
        "--aggregation", "mean",
        "--dimension", "0",
        "--run-seeds", *[str(seed) for seed in RUN_SEEDS],
        "--context-name", CONTEXT_NAME,
        "--feature-set-name", FEATURE_SET_NAME,
        "--write-report-copy",
    ], cwd=STAGE5_PROJECT_ROOT)


def patch_kaggle_config(stage2_checkpoint_bundle_root: Path, feature_order: list[str]):
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
    cfg["stage2_dependency"]["baseline_output_root"] = str(stage2_checkpoint_bundle_root / "outputs/stage2")
    cfg["stage2_dependency"]["selected_baseline_experiment"] = stage2_experiment_name()

    cfg["data"]["source_file"] = SOURCE_FILE

    cfg["context"]["source"] = "prebuilt"
    cfg["context"]["prebuilt_context_name"] = CONTEXT_NAME
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["feature_set_name"] = FEATURE_SET_NAME
    cfg["context"]["primary_features"] = feature_order

    cfg["runtime"]["num_workers"] = NUM_WORKERS
    cfg["runtime"]["pin_memory"] = True
    cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(feature_order)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = EXPERIMENT_SUFFIX
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = MODULATION_SCALE
    pretrained = dict(cfg["stage4_model"].get("pretrained_stage2", {}))
    pretrained["enabled"] = True
    pretrained["checkpoint_output_root"] = str(stage2_checkpoint_bundle_root)
    pretrained["freeze_visual_backbone"] = True
    pretrained["freeze_classifier"] = True
    pretrained["initialize_new_context_modules"] = True
    pretrained["strict_load"] = True
    cfg["stage4_model"]["pretrained_stage2"] = pretrained

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
    print("Context dim:", len(feature_order), flush=True)
    print("Context name:", CONTEXT_NAME, flush=True)
    print("Experiment:", stage4_experiment_name(), flush=True)


def output_check_cmd(run_seed: int):
    cmd = [
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
    if not RUN_GRADCAM:
        cmd.append("--skip-gradcam")
    return cmd


def is_completed(run_seed: int) -> bool:
    result = run(output_check_cmd(run_seed), capture=True, check=False)
    return result.returncode == 0


def read_result_row(run_seed: int, status: str, error: str = "") -> dict:
    exp = stage4_experiment_name()
    metrics_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_metrics.json"
    trading_path = PROJECT_ROOT / "outputs/stage4/metrics" / exp / f"seed_{run_seed}" / f"{EVAL_SPLIT}_trading_metrics.json"
    row = {
        "experiment_name": exp,
        "context_method": CONTEXT_METHOD,
        "feature_set_name": FEATURE_SET_NAME,
        "context_name": CONTEXT_NAME,
        "feature_source": "finbert",
        "finbert_run_id": FINBERT_RUN_ID,
        "modulation_scale": MODULATION_SCALE,
        "run_seed": run_seed,
        "status": status,
        "error": error,
        "classification_available": metrics_path.exists(),
        "trading_available": trading_path.exists(),
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in [
            "num_samples", "accuracy", "majority_class_accuracy", "roc_auc",
            "average_precision", "f1", "brier_score", "positive_rate",
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


def summarize_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    numeric = [
        "num_samples", "accuracy", "majority_class_accuracy", "roc_auc",
        "average_precision", "f1", "brier_score", "positive_rate",
        "predicted_positive_rate", "long_flat_sharpe_net",
        "long_short_sharpe_net", "long_flat_annualized_return_net",
        "long_short_annualized_return_net",
    ]
    row = {
        "experiment_name": stage4_experiment_name(),
        "feature_set_name": FEATURE_SET_NAME,
        "feature_source": "finbert",
        "finbert_run_id": FINBERT_RUN_ID,
        "modulation_scale": MODULATION_SCALE,
        "seed_count": int(df["run_seed"].nunique()),
    }
    for column in [item for item in numeric if item in df.columns]:
        values = pd.to_numeric(df[column], errors="coerce")
        row[f"{column}_mean"] = values.mean()
        row[f"{column}_std"] = values.std(ddof=1)
        row[f"{column}_count"] = int(values.notna().sum())
    return pd.DataFrame([row])


def zip_outputs():
    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "outputs/stage4/metrics",
        PROJECT_ROOT / "outputs/stage4/predictions",
        PROJECT_ROOT / "outputs/stage4/checkpoints",
        PROJECT_ROOT / "outputs/stage4/context" / CONTEXT_NAME,
        PROJECT_ROOT / "outputs/stage4/run_manifests",
    ]
    if RUN_GRADCAM:
        include_roots.append(PROJECT_ROOT / "outputs/stage4/figures")
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    print("Bundle:", BUNDLE_PATH, round(BUNDLE_PATH.stat().st_size / 1024 / 1024, 2), "MB", flush=True)


# ============================================================
# 1. Copy code/input snapshots and build prebuilt context
# ============================================================
stage4_code_input = find_stage4_code_input()
stage2_code_input = first_existing(STAGE2_CODE_INPUT_CANDIDATES)
stage5_input = first_existing(STAGE5_INPUT_CANDIDATES)

copy_or_extract_input(stage4_code_input, PROJECT_ROOT, expected_child="stage4_film_conditioning")
copy_or_extract_input(stage2_code_input, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
copy_or_extract_input(stage5_input, STAGE5_PROJECT_ROOT, expected_child="stage5_llm_news_embedding")
assert_required_code()

stage2_checkpoint_bundle_root = find_stage2_checkpoint_bundle()
print("Stage 4 code:", stage4_code_input, flush=True)
print("Stage 2 code:", stage2_code_input, flush=True)
print("Stage 5 bundle:", stage5_input, flush=True)
print("Stage 2 checkpoint bundle:", stage2_checkpoint_bundle_root, flush=True)

build_prebuilt_stage4_context()
feature_order = read_feature_order_from_context()
patch_kaggle_config(stage2_checkpoint_bundle_root, feature_order)


# ============================================================
# 2. Train/evaluate/check five seeds
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
    print("\n" + "=" * 90, flush=True)
    print(f"5-9D {stage4_experiment_name()} seed={run_seed}", flush=True)
    print("=" * 90, flush=True)

    if SKIP_COMPLETED and is_completed(run_seed):
        print(f"[skip] output check already passes for seed={run_seed}", flush=True)
        summary_rows.append(read_result_row(run_seed, status="skipped_completed"))
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
            "--experiment-suffix", EXPERIMENT_SUFFIX,
            "--context-feature-set-name", FEATURE_SET_NAME,
            "--context-features", *feature_order,
            "--modulation-scale", str(MODULATION_SCALE),
            "--stage2-pretrained-bundle-root", str(stage2_checkpoint_bundle_root),
            "--enable-stage2-pretrained",
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
        summary_rows.append(read_result_row(run_seed, status="ok"))
    except Exception as exc:
        print(f"[error] seed={run_seed}: {exc}", flush=True)
        summary_rows.append(read_result_row(run_seed, status="failed", error=str(exc)))
        if not CONTINUE_ON_ERROR:
            raise


# ============================================================
# 3. Summary and bundle
# ============================================================
tables_root = PROJECT_ROOT / "reports/tables"
tables_root.mkdir(parents=True, exist_ok=True)

seed_df = pd.DataFrame(summary_rows)
mean_std_df = summarize_results(summary_rows)

seed_csv = tables_root / f"{OUTPUT_PREFIX}_seed_results.csv"
mean_std_csv = tables_root / f"{OUTPUT_PREFIX}_mean_std_results.csv"
run_summary_json = tables_root / f"{OUTPUT_PREFIX}_run_summary.json"
manifest_json = tables_root / f"{OUTPUT_PREFIX}_manifest.json"

seed_df.to_csv(seed_csv, index=False)
mean_std_df.to_csv(mean_std_csv, index=False)
run_summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
manifest_json.write_text(
    json.dumps(
        {
            "status": "ok",
            "stage": "5-9D",
            "stage4_experiment_name": stage4_experiment_name(),
            "context_name": CONTEXT_NAME,
            "feature_set_name": FEATURE_SET_NAME,
            "feature_source": "finbert",
            "finbert_run_id": FINBERT_RUN_ID,
            "modulation_scale": MODULATION_SCALE,
            "run_seeds": RUN_SEEDS,
            "stage2_checkpoint_bundle_root": str(stage2_checkpoint_bundle_root),
            "seed_csv": str(seed_csv),
            "mean_std_csv": str(mean_std_csv),
        },
        indent=2,
    ),
    encoding="utf-8",
)

zip_outputs()

display(Markdown("# Stage 5 5-9D Seed Results"))
display(seed_df.sort_values(["run_seed"]))

display(Markdown("# Stage 5 5-9D Mean/Std Summary"))
display(mean_std_df)

print("\nDONE", flush=True)
print("Experiment:", stage4_experiment_name(), flush=True)
print("Context:", CONTEXT_NAME, flush=True)
print("Seed CSV:", seed_csv, flush=True)
print("Mean/std CSV:", mean_std_csv, flush=True)
print("Bundle:", BUNDLE_PATH, flush=True)
```
