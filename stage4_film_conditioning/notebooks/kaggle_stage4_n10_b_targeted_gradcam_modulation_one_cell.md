# Kaggle Stage 4 N10-B Targeted Grad-CAM and Modulation Export One Cell

Run this cell **after N10-A correction analysis** and after the selected N10
model outputs/checkpoints are available in `/kaggle/working`.

This cell does not train a model. It exports targeted interpretation artifacts
for the exact sample indices selected by N10-A:

- Stage 2 baseline Grad-CAM for `Stage2 wrong -> N10 correct` and regression
  samples.
- N10 Grad-CAM for the same samples.
- N10 context embedding plus FiLM gamma/beta modulation summaries and full JSON.
- A compact downloadable zip bundle.

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
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")

# Leave empty to auto-detect under /kaggle/input or /kaggle/working.
CODE_INPUT = ""
STAGE2_CODE_INPUT = ""
STAGE2_CHECKPOINT_BUNDLE = ""
STAGE2_BUNDLE_DIR_NAME = "stage2_i60_ohlc_ma_vb_r20_seed42_46_checkpoints_for_stage4_n8"

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
STAGE4_EXPERIMENT_SUFFIX = "news_tfidf_svd32_w7_20_60_pretrained_frozen_s0p02"
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
NEWS_FEATURE_SET_NAME = "tfidf_svd32_w7_20_60"
NEWS_CONTEXT_NAME = (
    f"stage4_news_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
    f"r{RETURN_HORIZON}_{NEWS_FEATURE_SET_NAME}"
)
MODULATION_SCALE = 0.02
FREEZE_CLASSIFIER = True

# Targeted mode: target true label for both Stage 2 and N10.
TARGET_CLASS_SOURCE = "label"

# 0 keeps all rows from N10-A. 6 keeps figures readable.
SELECTED_LIMIT_PER_PANEL = 6

N10A_PREFIX = "stage4_n10_stage2_vs_n10_correction_analysis"
OUTPUT_SUFFIX = "n10b_targeted_label"
BUNDLE_PATH = Path("/kaggle/working/stage4_n10b_targeted_gradcam_modulation_bundle.zip")


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


def find_input_dir(child_name: str, required_file: str) -> Path:
    """Find a Kaggle input/working directory containing a required file."""

    explicit = CODE_INPUT if child_name == "stage4_film_conditioning" else STAGE2_CODE_INPUT
    if explicit:
        candidate = Path(explicit)
        if (candidate / required_file).exists():
            return candidate
        nested = next(candidate.rglob(child_name), None) if candidate.exists() else None
        if nested is not None and (nested / required_file).exists():
            return nested
        raise FileNotFoundError(f"{child_name} input missing {required_file}: {candidate}")

    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not root.exists():
            continue
        for candidate in root.rglob(child_name):
            if candidate.is_dir() and (candidate / required_file).exists():
                return candidate
    raise FileNotFoundError(f"Could not auto-detect {child_name} with {required_file}")


def sync_project_code(src: Path, dst: Path):
    """Copy code/data files without deleting existing outputs/checkpoints."""

    if src.resolve() == dst.resolve():
        print(f"Sync skipped; source already equals destination: {dst}", flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    copy_names = [
        "configs",
        "docs",
        "notebooks",
        "scripts",
        "src",
        "FG_data",
        "news_data",
        "README.md",
        "checklist.md",
        "workflow_diagram.md",
        STAGE2_BUNDLE_DIR_NAME,
    ]
    for name in copy_names:
        source = src / name
        if not source.exists():
            continue
        target = dst / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    print(f"Synced code/data: {src} -> {dst}", flush=True)


def prepare_working_snapshots():
    """Ensure working dirs use latest uploaded code snapshots."""

    stage4_input = find_input_dir(
        "stage4_film_conditioning",
        "scripts/generate_stage4_gradcam_context.py",
    )
    sync_project_code(stage4_input, PROJECT_ROOT)

    stage2_input = find_input_dir(
        "stage2_btc_extension",
        "scripts/generate_stage2_gradcam.py",
    )
    sync_project_code(stage2_input, STAGE2_PROJECT_ROOT)


def patch_stage4_config():
    """Patch Stage 4 Kaggle paths and N10 selected-model settings."""

    config_path = PROJECT_ROOT / "configs/env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scaler_path = (
        PROJECT_ROOT
        / "outputs/stage4/context"
        / NEWS_CONTEXT_NAME
        / f"seed_{RUN_SEEDS[0]}"
        / "context_scaler.json"
    )
    if not scaler_path.exists():
        fallback = next(
            (PROJECT_ROOT / "outputs/stage4/context").rglob(
                f"{NEWS_CONTEXT_NAME}/seed_*/context_scaler.json"
            ),
            None,
        )
        if fallback is not None:
            scaler_path = fallback
    if not scaler_path.exists():
        raise FileNotFoundError(
            "N10 context scaler missing. Run the N10 selected news "
            f"interpretability cell first. Expected: {scaler_path}"
        )
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_order = list(scaler.get("feature_order", []))
    if not feature_order:
        raise ValueError(f"context_scaler.json has empty feature_order: {scaler_path}")
    normalized = [f"{feature}_normalized" for feature in feature_order]

    cfg["paths"]["project_root"] = str(PROJECT_ROOT)
    cfg["paths"]["data_root"] = "/kaggle/input"
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
    cfg["context"]["source"] = "prebuilt_news"
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["feature_set_name"] = NEWS_FEATURE_SET_NAME
    cfg["context"]["prebuilt_context_name"] = NEWS_CONTEXT_NAME
    cfg["context"]["primary_features"] = feature_order
    cfg["context"]["normalized_feature_columns"] = normalized
    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(feature_order)
    cfg["stage4_model"]["experiment_suffix"] = STAGE4_EXPERIMENT_SUFFIX
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"].setdefault("film_full_bounded_last_block", {})
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = MODULATION_SCALE
    cfg["stage4_model"]["pretrained_stage2"] = {
        "enabled": True,
        "checkpoint_output_root": str(STAGE2_PROJECT_ROOT),
        "freeze_visual_backbone": True,
        "freeze_classifier": FREEZE_CLASSIFIER,
        "initialize_new_context_modules": True,
        "strict_load": True,
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        "Config patched:",
        config_path,
        "experiment_suffix=",
        STAGE4_EXPERIMENT_SUFFIX,
        "context=",
        NEWS_CONTEXT_NAME,
        "context_dim=",
        len(feature_order),
        flush=True,
    )


def stage2_experiment_name():
    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def stage4_experiment_name():
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{STAGE4_EXPERIMENT_SUFFIX}"
    )


def selected_csv_path() -> Path:
    return PROJECT_ROOT / "reports/tables" / f"{N10A_PREFIX}_selected_for_gradcam.csv"


def assert_required_files():
    required_stage4 = [
        "scripts/generate_stage4_gradcam_context.py",
        "src/stage4_film/interpretability/gradcam_context.py",
    ]
    missing_stage4 = [path for path in required_stage4 if not (PROJECT_ROOT / path).exists()]
    if missing_stage4:
        raise RuntimeError("Stage 4 snapshot is missing: " + ", ".join(missing_stage4))

    required_stage2 = [
        "scripts/generate_stage2_gradcam.py",
        "src/stage2_btc/interpretability/gradcam.py",
    ]
    missing_stage2 = [path for path in required_stage2 if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing_stage2:
        raise RuntimeError("Stage 2 snapshot is missing: " + ", ".join(missing_stage2))

    if not selected_csv_path().exists():
        raise FileNotFoundError(
            "N10-A selected sample CSV missing. Run "
            "kaggle_stage4_n10_stage2_vs_n10_correction_analysis_one_cell.md first: "
            f"{selected_csv_path()}"
        )


def resolve_checkpoint_candidate(candidate: Path) -> Path:
    """Return a directory containing Stage 2 outputs/checkpoints."""

    exp = stage2_experiment_name()
    candidate = candidate.expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        extract_root = Path("/kaggle/working/stage2_checkpoint_bundle_extracted") / candidate.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(extract_root)
        return resolve_checkpoint_candidate(extract_root)

    if not candidate.is_dir():
        raise FileNotFoundError(candidate)

    direct = candidate / "outputs/stage2/checkpoints" / exp / "seed_42/best.pt"
    if direct.exists():
        return candidate

    nested = next(candidate.rglob(f"outputs/stage2/checkpoints/{exp}/seed_42/best.pt"), None)
    if nested is not None:
        return nested.parents[5]

    checkpoint_root = candidate / "checkpoints" / exp / "seed_42/best.pt"
    if checkpoint_root.exists():
        return candidate

    raise FileNotFoundError(f"No Stage 2 checkpoint found under {candidate}")


def find_stage2_checkpoint_bundle() -> Path:
    if STAGE2_CHECKPOINT_BUNDLE:
        return resolve_checkpoint_candidate(Path(STAGE2_CHECKPOINT_BUNDLE))

    exp = stage2_experiment_name()
    existing = STAGE2_PROJECT_ROOT / "outputs/stage2/checkpoints" / exp / "seed_42/best.pt"
    if existing.exists():
        return STAGE2_PROJECT_ROOT

    candidates = [
        PROJECT_ROOT / STAGE2_BUNDLE_DIR_NAME,
        STAGE2_PROJECT_ROOT / STAGE2_BUNDLE_DIR_NAME,
        Path("/kaggle/working") / STAGE2_BUNDLE_DIR_NAME,
    ]
    candidates.extend(Path("/kaggle/input").rglob(STAGE2_BUNDLE_DIR_NAME))
    candidates.extend(Path("/kaggle/working").glob(f"{STAGE2_BUNDLE_DIR_NAME}*.zip"))
    candidates.extend(Path("/kaggle/input").rglob(f"{STAGE2_BUNDLE_DIR_NAME}*.zip"))
    for candidate in candidates:
        try:
            resolved = resolve_checkpoint_candidate(candidate)
        except Exception:
            continue
        print("Stage 2 checkpoint bundle:", resolved, flush=True)
        return resolved
    raise FileNotFoundError("Could not find Stage 2 checkpoint bundle.")


def ensure_stage2_outputs():
    """Ensure Stage 2 selected-baseline checkpoints/predictions exist."""

    exp = stage2_experiment_name()
    missing_checkpoint = [
        seed
        for seed in RUN_SEEDS
        if not (
            STAGE2_PROJECT_ROOT
            / "outputs/stage2/checkpoints"
            / exp
            / f"seed_{seed}/best.pt"
        ).exists()
    ]
    if missing_checkpoint:
        bundle_root = find_stage2_checkpoint_bundle()
        src_outputs = bundle_root / "outputs/stage2"
        if src_outputs.exists():
            shutil.copytree(src_outputs, STAGE2_PROJECT_ROOT / "outputs/stage2", dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"Stage 2 bundle does not contain outputs/stage2: {bundle_root}")

    for seed in RUN_SEEDS:
        prediction = (
            STAGE2_PROJECT_ROOT
            / "outputs/stage2/predictions"
            / exp
            / f"seed_{seed}"
            / f"{EVAL_SPLIT}_predictions.csv"
        )
        if prediction.exists() and prediction.stat().st_size > 0:
            continue
        run([
            sys.executable, "-u",
            "scripts/evaluate_stage2_predictions.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(seed),
            "--split", EVAL_SPLIT,
        ], cwd=STAGE2_PROJECT_ROOT)


def ensure_stage4_predictions():
    """Generate N10 predictions if checkpoint exists but CSV is missing."""

    exp = stage4_experiment_name()
    for seed in RUN_SEEDS:
        prediction = (
            PROJECT_ROOT
            / "outputs/stage4/predictions"
            / exp
            / f"seed_{seed}"
            / f"{EVAL_SPLIT}_predictions.csv"
        )
        if prediction.exists() and prediction.stat().st_size > 0:
            continue
        checkpoint = (
            PROJECT_ROOT
            / "outputs/stage4/checkpoints"
            / exp
            / f"seed_{seed}/best.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"N10 checkpoint missing for seed {seed}. Run the N10 selected "
                f"interpretability cell first. Expected: {checkpoint}"
            )
        run([
            sys.executable, "-u",
            "scripts/evaluate_stage4_predictions.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--context-method", CONTEXT_METHOD,
            "--run-seed", str(seed),
            "--split", EVAL_SPLIT,
        ])


def run_targeted_exports():
    """Run Stage 2 and N10 targeted Grad-CAM exports for each seed."""

    selection = selected_csv_path()
    for seed in RUN_SEEDS:
        print("\n" + "=" * 90, flush=True)
        print(f"N10-B targeted Grad-CAM seed {seed}", flush=True)
        print("=" * 90, flush=True)

        run([
            sys.executable, "-u",
            "scripts/generate_stage2_gradcam.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(seed),
            "--split", EVAL_SPLIT,
            "--samples-per-class", "2",
            "--selected-samples-csv", str(selection),
            "--target-class-source", TARGET_CLASS_SOURCE,
            "--selected-limit-per-panel", str(SELECTED_LIMIT_PER_PANEL),
            "--output-suffix", OUTPUT_SUFFIX,
            "--write-report-copy",
        ], cwd=STAGE2_PROJECT_ROOT)

        run([
            sys.executable, "-u",
            "scripts/generate_stage4_gradcam_context.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--context-method", CONTEXT_METHOD,
            "--run-seed", str(seed),
            "--split", EVAL_SPLIT,
            "--samples-per-class", "2",
            "--selected-samples-csv", str(selection),
            "--target-class-source", TARGET_CLASS_SOURCE,
            "--selected-limit-per-panel", str(SELECTED_LIMIT_PER_PANEL),
            "--output-suffix", OUTPUT_SUFFIX,
            "--write-report-copy",
        ], cwd=PROJECT_ROOT)


def zip_targeted_outputs():
    """Create a compact downloadable bundle."""

    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    manifest = {"files": []}
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((PROJECT_ROOT / "reports/tables").glob(f"{N10A_PREFIX}*")):
            arcname = Path("stage4/reports/tables") / path.name
            archive.write(path, arcname)
            manifest["files"].append(str(arcname))

        stage4_figures = sorted((PROJECT_ROOT / "reports/figures/gradcam").glob(f"*{OUTPUT_SUFFIX}*"))
        for path in stage4_figures:
            arcname = Path("stage4/reports/figures/gradcam") / path.name
            archive.write(path, arcname)
            manifest["files"].append(str(arcname))

        stage2_figures = sorted((STAGE2_PROJECT_ROOT / "reports/figures/gradcam").glob(f"*{OUTPUT_SUFFIX}*"))
        for path in stage2_figures:
            arcname = Path("stage2/reports/figures/gradcam") / path.name
            archive.write(path, arcname)
            manifest["files"].append(str(arcname))

        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print("Bundle:", BUNDLE_PATH, flush=True)


prepare_working_snapshots()
patch_stage4_config()
assert_required_files()
ensure_stage2_outputs()
ensure_stage4_predictions()
run_targeted_exports()
zip_targeted_outputs()

selection_df = pd.read_csv(selected_csv_path())
display(Markdown("# N10-B Selected Samples"))
display(selection_df[["run_seed", "sample_index", "Date", "label", "transition", "prob_up_stage2", "prob_up_stage4", "news_count_7d", "news_count_20d", "news_count_60d"]].head(30))

display(Markdown("# N10-B Download Bundle"))
print(BUNDLE_PATH)
print("DONE", flush=True)
```
