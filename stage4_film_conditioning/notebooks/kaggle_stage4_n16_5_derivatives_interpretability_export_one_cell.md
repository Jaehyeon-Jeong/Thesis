# Kaggle Stage 4 N16-5 Derivatives Interpretability Export One Cell

This runner executes `4-N16-5`.

It does **not** train a new model. It reuses the N16-4 selected candidate:

```text
image: I60/R20/ohlc_vb
context: funding_plus_cftc_oi
model: Stage2 CNN/classifier frozen + bounded last-block FiLM
scale: 0.02
```

It exports matched interpretation artifacts:

- Stage 2 `ohlc_vb` baseline targeted Grad-CAM.
- Stage 4 `ohlc_vb + funding/CFTC` targeted Grad-CAM.
- Stage 4 gamma/beta modulation summary for the same samples.
- One downloadable bundle.

Before running, attach:

- Latest `stage4_film_conditioning` snapshot.
- Latest `stage2_btc_extension` source snapshot.
- N15-A Stage 2 checkpoint bundle containing `stage2_i60_ohlc_vb_r20`.
- N16-4 result bundle or a Stage 4 dataset that already includes N16-4 outputs.
- BTC OHLCV dataset.

```python
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

import yaml
from IPython.display import display, Markdown

# ============================================================
# User settings
# ============================================================
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect under /kaggle/input.
CODE_INPUT = ""
STAGE2_CODE_INPUT = ""
STAGE4_RESULT_BUNDLE = ""
STAGE2_CHECKPOINT_BUNDLE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
CONTEXT_FEATURE_SET = "funding_plus_cftc_oi"
CONTEXT_FEATURE_SET_NAME = "n16d_funding_plus_cftc_oi"
CONTEXT_NAME = "stage4_derivatives_context_i60_ohlc_vb_r20_n16d_funding_plus_cftc_oi"
EXPERIMENT_SUFFIX = "n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02"
STAGE4_EXPERIMENT_NAME = (
    "stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_"
    "n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02"
)
STAGE2_EXPERIMENT_NAME = "stage2_i60_ohlc_vb_r20"

RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
SAMPLES_PER_CLASS = 2
SELECTED_LIMIT_PER_PANEL = 4
TARGET_CLASS_SOURCE = "label"
OUTPUT_SUFFIX = "n16_5_derivatives_targeted_label"

SELECTED_SAMPLES_CSV = (
    PROJECT_ROOT / "reports/tables/"
    "stage4_n16_5_ohlc_vb_derivatives_interpretability_selected_for_gradcam_augmented.csv"
)
BUNDLE_PATH = Path("/kaggle/working/stage4_n16_5_derivatives_interpretability_export_bundle.zip")


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


def find_input_dir(child_name: str, required_file: str, explicit: str | Path = "") -> Path:
    if explicit:
        candidate = Path(explicit)
        return resolve_project_candidate(candidate, child_name, required_file)

    candidates = []
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            candidates.extend(root.rglob(child_name))
            candidates.extend(root.rglob(f"{child_name}*.zip"))
    for candidate in candidates:
        try:
            return resolve_project_candidate(candidate, child_name, required_file)
        except (FileNotFoundError, zipfile.BadZipFile, NotADirectoryError):
            continue
    raise FileNotFoundError(f"Could not auto-detect {child_name} with {required_file}")


def resolve_project_candidate(candidate: Path, child_name: str, required_file: str) -> Path:
    candidate = Path(candidate)
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        extract_root = Path("/kaggle/working") / f"{candidate.stem}_unzipped"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(candidate) as archive:
            archive.extractall(extract_root)
        return resolve_project_candidate(extract_root, child_name, required_file)
    if candidate.is_dir() and (candidate / required_file).exists():
        return candidate
    if candidate.is_dir():
        nested = next(
            (
                path
                for path in candidate.rglob(child_name)
                if path.is_dir() and (path / required_file).exists()
            ),
            None,
        )
        if nested is not None:
            return nested
    raise FileNotFoundError(f"{child_name} input missing {required_file}: {candidate}")


def copy_project(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {src} -> {dst}", flush=True)


def merge_tree(src: Path, dst: Path):
    if not src.exists():
        return
    for path in src.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def resolve_bundle_or_dir(explicit: str | Path, name_hint: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            candidates.extend(root.rglob(name_hint))
            candidates.extend(root.rglob(f"{name_hint}.zip"))
    for candidate in candidates:
        candidate = Path(candidate)
        try:
            if candidate.is_file() and candidate.suffix.lower() == ".zip":
                extract_root = Path("/kaggle/working") / f"{candidate.stem}_extracted"
                if not extract_root.exists():
                    extract_root.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(candidate) as archive:
                        archive.extractall(extract_root)
                return extract_root
            if candidate.is_dir():
                return candidate
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError(f"Could not find bundle/dir matching {name_hint}")


def has_stage2_checkpoint(root: Path, seed: int) -> bool:
    return (
        root / "outputs/stage2/checkpoints" / STAGE2_EXPERIMENT_NAME / f"seed_{seed}" / "best.pt"
    ).exists()


def resolve_stage2_checkpoint_bundle() -> Path:
    candidates = []
    if STAGE2_CHECKPOINT_BUNDLE:
        candidates.append(Path(STAGE2_CHECKPOINT_BUNDLE))
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            candidates.extend(root.rglob("stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"))
            candidates.extend(root.rglob("stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"))

    for candidate in candidates:
        candidate = Path(candidate)
        try:
            root = resolve_bundle_or_dir(candidate, candidate.name)
            nested = next(
                (
                    path.parents[5]
                    for path in root.rglob(f"outputs/stage2/checkpoints/{STAGE2_EXPERIMENT_NAME}/seed_42/best.pt")
                ),
                None,
            )
            resolved = nested if nested is not None else root
            if all(has_stage2_checkpoint(resolved, seed) for seed in RUN_SEEDS):
                print("Using Stage 2 checkpoint bundle:", resolved, flush=True)
                return resolved
        except (FileNotFoundError, IndexError):
            continue
    raise FileNotFoundError("Could not find N15-A Stage 2 checkpoint bundle with ohlc_vb.")


def patch_stage2_config():
    config_path = STAGE2_PROJECT_ROOT / "configs/env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(STAGE2_PROJECT_ROOT)
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = ""
    cfg["paths"]["output_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2")
    cfg["paths"]["checkpoint_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/checkpoints")
    cfg["paths"]["metrics_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/metrics")
    cfg["paths"]["predictions_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/predictions")
    cfg["paths"]["figures_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/figures")
    cfg["paths"]["reports_root"] = str(STAGE2_PROJECT_ROOT / "reports")
    cfg["paths"]["tables_root"] = str(STAGE2_PROJECT_ROOT / "reports/tables")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def patch_stage4_config(stage2_checkpoint_root: Path):
    config_path = PROJECT_ROOT / "configs/env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    scaler_path = (
        PROJECT_ROOT / "outputs/stage4/context" / CONTEXT_NAME / "seed_42" / "context_scaler.json"
    )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Context scaler missing: {scaler_path}")
    feature_order = json.loads(scaler_path.read_text(encoding="utf-8"))["feature_order"]

    cfg["paths"]["project_root"] = str(PROJECT_ROOT)
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = ""
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
    cfg["stage2_dependency"]["baseline_output_root"] = str(stage2_checkpoint_root / "outputs/stage2")

    cfg["context"]["source"] = "prebuilt"
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["feature_set_name"] = CONTEXT_FEATURE_SET_NAME
    cfg["context"]["prebuilt_context_name"] = CONTEXT_NAME
    cfg["context"]["primary_features"] = list(feature_order)

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(feature_order)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = EXPERIMENT_SUFFIX
    cfg["stage4_model"].setdefault("film_full_bounded_last_block", {})
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = 0.02
    cfg["stage4_model"]["pretrained_stage2"] = {
        "enabled": True,
        "checkpoint_output_root": str(stage2_checkpoint_root),
        "freeze_visual_backbone": True,
        "freeze_classifier": True,
        "initialize_new_context_modules": True,
        "strict_load": True,
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def assert_ready():
    required = [
        PROJECT_ROOT / "scripts/generate_stage4_gradcam_context.py",
        STAGE2_PROJECT_ROOT / "scripts/generate_stage2_gradcam.py",
        SELECTED_SAMPLES_CSV,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    for seed in RUN_SEEDS:
        stage4_ckpt = (
            PROJECT_ROOT / "outputs/stage4/checkpoints" / STAGE4_EXPERIMENT_NAME / f"seed_{seed}" / "best.pt"
        )
        stage4_pred = (
            PROJECT_ROOT / "outputs/stage4/predictions" / STAGE4_EXPERIMENT_NAME / f"seed_{seed}" / f"{EVAL_SPLIT}_predictions.csv"
        )
        if not stage4_ckpt.exists() or not stage4_pred.exists():
            raise FileNotFoundError(f"Missing N16-4 Stage4 output for seed {seed}: {stage4_ckpt}")


def export_seed(seed: int):
    run([
        sys.executable, "-u",
        "scripts/generate_stage2_gradcam.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--run-seed", str(seed),
        "--split", EVAL_SPLIT,
        "--samples-per-class", str(SAMPLES_PER_CLASS),
        "--selected-samples-csv", str(SELECTED_SAMPLES_CSV),
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
        "--samples-per-class", str(SAMPLES_PER_CLASS),
        "--selected-samples-csv", str(SELECTED_SAMPLES_CSV),
        "--target-class-source", TARGET_CLASS_SOURCE,
        "--selected-limit-per-panel", str(SELECTED_LIMIT_PER_PANEL),
        "--output-suffix", OUTPUT_SUFFIX,
        "--write-report-copy",
    ], cwd=PROJECT_ROOT)


def make_bundle():
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "reports/figures/gradcam",
        PROJECT_ROOT / "outputs/stage4/figures" / STAGE4_EXPERIMENT_NAME,
        STAGE2_PROJECT_ROOT / "reports/figures/gradcam",
        STAGE2_PROJECT_ROOT / "outputs/stage2/figures" / STAGE2_EXPERIMENT_NAME,
    ]
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if root.exists():
                base = root.parents[1] if root.name == "gradcam" else root.parents[0]
                for path in root.rglob("*"):
                    if path.is_file():
                        zf.write(path, arcname=str(path.relative_to(base)))
    print("Bundle:", BUNDLE_PATH, round(BUNDLE_PATH.stat().st_size / 1024 / 1024, 2), "MB")


# ============================================================
# 1. Copy code snapshots
# ============================================================
stage4_src = find_input_dir("stage4_film_conditioning", "scripts/generate_stage4_gradcam_context.py", CODE_INPUT)
stage2_src = find_input_dir("stage2_btc_extension", "scripts/generate_stage2_gradcam.py", STAGE2_CODE_INPUT)
copy_project(stage4_src, PROJECT_ROOT)
copy_project(stage2_src, STAGE2_PROJECT_ROOT)

# ============================================================
# 2. Attach/copy previous result bundle and Stage2 checkpoints
# ============================================================
try:
    result_root = resolve_bundle_or_dir(STAGE4_RESULT_BUNDLE, "stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle")
    merge_tree(result_root / "outputs", PROJECT_ROOT / "outputs")
    merge_tree(result_root / "reports", PROJECT_ROOT / "reports")
    print("Merged N16-4 result bundle:", result_root, flush=True)
except FileNotFoundError:
    print("No explicit N16-4 bundle found; using files already present in Stage4 snapshot/working dir.", flush=True)

stage2_checkpoint_root = resolve_stage2_checkpoint_bundle()
merge_tree(stage2_checkpoint_root / "outputs/stage2", STAGE2_PROJECT_ROOT / "outputs/stage2")
merge_tree(stage2_checkpoint_root / "reports", STAGE2_PROJECT_ROOT / "reports")

# ============================================================
# 3. Patch configs and export
# ============================================================
patch_stage2_config()
patch_stage4_config(stage2_checkpoint_root)
assert_ready()

for seed in RUN_SEEDS:
    print("\n" + "=" * 90, flush=True)
    print(f"N16-5 targeted export seed {seed}", flush=True)
    print("=" * 90, flush=True)
    export_seed(seed)

make_bundle()

display(Markdown("# N16-5 export complete"))
display(Markdown(f"Bundle: `{BUNDLE_PATH}`"))
```

