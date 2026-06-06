# Kaggle Stage 4 N13-6 Interpretability Export One Cell

This runner executes `4-N13-6`.

It does **not** train a new model. It reuses already trained Stage 2 frozen
context-FiLM candidates and exports matched interpretation artifacts:

- Stage 2 baseline Grad-CAM.
- Context-FiLM Grad-CAM for the same sample indices.
- Context values, probability changes, and gamma/beta modulation summaries.
- Correction/regression tables and extreme-context sample tables.
- One downloadable bundle.

Main candidates:

```text
N8-B  : F&G-only, bounded last-block FiLM, scale 0.02
N10   : news TF-IDF/SVD32, bounded last-block FiLM, scale 0.02
Stage2: frozen visual baseline, I60/R20/ohlc_ma_vb
```

Run this after the N8-B and N10 candidate outputs/checkpoints are available in
the attached Stage 4 dataset or in `/kaggle/working`.

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
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
TOP_K = 30

# Targeted mode: use true label as the Grad-CAM class for both Stage 2 and Stage 4.
TARGET_CLASS_SOURCE = "label"

# 0 keeps all selected rows. 4 to 6 keeps figures readable.
SELECTED_LIMIT_PER_PANEL = 4
EXTREME_SAMPLES_PER_SEED = 2

RUN_CORRECTION_ANALYSIS = True
RUN_TARGETED_GRADCAM = True
FREEZE_CLASSIFIER = True
MIN_PREDICTIONS = 1000
BUNDLE_PATH = Path("/kaggle/working/stage4_n13_6_interpretability_export_bundle.zip")

CANDIDATES = [
    {
        "key": "fg",
        "analysis_name": "N8-B F&G-only bounded FiLM s0.02",
        "experiment_suffix": "n8b_fg_pretrained_frozen_s0p02",
        "context_source": "structured",
        "feature_set_name": "fg_only",
        "context_name": f"stage4_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_fg_only",
        "context_features": ["fg_value", "fg_mean_60", "fg_delta_60", "fg_std_60"],
        "modulation_scale": 0.02,
        "output_prefix": "stage4_n13_6_fg_stage2_vs_film_correction_analysis",
        "output_suffix": "n13_6_fg_targeted_label",
        "extreme_specs": [
            {
                "panel": "fg_extreme_fear",
                "label": "F&G extreme fear",
                "column": "fg_mean_60_normalized",
                "ascending": True,
            },
            {
                "panel": "fg_extreme_greed",
                "label": "F&G extreme greed",
                "column": "fg_mean_60_normalized",
                "ascending": False,
            },
        ],
    },
    {
        "key": "news",
        "analysis_name": "N10 news SVD32 bounded FiLM s0.02",
        "experiment_suffix": "news_tfidf_svd32_w7_20_60_pretrained_frozen_s0p02",
        "context_source": "prebuilt_news",
        "feature_set_name": "tfidf_svd32_w7_20_60",
        "context_name": f"stage4_news_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}_tfidf_svd32_w7_20_60",
        "context_features": None,  # loaded from context_scaler.json
        "modulation_scale": 0.02,
        "output_prefix": "stage4_n13_6_news_stage2_vs_film_correction_analysis",
        "output_suffix": "n13_6_news_targeted_label",
        "extreme_specs": [
            {
                "panel": "news_high_count_60d",
                "label": "High 60d news count",
                "column": "news_count_60d",
                "ascending": False,
            },
            {
                "panel": "news_svd60_09_high",
                "label": "News SVD60-09 high",
                "column": "news_svd_60d_09_normalized",
                "ascending": False,
            },
            {
                "panel": "news_svd60_09_low",
                "label": "News SVD60-09 low",
                "column": "news_svd_60d_09_normalized",
                "ascending": True,
            },
        ],
    },
]

# Use a subset if needed, for example ["fg"].
RUN_CANDIDATE_KEYS = ["fg", "news"]


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


def find_input_dir(child_name: str, required_file: str, explicit: str | Path = "") -> Path:
    """Find a Kaggle input/working directory containing a required file."""

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


def sync_project_snapshot(src: Path, dst: Path, *, include_outputs: bool):
    """Copy code/data, and optionally existing outputs, without deleting working state."""

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
        "data_inventory",
        "README.md",
        "checklist.md",
        "workflow_diagram.md",
        STAGE2_BUNDLE_DIR_NAME,
    ]
    if include_outputs:
        copy_names += ["outputs", "reports"]
    for name in copy_names:
        source = src / name
        if not source.exists():
            continue
        target = dst / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    print(f"Synced snapshot: {src} -> {dst}", flush=True)


def prepare_working_snapshots():
    """Ensure working directories use the latest uploaded snapshots."""

    stage4_input = find_input_dir(
        "stage4_film_conditioning",
        "scripts/generate_stage4_gradcam_context.py",
        explicit=CODE_INPUT,
    )
    sync_project_snapshot(stage4_input, PROJECT_ROOT, include_outputs=True)

    stage2_input = find_input_dir(
        "stage2_btc_extension",
        "scripts/generate_stage2_gradcam.py",
        explicit=STAGE2_CODE_INPUT,
    )
    sync_project_snapshot(stage2_input, STAGE2_PROJECT_ROOT, include_outputs=True)


def stage2_experiment_name() -> str:
    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def stage4_experiment_name(candidate: dict) -> str:
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{candidate['experiment_suffix']}"
    )


def assert_required_files():
    required_stage4 = [
        "scripts/analyze_stage4_stage2_context_corrections.py",
        "scripts/evaluate_stage4_predictions.py",
        "scripts/generate_stage4_gradcam_context.py",
        "src/stage4_film/interpretability/gradcam_context.py",
    ]
    missing_stage4 = [path for path in required_stage4 if not (PROJECT_ROOT / path).exists()]
    if missing_stage4:
        raise RuntimeError("Stage 4 snapshot is missing: " + ", ".join(missing_stage4))

    required_stage2 = [
        "scripts/evaluate_stage2_predictions.py",
        "scripts/generate_stage2_gradcam.py",
        "src/stage2_btc/interpretability/gradcam.py",
    ]
    missing_stage2 = [path for path in required_stage2 if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing_stage2:
        raise RuntimeError("Stage 2 snapshot is missing: " + ", ".join(missing_stage2))


def patch_stage2_config():
    """Patch Stage 2 Kaggle paths."""

    config_path = STAGE2_PROJECT_ROOT / "configs/env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(STAGE2_PROJECT_ROOT)
    cfg["paths"]["data_root"] = "/kaggle/input"
    cfg["paths"]["output_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2")
    cfg["paths"]["checkpoint_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/checkpoints")
    cfg["paths"]["metrics_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/metrics")
    cfg["paths"]["predictions_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/predictions")
    cfg["paths"]["figures_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2/figures")
    cfg["paths"]["reports_root"] = str(STAGE2_PROJECT_ROOT / "reports")
    cfg["paths"]["tables_root"] = str(STAGE2_PROJECT_ROOT / "reports/tables")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def load_feature_order(candidate: dict) -> list[str]:
    """Return candidate context feature order."""

    if candidate.get("context_features"):
        return list(candidate["context_features"])

    context_root = PROJECT_ROOT / "outputs/stage4/context" / candidate["context_name"]
    for seed in RUN_SEEDS:
        scaler_path = context_root / f"seed_{seed}" / "context_scaler.json"
        if scaler_path.exists():
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
            feature_order = [str(feature) for feature in scaler.get("feature_order", [])]
            if feature_order:
                return feature_order
    fallback = next(context_root.rglob("context_scaler.json"), None) if context_root.exists() else None
    if fallback is not None:
        scaler = json.loads(fallback.read_text(encoding="utf-8"))
        feature_order = [str(feature) for feature in scaler.get("feature_order", [])]
        if feature_order:
            return feature_order
    raise FileNotFoundError(f"Could not load context feature order for {candidate['key']}: {context_root}")


def patch_stage4_config(candidate: dict):
    """Patch Stage 4 config for one candidate."""

    feature_order = load_feature_order(candidate)
    normalized = [f"{feature}_normalized" for feature in feature_order]

    config_path = PROJECT_ROOT / "configs/env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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

    cfg["context"]["source"] = candidate["context_source"]
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["feature_set_name"] = candidate["feature_set_name"]
    cfg["context"]["primary_features"] = feature_order
    cfg["context"]["normalized_feature_columns"] = normalized
    if candidate["context_source"] in {"prebuilt", "prebuilt_news", "news_prebuilt"}:
        cfg["context"]["prebuilt_context_name"] = candidate["context_name"]
    else:
        cfg["context"].pop("prebuilt_context_name", None)

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(feature_order)
    cfg["stage4_model"]["experiment_suffix"] = candidate["experiment_suffix"]
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"].setdefault(CONTEXT_METHOD, {})
    cfg["stage4_model"][CONTEXT_METHOD]["modulation_scale"] = float(candidate["modulation_scale"])
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
        "Stage4 config patched:",
        candidate["key"],
        "experiment_suffix=",
        candidate["experiment_suffix"],
        "context_dim=",
        len(feature_order),
        flush=True,
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


def ensure_stage4_predictions(candidate: dict):
    """Generate candidate predictions if checkpoint exists but CSV is missing."""

    patch_stage4_config(candidate)
    exp = stage4_experiment_name(candidate)
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
                f"Candidate checkpoint missing for {candidate['key']} seed {seed}. "
                f"Run or attach the candidate result bundle first. Expected: {checkpoint}"
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


def candidate_by_key(key: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["key"] == key:
            return candidate
    raise KeyError(key)


def analysis_selected_csv(candidate: dict, *, augmented: bool) -> Path:
    suffix = "_selected_for_gradcam_augmented.csv" if augmented else "_selected_for_gradcam.csv"
    return PROJECT_ROOT / "reports/tables" / f"{candidate['output_prefix']}{suffix}"


def run_correction_analysis(candidate: dict):
    """Run Stage2-vs-candidate correction/regression analysis."""

    if not RUN_CORRECTION_ANALYSIS:
        return
    run([
        sys.executable, "-u",
        "scripts/analyze_stage4_stage2_context_corrections.py",
        "--config", "configs/env_kaggle.yaml",
        "--image-window", str(IMAGE_WINDOW),
        "--image-spec", IMAGE_SPEC,
        "--return-horizon", str(RETURN_HORIZON),
        "--context-window", str(CONTEXT_WINDOW),
        "--context-method", CONTEXT_METHOD,
        "--stage4-experiment-suffix", candidate["experiment_suffix"],
        "--context-name", candidate["context_name"],
        "--stage2-output-root", str(STAGE2_PROJECT_ROOT / "outputs/stage2"),
        "--stage4-output-root", str(PROJECT_ROOT / "outputs/stage4"),
        "--run-seeds", *map(str, RUN_SEEDS),
        "--split", EVAL_SPLIT,
        "--top-k", str(TOP_K),
        "--output-prefix", candidate["output_prefix"],
        "--analysis-name", candidate["analysis_name"],
    ])
    augment_selected_samples(candidate)


def augment_selected_samples(candidate: dict):
    """Add extreme-context samples to the correction/regression selected table."""

    selected_path = analysis_selected_csv(candidate, augmented=False)
    all_path = PROJECT_ROOT / "reports/tables" / f"{candidate['output_prefix']}_all_transitions.csv"
    if not selected_path.exists() or not all_path.exists():
        raise FileNotFoundError(f"Correction analysis output missing for {candidate['key']}")

    selected = pd.read_csv(selected_path)
    all_rows = pd.read_csv(all_path)
    extras = []
    for spec in candidate.get("extreme_specs", []):
        column = str(spec["column"])
        if column not in all_rows.columns:
            print(f"[warning] extreme column missing for {candidate['key']}: {column}", flush=True)
            continue
        frame = all_rows.loc[all_rows[column].notna()].copy()
        if frame.empty:
            continue
        frame["extreme_sort_value"] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[frame["extreme_sort_value"].notna()].copy()
        if frame.empty:
            continue
        for seed, seed_frame in frame.groupby("run_seed"):
            rows = (
                seed_frame.sort_values("extreme_sort_value", ascending=bool(spec["ascending"]))
                .head(int(EXTREME_SAMPLES_PER_SEED))
                .copy()
            )
            rows["analysis_group"] = str(spec["panel"])
            rows["transition"] = str(spec["panel"])
            rows["selection_score"] = rows["extreme_sort_value"]
            rows["gradcam_panel"] = str(spec["panel"])
            rows["gradcam_panel_label"] = str(spec["label"])
            extras.append(rows)

    if extras:
        augmented = pd.concat([selected, *extras], ignore_index=True, sort=False)
    else:
        augmented = selected.copy()
    augmented = augmented.drop_duplicates(
        subset=["run_seed", "sample_index", "analysis_group"],
        keep="first",
    ).reset_index(drop=True)
    output_path = analysis_selected_csv(candidate, augmented=True)
    augmented.to_csv(output_path, index=False)
    print(
        "Selected samples:",
        candidate["key"],
        "base=",
        len(selected),
        "augmented=",
        len(augmented),
        output_path,
        flush=True,
    )


def run_targeted_exports(candidate: dict):
    """Run Stage 2 and candidate targeted Grad-CAM exports for each seed."""

    if not RUN_TARGETED_GRADCAM:
        return
    patch_stage4_config(candidate)
    selection = analysis_selected_csv(candidate, augmented=True)
    if not selection.exists():
        selection = analysis_selected_csv(candidate, augmented=False)
    if not selection.exists():
        raise FileNotFoundError(f"Selected sample CSV missing: {selection}")

    for seed in RUN_SEEDS:
        print("\n" + "=" * 90, flush=True)
        print(f"N13-6 targeted Grad-CAM {candidate['key']} seed {seed}", flush=True)
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
            "--output-suffix", candidate["output_suffix"],
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
            "--output-suffix", candidate["output_suffix"],
            "--write-report-copy",
        ], cwd=PROJECT_ROOT)


def build_compact_summary(candidates: list[dict]) -> pd.DataFrame:
    """Build one compact N13-6 summary table."""

    rows = []
    for candidate in candidates:
        seed_path = PROJECT_ROOT / "reports/tables" / f"{candidate['output_prefix']}_seed_summary.csv"
        transition_path = PROJECT_ROOT / "reports/tables" / f"{candidate['output_prefix']}_transition_summary.csv"
        if not seed_path.exists():
            continue
        seed_df = pd.read_csv(seed_path)
        transition_df = pd.read_csv(transition_path) if transition_path.exists() else pd.DataFrame()
        row = {
            "candidate_key": candidate["key"],
            "analysis_name": candidate["analysis_name"],
            "stage4_experiment": stage4_experiment_name(candidate),
            "context_name": candidate["context_name"],
            "num_seeds": int(seed_df["run_seed"].nunique()),
            "stage2_accuracy_mean": float(seed_df["stage2_accuracy"].mean()),
            "stage4_accuracy_mean": float(seed_df["stage4_accuracy"].mean()),
            "accuracy_delta_mean": float(seed_df["accuracy_delta_stage4_minus_stage2"].mean()),
            "stage2_wrong_to_stage4_correct_total": int(seed_df["stage2_wrong_to_stage4_correct"].sum()),
            "stage2_correct_to_stage4_wrong_total": int(seed_df["stage2_correct_to_stage4_wrong"].sum()),
            "net_correction_total": int(
                seed_df["stage2_wrong_to_stage4_correct"].sum()
                - seed_df["stage2_correct_to_stage4_wrong"].sum()
            ),
            "selected_for_gradcam": str(analysis_selected_csv(candidate, augmented=True)),
        }
        if not transition_df.empty:
            row["transition_rows"] = int(len(transition_df))
        rows.append(row)
    summary = pd.DataFrame(rows)
    output = PROJECT_ROOT / "reports/tables/stage4_n13_6_interpretability_candidate_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def zip_interpretability_outputs(candidates: list[dict]):
    """Create compact downloadable N13-6 bundle."""

    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    manifest = {"files": []}
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        table_patterns = ["stage4_n13_6_*"]
        for pattern in table_patterns:
            for path in sorted((PROJECT_ROOT / "reports/tables").glob(pattern)):
                if path.is_file():
                    arcname = Path("stage4/reports/tables") / path.name
                    archive.write(path, arcname)
                    manifest["files"].append(str(arcname))

        suffixes = [candidate["output_suffix"] for candidate in candidates]
        for root, prefix in [
            (PROJECT_ROOT / "reports/figures/gradcam", "stage4/reports/figures/gradcam"),
            (STAGE2_PROJECT_ROOT / "reports/figures/gradcam", "stage2/reports/figures/gradcam"),
        ]:
            if not root.exists():
                continue
            for suffix in suffixes:
                for path in sorted(root.glob(f"*{suffix}*")):
                    if path.is_file():
                        arcname = Path(prefix) / path.name
                        archive.write(path, arcname)
                        manifest["files"].append(str(arcname))

        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print("Bundle:", BUNDLE_PATH, round(BUNDLE_PATH.stat().st_size / (1024 * 1024), 2), "MB", flush=True)


prepare_working_snapshots()
assert_required_files()
patch_stage2_config()
ensure_stage2_outputs()

active_candidates = [candidate_by_key(key) for key in RUN_CANDIDATE_KEYS]
for candidate in active_candidates:
    ensure_stage4_predictions(candidate)
    run_correction_analysis(candidate)
    run_targeted_exports(candidate)

summary = build_compact_summary(active_candidates)
zip_interpretability_outputs(active_candidates)

display(Markdown("# N13-6 Interpretability Candidate Summary"))
display(summary)
for candidate in active_candidates:
    selected_path = analysis_selected_csv(candidate, augmented=True)
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        display(Markdown(f"## Selected samples: {candidate['analysis_name']}"))
        columns = [
            column for column in [
                "run_seed",
                "sample_index",
                "Date",
                "label",
                "analysis_group",
                "prob_up_stage2",
                "prob_up_stage4",
                "true_prob_delta_stage4_minus_stage2",
                "future_return",
            ] if column in selected.columns
        ]
        display(selected[columns].head(40))

display(Markdown("# N13-6 Download Bundle"))
print(BUNDLE_PATH)
print("DONE", flush=True)
```
