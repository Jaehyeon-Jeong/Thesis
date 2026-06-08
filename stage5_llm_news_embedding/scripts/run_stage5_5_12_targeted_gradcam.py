#!/usr/bin/env python
"""Run Stage5 5-12 targeted Grad-CAM and FiLM modulation export locally.

The script uses the already-finished 5-9E FinBERT+F&G result bundle and the
Stage2 I60/R20/ohlc_ma_vb checkpoint bundle. It does not train a model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STAGE5_ROOT = WORKSPACE_ROOT / "stage5_llm_news_embedding"
STAGE4_ROOT = WORKSPACE_ROOT / "stage4_film_conditioning"
STAGE2_ROOT = WORKSPACE_ROOT / "stage2_btc_extension"

RESULT_ROOT = WORKSPACE_ROOT / "5_9e_results"
STAGE2_BUNDLE_ROOT = WORKSPACE_ROOT / "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
MODULATION_SCALE = 0.02
RUN_SEEDS = [42, 43, 44, 45, 46]
SPLIT = "test"
SELECTED_LIMIT_PER_PANEL = 4
OUTPUT_SUFFIX = "5_12_finbert_fg_targeted_label"

FEATURE_SET_NAME = "stage5_finbert_fg_sentiment_v1"
SCALE_LABEL = "s0p02"
EXPERIMENT_SUFFIX = f"{FEATURE_SET_NAME}_pretrained_frozen_{SCALE_LABEL}"
STAGE2_EXPERIMENT = f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"
STAGE5_EXPERIMENT = (
    f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
    f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{EXPERIMENT_SUFFIX}"
)
CONTEXT_NAME = (
    f"stage5_finbert_fg_context_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
    f"r{RETURN_HORIZON}_{FEATURE_SET_NAME}"
)

SELECTED_SAMPLES = (
    STAGE5_ROOT
    / "reports/tables/stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv"
)
OUTPUT_PREFIX = "stage5_5_12_finbert_fg_targeted_gradcam"
REPORT_TABLE_DIR = STAGE5_ROOT / "reports/tables"
REPORT_FIGURE_DIR = STAGE5_ROOT / "reports/figures/gradcam"


def main() -> None:
    """Run the 5-12 export pipeline."""

    _assert_inputs()
    _overlay_outputs()
    config_path = _write_stage4_export_config()

    result_rows: list[dict[str, Any]] = []
    for seed in RUN_SEEDS:
        result_rows.append(_run_stage2_gradcam(seed))
        result_rows.append(_run_stage5_gradcam(seed, config_path))

    copied_files = _copy_export_artifacts()
    summary = _build_summary_tables()
    report_path = _write_report(result_rows, copied_files, summary)
    manifest_path = _write_manifest(result_rows, copied_files, summary, report_path, config_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "manifest": str(manifest_path),
                "copied_files": len(copied_files),
            },
            indent=2,
        )
    )


def _assert_inputs() -> None:
    required = [
        RESULT_ROOT / "outputs/stage4",
        RESULT_ROOT
        / "outputs/stage4/checkpoints"
        / STAGE5_EXPERIMENT
        / "seed_42/best.pt",
        RESULT_ROOT / "outputs/stage4/context" / CONTEXT_NAME / "seed_42/context_scaler.json",
        STAGE2_BUNDLE_ROOT / "outputs/stage2/checkpoints" / STAGE2_EXPERIMENT / "seed_42/best.pt",
        SELECTED_SAMPLES,
        STAGE4_ROOT / "scripts/generate_stage4_gradcam_context.py",
        STAGE2_ROOT / "scripts/generate_stage2_gradcam.py",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def _overlay_outputs() -> None:
    """Copy saved Stage2 and Stage5 outputs into the configured local roots."""

    shutil.copytree(
        STAGE2_BUNDLE_ROOT / "outputs/stage2",
        STAGE2_ROOT / "outputs/stage2",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        RESULT_ROOT / "outputs/stage4",
        STAGE4_ROOT / "outputs/stage4",
        dirs_exist_ok=True,
    )


def _write_stage4_export_config() -> Path:
    """Write a local Stage4 config matching the 5-9E trained checkpoint."""

    base_config = STAGE4_ROOT / "configs/env_local.yaml"
    config_path = STAGE4_ROOT / "configs/env_stage5_5_12_local.yaml"
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    scaler_path = (
        STAGE4_ROOT
        / "outputs/stage4/context"
        / CONTEXT_NAME
        / "seed_42/context_scaler.json"
    )
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_order = [str(feature) for feature in scaler["feature_order"]]

    cfg["paths"]["project_root"] = str(STAGE4_ROOT)
    cfg["paths"]["output_root"] = str(STAGE4_ROOT / "outputs/stage4")
    cfg["paths"]["checkpoint_root"] = str(STAGE4_ROOT / "outputs/stage4/checkpoints")
    cfg["paths"]["metrics_root"] = str(STAGE4_ROOT / "outputs/stage4/metrics")
    cfg["paths"]["predictions_root"] = str(STAGE4_ROOT / "outputs/stage4/predictions")
    cfg["paths"]["figures_root"] = str(STAGE4_ROOT / "outputs/stage4/figures")
    cfg["paths"]["context_root"] = str(STAGE4_ROOT / "outputs/stage4/context")
    cfg["paths"]["run_manifest_root"] = str(STAGE4_ROOT / "outputs/stage4/run_manifests")
    cfg["paths"]["reports_root"] = str(STAGE4_ROOT / "reports")
    cfg["paths"]["tables_root"] = str(STAGE4_ROOT / "reports/tables")

    cfg["stage2_dependency"]["project_root"] = str(STAGE2_ROOT)
    cfg["stage2_dependency"]["src_path"] = str(STAGE2_ROOT / "src")
    cfg["stage2_dependency"]["baseline_output_root"] = str(STAGE2_ROOT / "outputs/stage2")
    cfg["stage2_dependency"]["selected_baseline_experiment"] = STAGE2_EXPERIMENT

    cfg["context"]["source"] = "prebuilt"
    cfg["context"]["prebuilt_context_name"] = CONTEXT_NAME
    cfg["context"]["context_window"] = CONTEXT_WINDOW
    cfg["context"]["feature_set_name"] = FEATURE_SET_NAME
    cfg["context"]["primary_features"] = feature_order

    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["num_workers"] = 0
    cfg["runtime"]["pin_memory"] = False
    cfg["runtime"]["persistent_workers"] = False

    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON
    cfg["stage4_model"]["context_dim"] = len(feature_order)
    cfg["stage4_model"]["context_methods"] = [CONTEXT_METHOD]
    cfg["stage4_model"]["experiment_suffix"] = EXPERIMENT_SUFFIX
    cfg["stage4_model"]["film_full_bounded_last_block"]["modulation_scale"] = MODULATION_SCALE
    pretrained = dict(cfg["stage4_model"].get("pretrained_stage2", {}))
    pretrained["enabled"] = True
    pretrained["checkpoint_output_root"] = str(STAGE2_ROOT / "outputs/stage2")
    pretrained["freeze_visual_backbone"] = True
    pretrained["freeze_classifier"] = True
    pretrained["initialize_new_context_modules"] = True
    pretrained["strict_load"] = True
    cfg["stage4_model"]["pretrained_stage2"] = pretrained

    cfg["training"]["batch_size"] = 128
    cfg["training"]["mixed_precision"] = False
    cfg["training"]["data_parallel"] = False
    cfg["evaluation"]["batch_size"] = 128

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def _run_stage2_gradcam(seed: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/generate_stage2_gradcam.py",
        "--config",
        "configs/env_local.yaml",
        "--image-window",
        str(IMAGE_WINDOW),
        "--image-spec",
        IMAGE_SPEC,
        "--return-horizon",
        str(RETURN_HORIZON),
        "--run-seed",
        str(seed),
        "--split",
        SPLIT,
        "--samples-per-class",
        "2",
        "--selected-samples-csv",
        str(SELECTED_SAMPLES),
        "--target-class-source",
        "label",
        "--selected-limit-per-panel",
        str(SELECTED_LIMIT_PER_PANEL),
        "--output-suffix",
        OUTPUT_SUFFIX,
        "--write-report-copy",
    ]
    _run(cmd, cwd=STAGE2_ROOT)
    return {"model": "stage2", "seed": int(seed), "command": cmd}


def _run_stage5_gradcam(seed: int, config_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/generate_stage4_gradcam_context.py",
        "--config",
        str(config_path),
        "--image-window",
        str(IMAGE_WINDOW),
        "--image-spec",
        IMAGE_SPEC,
        "--return-horizon",
        str(RETURN_HORIZON),
        "--context-method",
        CONTEXT_METHOD,
        "--run-seed",
        str(seed),
        "--split",
        SPLIT,
        "--samples-per-class",
        "2",
        "--selected-samples-csv",
        str(SELECTED_SAMPLES),
        "--target-class-source",
        "label",
        "--selected-limit-per-panel",
        str(SELECTED_LIMIT_PER_PANEL),
        "--output-suffix",
        OUTPUT_SUFFIX,
        "--write-report-copy",
    ]
    _run(cmd, cwd=STAGE4_ROOT)
    return {"model": "stage5", "seed": int(seed), "command": cmd}


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, text=True)


def _copy_export_artifacts() -> list[str]:
    """Copy report-level Grad-CAM artifacts into Stage5 reports."""

    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    patterns = [
        STAGE2_ROOT / "reports/figures/gradcam" / f"*{OUTPUT_SUFFIX}*",
        STAGE4_ROOT / "reports/figures/gradcam" / f"*{OUTPUT_SUFFIX}*",
    ]
    for pattern in patterns:
        for source in sorted(pattern.parent.glob(pattern.name)):
            target = REPORT_FIGURE_DIR / source.name
            shutil.copy2(source, target)
            copied.append(str(target))
    return copied


def _build_summary_tables() -> dict[str, str]:
    """Combine selected Stage5 sample/modulation summaries across seeds."""

    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stage5_modulation_frames = []
    stage5_sample_frames = []
    stage2_sample_frames = []

    for seed in RUN_SEEDS:
        stage5_report_prefix = (
            STAGE4_ROOT
            / "reports/figures/gradcam"
            / f"{STAGE5_EXPERIMENT}_seed_{seed}_{SPLIT}"
        )
        stage2_report_prefix = (
            STAGE2_ROOT
            / "reports/figures/gradcam"
            / f"{STAGE2_EXPERIMENT}_seed_{seed}_{SPLIT}"
        )
        stage5_mod = Path(f"{stage5_report_prefix}_modulation_summary_{OUTPUT_SUFFIX}.csv")
        stage5_samples = Path(f"{stage5_report_prefix}_context_gradcam_samples_{OUTPUT_SUFFIX}.csv")
        stage2_samples = Path(f"{stage2_report_prefix}_gradcam_samples_{OUTPUT_SUFFIX}.csv")
        if stage5_mod.exists():
            frame = pd.read_csv(stage5_mod)
            frame["run_seed"] = int(seed)
            stage5_modulation_frames.append(frame)
        if stage5_samples.exists():
            frame = pd.read_csv(stage5_samples)
            frame["run_seed"] = int(seed)
            stage5_sample_frames.append(frame)
        if stage2_samples.exists():
            frame = pd.read_csv(stage2_samples)
            frame["run_seed"] = int(seed)
            stage2_sample_frames.append(frame)

    outputs: dict[str, str] = {}
    if stage5_modulation_frames:
        modulation = pd.concat(stage5_modulation_frames, ignore_index=True)
        modulation_path = REPORT_TABLE_DIR / f"{OUTPUT_PREFIX}_modulation_summary.csv"
        modulation.to_csv(modulation_path, index=False)
        outputs["modulation_summary"] = str(modulation_path)

        stats_columns = [
            column
            for column in modulation.columns
            if column.startswith("block4_")
            and (
                column.endswith("_mean")
                or column.endswith("_std")
                or column.endswith("_l2")
            )
        ]
        group_columns = ["gradcam_panel"]
        available_stats = [column for column in stats_columns if column in modulation.columns]
        if available_stats:
            by_panel = (
                modulation.groupby(group_columns, dropna=False)[available_stats]
                .mean(numeric_only=True)
                .reset_index()
            )
            by_panel_path = REPORT_TABLE_DIR / f"{OUTPUT_PREFIX}_modulation_by_panel.csv"
            by_panel.to_csv(by_panel_path, index=False)
            outputs["modulation_by_panel"] = str(by_panel_path)

    if stage5_sample_frames:
        samples = pd.concat(stage5_sample_frames, ignore_index=True)
        samples_path = REPORT_TABLE_DIR / f"{OUTPUT_PREFIX}_stage5_samples.csv"
        samples.to_csv(samples_path, index=False)
        outputs["stage5_samples"] = str(samples_path)

    if stage2_sample_frames:
        samples = pd.concat(stage2_sample_frames, ignore_index=True)
        samples_path = REPORT_TABLE_DIR / f"{OUTPUT_PREFIX}_stage2_samples.csv"
        samples.to_csv(samples_path, index=False)
        outputs["stage2_samples"] = str(samples_path)

    return outputs


def _write_report(
    result_rows: list[dict[str, Any]],
    copied_files: list[str],
    summary: dict[str, str],
) -> Path:
    selected = pd.read_csv(SELECTED_SAMPLES)
    lines = [
        "# 5-12 Stage5 FinBERT+F&G Targeted Grad-CAM and Modulation Export",
        "",
        "## Purpose",
        "",
        "Export Stage2 and Stage5 Grad-CAM figures for the selected 5-11",
        "correction/regression samples, together with Stage5 gamma/beta",
        "modulation summaries and aggregated FinBERT/F&G context values.",
        "",
        "## Scope",
        "",
        f"- Stage2 baseline: `{STAGE2_EXPERIMENT}`",
        f"- Stage5 candidate: `{STAGE5_EXPERIMENT}`",
        f"- Context: `{CONTEXT_NAME}`",
        f"- Seeds: `{RUN_SEEDS}`",
        f"- Target class source: true `label`",
        f"- Selected limit per transition panel and seed: `{SELECTED_LIMIT_PER_PANEL}`",
        f"- Selected input rows before per-seed/panel limit: `{len(selected)}`",
        "",
        "## Exported Artifacts",
        "",
    ]
    for label, path in summary.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            f"- copied Grad-CAM/report artifacts: `{len(copied_files)}` files in "
            f"`{REPORT_FIGURE_DIR}`",
            "",
            "## Interpretation Notes",
            "",
            "- This export is post-hoc. It does not train or tune the model.",
            "- Stage2 and Stage5 use the same selected samples and true-label Grad-CAM target.",
            "- Stage5 modulation uses the 5-9E trained checkpoint with bounded FiLM scale `0.02`.",
            "- Raw news text is not redistributed here. The exported context is the aggregated",
            "  FinBERT/F&G numeric context already used by the model.",
            "",
            "## Commands",
            "",
            f"- Total commands run: `{len(result_rows)}`",
        ]
    )
    report_path = REPORT_TABLE_DIR / f"{OUTPUT_PREFIX}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_manifest(
    result_rows: list[dict[str, Any]],
    copied_files: list[str],
    summary: dict[str, str],
    report_path: Path,
    config_path: Path,
) -> Path:
    manifest = {
        "status": "ok",
        "stage": "5-12",
        "stage2_experiment": STAGE2_EXPERIMENT,
        "stage5_experiment": STAGE5_EXPERIMENT,
        "context_name": CONTEXT_NAME,
        "run_seeds": RUN_SEEDS,
        "selected_samples": str(SELECTED_SAMPLES),
        "selected_limit_per_panel": SELECTED_LIMIT_PER_PANEL,
        "output_suffix": OUTPUT_SUFFIX,
        "config_path": str(config_path),
        "report": str(report_path),
        "summary_outputs": summary,
        "copied_files": copied_files,
        "commands": result_rows,
    }
    path = REPORT_TABLE_DIR / f"{OUTPUT_PREFIX}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
