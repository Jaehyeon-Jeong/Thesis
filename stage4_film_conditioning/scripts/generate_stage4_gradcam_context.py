#!/usr/bin/env python
"""Export Stage 4 Grad-CAM plus context/gate/gamma/beta metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import (
    CONTEXT_METHODS,
    get_stage4_model_config,
    stage4_run_context_base,
    validate_context_method,
)
from stage4_film.evaluation import load_stage4_checkpoint_into_model
from stage4_film.interpretability import generate_stage4_gradcam_context_figure
from stage4_film.paths import experiment_output_roots
from stage4_film.runners import (
    build_stage4_context_model,
    prepare_stage4_context_experiment_data,
)
from stage4_film.runtime import select_device


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=0)
    parser.add_argument("--image-spec", default="")
    parser.add_argument("--return-horizon", type=int, default=0)
    parser.add_argument(
        "--context-method",
        default="concat",
        choices=list(CONTEXT_METHODS),
    )
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--write-report-copy", action="store_true")
    parser.add_argument(
        "--selected-samples-csv",
        default="",
        help="Optional N10-A selected sample CSV for targeted Grad-CAM export.",
    )
    parser.add_argument(
        "--target-class-source",
        default="label",
        choices=["label", "pred_class", "pred_class_stage2", "pred_class_stage4"],
        help="Column used as the Grad-CAM target class in targeted mode.",
    )
    parser.add_argument(
        "--selected-limit-per-panel",
        type=int,
        default=0,
        help="Optional max samples per correction/regression panel; 0 keeps all selected rows.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for targeted output filenames, without extension.",
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Generate Stage 4 Grad-CAM/context/modulation outputs."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_method = validate_context_method(args.context_method)
    run_seed = int(args.run_seed)
    device = select_device(config)

    run_context = stage4_run_context_base(
        config,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_method=context_method,
        run_seed=run_seed,
    )
    experiment_name = str(run_context["stage4_experiment_name"])
    output_roots = experiment_output_roots(paths, experiment_name, run_seed)
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    checkpoint_path = output_roots["checkpoint"] / "best.pt"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")

    predictions = pd.read_csv(prediction_path)
    selected_samples = _load_targeted_samples(
        args.selected_samples_csv,
        run_seed=run_seed,
        model_prefix="stage4",
        target_class_source=str(args.target_class_source),
        selected_limit_per_panel=int(args.selected_limit_per_panel),
    )
    prepared = prepare_stage4_context_experiment_data(
        config=config,
        paths=paths,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        run_seed=run_seed,
        max_train_rows=_optional_positive_int(args.max_train_rows),
        max_validation_rows=_optional_positive_int(args.max_validation_rows),
        max_test_rows=_optional_positive_int(args.max_test_rows),
    )
    model = build_stage4_context_model(
        config=config,
        image_window=image_window,
        context_method=context_method,
    )
    checkpoint = load_stage4_checkpoint_into_model(model, checkpoint_path, device)
    context_feature_names = [
        f"{feature}_normalized" for feature in prepared.context_scaler.feature_order
    ]
    figure_path = (
        output_roots["figures"]
        / "gradcam"
        / args.split
        / _gradcam_filename(args.split, args.samples_per_class, args.output_suffix)
    )
    written = generate_stage4_gradcam_context_figure(
        model,
        prepared.datasets[args.split],
        predictions,
        figure_path,
        samples_per_class=int(args.samples_per_class),
        device=device,
        context_method=context_method,
        context_feature_names=context_feature_names,
        selected_samples=selected_samples,
    )
    result = {
        "status": "ok",
        "experiment_name": experiment_name,
        "context_method": context_method,
        "split": args.split,
        "run_seed": run_seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "prediction_path": str(prediction_path),
        "written": {
            "figure": str(written["figure"]),
            "samples": str(written["samples"]),
            "modulation_summary": str(written["modulation_summary"]),
            "modulation_values": str(written["modulation_values"]),
        },
    }
    if args.write_report_copy:
        report_dir = paths.reports_root / "figures" / "gradcam"
        report_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        report_figure = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_context_gradcam{suffix}.png"
        )
        report_samples = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_context_gradcam_samples{suffix}.csv"
        )
        report_summary = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_modulation_summary{suffix}.csv"
        )
        report_values = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_modulation_values{suffix}.json"
        )
        report_figure.write_bytes(written["figure"].read_bytes())
        report_samples.write_bytes(written["samples"].read_bytes())
        report_summary.write_bytes(written["modulation_summary"].read_bytes())
        report_values.write_bytes(written["modulation_values"].read_bytes())
        result["report_copy"] = {
            "figure": str(report_figure),
            "samples": str(report_samples),
            "modulation_summary": str(report_summary),
            "modulation_values": str(report_values),
        }
    print(json.dumps(result, indent=2, default=str))


def _load_targeted_samples(
    selected_samples_csv: str,
    *,
    run_seed: int,
    model_prefix: str,
    target_class_source: str,
    selected_limit_per_panel: int,
) -> pd.DataFrame | None:
    """Read N10-A selected samples and map model-specific columns."""

    if not selected_samples_csv:
        return None
    path = Path(selected_samples_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Selected samples CSV not found: {path}")
    frame = pd.read_csv(path)
    if "run_seed" in frame.columns:
        frame = frame[pd.to_numeric(frame["run_seed"], errors="raise").astype(int).eq(run_seed)]
    if frame.empty:
        raise ValueError(f"No selected samples for seed {run_seed}: {path}")

    transition_column = "analysis_group" if "analysis_group" in frame.columns else "transition"
    if transition_column in frame.columns and selected_limit_per_panel > 0:
        frame = (
            frame.groupby(transition_column, group_keys=False)
            .head(int(selected_limit_per_panel))
            .reset_index(drop=True)
        )
    frame = frame.copy()
    if target_class_source == "pred_class":
        source_column = f"pred_class_{model_prefix}"
    else:
        source_column = target_class_source
    if source_column not in frame.columns:
        raise KeyError(f"Target class column missing in selected CSV: {source_column}")

    mapped = frame.copy()
    mapped["target_class"] = pd.to_numeric(mapped[source_column], errors="raise").astype(int)
    mapped["target_class_name"] = mapped["target_class"].map(_class_name)
    mapped["gradcam_panel"] = mapped[transition_column].map(_transition_panel) if transition_column in mapped.columns else "targeted"
    mapped["gradcam_panel_label"] = mapped["gradcam_panel"].map(_panel_label)
    mapped["gradcam_sample_fallback"] = False

    for column in ["Date", "label", "future_return"]:
        preferred = f"{column}_{model_prefix}"
        if preferred in mapped.columns:
            mapped[column] = mapped[preferred]
    for column in ["prob_down", "prob_up", "pred_class", "correct"]:
        preferred = f"{column}_{model_prefix}"
        if preferred in mapped.columns:
            mapped[column] = mapped[preferred]
    return mapped


def _gradcam_filename(split: str, samples_per_class: int, output_suffix: str) -> str:
    """Return default or targeted Stage 4 Grad-CAM filename."""

    if output_suffix:
        return f"btc_context_gradcam_{split}_{output_suffix}.png"
    return f"btc_context_gradcam_{split}_{samples_per_class}perclass.png"


def _transition_panel(value: object) -> str:
    """Convert transition labels to short panel ids."""

    text = str(value)
    if "wrong_to_stage4_correct" in text:
        return "correction"
    if "correct_to_stage4_wrong" in text:
        return "regression"
    return text.replace(" ", "_").lower()


def _panel_label(panel: object) -> str:
    """Readable targeted panel label."""

    text = str(panel)
    if text == "correction":
        return "Stage2 wrong -> N10 correct"
    if text == "regression":
        return "Stage2 correct -> N10 wrong"
    return text.replace("_", " ").title()


def _class_name(class_id: int) -> str:
    """Return readable class name."""

    return "Up" if int(class_id) == 1 else "Down"


def _optional_positive_int(value: int) -> int | None:
    """Convert CLI row limit values to optional positive integers."""

    integer = int(value)
    return integer if integer > 0 else None


if __name__ == "__main__":
    main()
