#!/usr/bin/env python
"""2-I8: Generate BTC Grad-CAM figure for one Stage 2 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def add_src_to_path() -> Path:
    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def main() -> None:
    stage_root = add_src_to_path()
    from stage2_btc import (
        build_stage2_paths,
        experiment_output_roots,
        load_config,
        make_experiment_name,
        select_device,
    )
    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.interpretability import generate_gradcam_figure
    from stage2_btc.models import build_stock_cnn_for_window
    from stage2_btc.runners.btc_baseline import prepare_btc_experiment_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
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
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    device = select_device(config)
    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    prediction_path = output_roots["predictions"] / f"{args.split}_predictions.csv"
    checkpoint_path = output_roots["checkpoint"] / "best.pt"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    selected_samples = _load_targeted_samples(
        args.selected_samples_csv,
        run_seed=int(args.run_seed),
        model_prefix="stage2",
        target_class_source=str(args.target_class_source),
        selected_limit_per_panel=int(args.selected_limit_per_panel),
    )
    prepared = prepare_btc_experiment_data(
        config,
        paths,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        max_test_rows=args.max_test_rows,
    )
    model = build_stock_cnn_for_window(config, args.image_window)
    load_checkpoint_into_model(model, checkpoint_path, device)
    figure_path = (
        output_roots["figures"]
        / "gradcam"
        / args.split
        / _gradcam_filename(args.split, args.samples_per_class, args.output_suffix)
    )
    written_figure, selected = generate_gradcam_figure(
        model,
        prepared.datasets[args.split],
        predictions,
        figure_path,
        samples_per_class=args.samples_per_class,
        device=device,
        selected_samples=selected_samples,
    )
    samples_path = written_figure.parent / "samples.csv"
    selected.to_csv(samples_path, index=False)
    result = {
        "status": "ok",
        "figure": str(written_figure),
        "samples": str(samples_path),
    }
    if args.write_report_copy:
        report_dir = paths.reports_root / "figures" / "gradcam"
        report_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        report_copy = report_dir / (
            f"{experiment_name}_seed_{args.run_seed}_{args.split}_gradcam{suffix}.png"
        )
        report_samples = report_dir / (
            f"{experiment_name}_seed_{args.run_seed}_{args.split}_gradcam_samples{suffix}.csv"
        )
        report_copy.write_bytes(written_figure.read_bytes())
        report_samples.write_bytes(samples_path.read_bytes())
        result["report_copy"] = {
            "figure": str(report_copy),
            "samples": str(report_samples),
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
    """Return default or targeted Grad-CAM filename."""

    if output_suffix:
        return f"btc_gradcam_{split}_{output_suffix}.png"
    return f"btc_gradcam_{split}_{samples_per_class}perclass.png"


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


if __name__ == "__main__":
    main()
