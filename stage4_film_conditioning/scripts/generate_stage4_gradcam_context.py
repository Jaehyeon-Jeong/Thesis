#!/usr/bin/env python
"""Export Stage 4 Grad-CAM plus context/gate/gamma/beta metadata."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import (
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
        choices=["concat", "gating", "film_gamma", "film_full"],
    )
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--write-report-copy", action="store_true")
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
        / f"btc_context_gradcam_{args.split}_{args.samples_per_class}perclass.png"
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
        report_figure = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_context_gradcam.png"
        )
        report_samples = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_context_gradcam_samples.csv"
        )
        report_summary = report_dir / (
            f"{experiment_name}_seed_{run_seed}_{args.split}_modulation_summary.csv"
        )
        report_figure.write_bytes(written["figure"].read_bytes())
        report_samples.write_bytes(written["samples"].read_bytes())
        report_summary.write_bytes(written["modulation_summary"].read_bytes())
        result["report_copy"] = {
            "figure": str(report_figure),
            "samples": str(report_samples),
            "modulation_summary": str(report_summary),
        }
    print(json.dumps(result, indent=2, default=str))


def _optional_positive_int(value: int) -> int | None:
    """Convert CLI row limit values to optional positive integers."""

    integer = int(value)
    return integer if integer > 0 else None


if __name__ == "__main__":
    main()
