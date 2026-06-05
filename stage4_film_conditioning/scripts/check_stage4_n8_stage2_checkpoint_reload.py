#!/usr/bin/env python
"""4-N8-A1: Stage 2 checkpoint reload sanity for baseline-preserving FiLM.

This script does not train a Stage 4 model. It verifies that Stage 4 can use the
selected Stage 2 BTC checkpoint bundle as the pretrained visual baseline before
running frozen or partial-frozen context-FiLM experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


def _resolve_stage2_output_root(bundle_root: Path) -> Path:
    """Return the `outputs/stage2` root from a bundle root or direct output path."""

    root = bundle_root.expanduser().resolve()
    candidates = [
        root / "outputs" / "stage2",
        root / "stage2",
        root,
    ]
    for candidate in candidates:
        if (candidate / "checkpoints").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Stage 2 checkpoint root. Expected one of: "
        + ", ".join(str(candidate / "checkpoints") for candidate in candidates)
    )


def _find_seed_results_csv(bundle_root: Path) -> Path | None:
    """Find the seed-level metric table inside the Stage 2 checkpoint bundle."""

    tables_root = bundle_root.expanduser().resolve() / "reports" / "tables"
    if not tables_root.exists():
        return None
    matches = sorted(tables_root.glob("*seed_results.csv"))
    return matches[0] if matches else None


def _patch_stage2_config_for_bundle(
    config: dict[str, Any],
    output_root: Path,
    stage4_config: dict[str, Any],
) -> dict[str, Any]:
    """Use the local Stage 4 data source while reading checkpoints from bundle."""

    patched = json.loads(json.dumps(config, default=str))
    paths = patched["paths"]
    stage4_paths = stage4_config["paths"]

    paths["data_root"] = stage4_paths["data_root"]
    paths["source_file"] = stage4_paths["source_file"]
    paths["output_root"] = str(output_root)
    paths["checkpoint_root"] = str(output_root / "checkpoints")
    paths["metrics_root"] = str(output_root / "metrics")
    paths["predictions_root"] = str(output_root / "predictions")
    paths["figures_root"] = str(output_root / "figures")
    paths["run_manifest_root"] = str(output_root / "run_manifests")

    patched["data"]["source_file"] = stage4_paths["source_file"]
    patched["runtime"]["num_workers"] = 0
    patched["runtime"]["pin_memory"] = False
    patched["runtime"]["persistent_workers"] = False
    return patched


def _expected_row(
    expected: pd.DataFrame | None,
    experiment_name: str,
    run_seed: int,
) -> dict[str, Any]:
    """Return the matching expected metric row from bundle CSV, if available."""

    if expected is None:
        return {}
    subset = expected[
        (expected["experiment_name"].astype(str) == str(experiment_name))
        & (expected["run_seed"].astype(int) == int(run_seed))
    ]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _metric_delta_rows(
    observed: dict[str, Any],
    expected: dict[str, Any],
    metric_names: list[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Build one comparison row per metric."""

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        obs_value = observed.get(metric)
        exp_value = expected.get(metric)
        if obs_value is None or exp_value is None or pd.isna(exp_value):
            delta = None
            abs_delta = None
            ok = None
        else:
            delta = float(obs_value) - float(exp_value)
            abs_delta = abs(delta)
            ok = abs_delta <= float(tolerance)
        rows.append(
            {
                "metric": metric,
                "observed": obs_value,
                "expected": exp_value,
                "delta": delta,
                "abs_delta": abs_delta,
                "within_tolerance": ok,
                "tolerance": float(tolerance),
            }
        )
    return rows


def main() -> None:
    stage_root = add_stage4_and_stage2_src_from_argv(sys.argv)

    from stage2_btc import (
        build_stage2_paths,
        experiment_output_roots,
        load_config as load_stage2_config,
        make_experiment_name,
        select_device,
    )
    from stage2_btc.evaluation import compute_classification_metrics, predict_loader, write_json
    from stage2_btc.evaluation.prediction import load_checkpoint_into_model
    from stage2_btc.models import build_stock_cnn_for_window
    from stage2_btc.runners.btc_baseline import build_dataloaders, prepare_btc_experiment_data
    from stage4_film import load_config as load_stage4_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--stage2-config", default="")
    parser.add_argument("--stage2-bundle-root", required=True)
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-prefix", default="stage4_n8_stage2_checkpoint_reload")
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--comparison-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    stage4_config_path = Path(args.config)
    if not stage4_config_path.is_absolute():
        stage4_config_path = stage_root / stage4_config_path
    stage4_config = load_stage4_config(stage4_config_path)

    stage2_project_root = Path(stage4_config["stage2_dependency"]["project_root"]).expanduser()
    stage2_config_path = Path(args.stage2_config).expanduser() if args.stage2_config else (
        stage2_project_root / "configs" / "env_local.yaml"
    )
    stage2_config = load_stage2_config(stage2_config_path)

    bundle_root = Path(args.stage2_bundle_root).expanduser().resolve()
    stage2_output_root = _resolve_stage2_output_root(bundle_root)
    patched_config = _patch_stage2_config_for_bundle(stage2_config, stage2_output_root, stage4_config)
    paths = build_stage2_paths(patched_config)
    device = select_device(patched_config)
    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)

    expected_csv = _find_seed_results_csv(bundle_root)
    expected_df = pd.read_csv(expected_csv) if expected_csv is not None else None

    prepared = prepare_btc_experiment_data(
        patched_config,
        paths,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        max_test_rows=args.max_test_rows,
    )
    loaders = build_dataloaders(prepared.datasets, patched_config, shuffle_train=False)

    tables_root = Path(stage4_config["paths"]["tables_root"]).expanduser()
    tables_root.mkdir(parents=True, exist_ok=True)
    prediction_root = Path(stage4_config["paths"]["predictions_root"]).expanduser()

    metric_names = [
        "num_samples",
        "accuracy",
        "majority_class_accuracy",
        "roc_auc",
        "average_precision",
        "f1",
        "brier_score",
        "positive_rate",
        "predicted_positive_rate",
    ]
    seed_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for run_seed in args.run_seeds:
        output_roots = experiment_output_roots(paths, experiment_name, run_seed)
        checkpoint_path = output_roots["checkpoint"] / "best.pt"
        model = build_stock_cnn_for_window(patched_config, args.image_window)
        checkpoint = load_checkpoint_into_model(model, checkpoint_path, device)
        run_context = {
            "experiment_name": experiment_name,
            "image_window": int(args.image_window),
            "image_spec": str(args.image_spec),
            "return_horizon": int(args.return_horizon),
            "run_seed": int(run_seed),
        }
        predictions = predict_loader(
            model,
            loaders[args.split],
            patched_config,
            device=device,
            run_context=run_context,
            checkpoint_path=checkpoint_path,
            split_name=args.split,
        )
        metrics = compute_classification_metrics(predictions)
        expected = _expected_row(expected_df, experiment_name, run_seed)
        row = {
            "experiment_name": experiment_name,
            "run_seed": int(run_seed),
            "split": args.split,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "expected_metrics_available": bool(expected),
        }
        for metric in metric_names:
            row[metric] = metrics.get(metric)
            if metric in expected and not pd.isna(expected[metric]):
                row[f"expected_{metric}"] = expected[metric]
                row[f"{metric}_delta"] = float(metrics[metric]) - float(expected[metric])
        seed_rows.append(row)

        for comparison in _metric_delta_rows(
            metrics,
            expected,
            metric_names,
            tolerance=float(args.comparison_tolerance),
        ):
            comparison.update(
                {
                    "experiment_name": experiment_name,
                    "run_seed": int(run_seed),
                    "split": args.split,
                }
            )
            comparison_rows.append(comparison)

        if args.write_predictions:
            prediction_path = (
                prediction_root
                / f"{args.output_prefix}_{experiment_name}_seed_{int(run_seed)}_{args.split}_predictions.csv"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)

    seed_df = pd.DataFrame(seed_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    seed_csv = tables_root / f"{args.output_prefix}_seed_results.csv"
    comparison_csv = tables_root / f"{args.output_prefix}_metric_comparison.csv"
    report_json = tables_root / f"{args.output_prefix}_report.json"
    seed_df.to_csv(seed_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)

    comparison_ok = bool(
        comparison_df["within_tolerance"].dropna().all()
        if not comparison_df.empty
        else False
    )
    report = {
        "status": "ok",
        "stage": "4-N8-A1",
        "purpose": "Stage 2 checkpoint reload sanity before baseline-preserving FiLM.",
        "stage4_config": str(stage4_config_path),
        "stage2_config": str(stage2_config_path),
        "stage2_bundle_root": str(bundle_root),
        "stage2_output_root": str(stage2_output_root),
        "experiment_name": experiment_name,
        "run_seeds": [int(seed) for seed in args.run_seeds],
        "split": args.split,
        "expected_csv": str(expected_csv) if expected_csv is not None else None,
        "comparison_tolerance": float(args.comparison_tolerance),
        "comparison_within_tolerance": comparison_ok,
        "seed_results_csv": str(seed_csv),
        "metric_comparison_csv": str(comparison_csv),
    }
    write_json(report_json, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
