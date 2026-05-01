#!/usr/bin/env python
"""2-I5: Train one Stage 2 BTC baseline experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
        set_global_seed,
    )
    from stage2_btc.models import build_stock_cnn_for_window, count_parameters
    from stage2_btc.runners.btc_baseline import (
        build_dataloaders,
        prepare_btc_experiment_data,
        split_summary,
    )
    from stage2_btc.training import fit_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = int(args.max_epochs)
    paths = build_stage2_paths(config)
    device = select_device(config)
    deterministic = bool(config["training"]["determinism"].get("enabled", True))
    seed_info = set_global_seed(args.run_seed, deterministic=deterministic)

    experiment_name = make_experiment_name(args.image_window, args.image_spec, args.return_horizon)
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
    loaders = build_dataloaders(prepared.datasets, config, shuffle_train=True)
    model = build_stock_cnn_for_window(config, args.image_window)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    run_context = {
        "stage": "stage2",
        "experiment_name": experiment_name,
        "image_window": int(args.image_window),
        "image_spec": str(args.image_spec),
        "return_horizon": int(args.return_horizon),
        "run_seed": int(args.run_seed),
        "device": device,
        "source_file": prepared.source_file,
        "num_parameters": count_parameters(model),
        "split_summary": split_summary(prepared.splits),
        "seed_info": seed_info,
    }

    result = fit_model(
        model,
        loaders["train"],
        loaders["validation"],
        config,
        device=device,
        checkpoint_dir=output_roots["checkpoint"],
        metrics_dir=output_roots["metrics"],
        run_context=run_context,
        normalization_metadata=prepared.normalization.as_dict(),
    )
    manifest = {
        "status": "ok",
        "run_context": run_context,
        "result": result.as_dict(),
        "normalization": prepared.normalization.as_dict(),
    }
    manifest_path = paths.run_manifest_root / experiment_name / f"seed_{args.run_seed}" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "run_manifest": str(manifest_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
