#!/usr/bin/env python
"""3-I3: Train one Stage 3 BTC CNN + Linear adapter experiment."""

from __future__ import annotations

import argparse
import json
import sys

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """Stage 2 data pipeline을 그대로 사용해 Stage 3 Linear model을 학습한다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import (
        build_stage2_paths,
        experiment_output_roots,
        load_config,
        select_device,
        set_global_seed,
    )
    from stage2_btc.runners.btc_baseline import (
        build_dataloaders,
        prepare_btc_experiment_data,
        split_summary,
    )
    from stage2_btc.training import fit_model
    from stage3_linear import make_stage3_experiment_name
    from stage3_linear.config import stage3_run_context_base
    from stage3_linear.models import build_linear_stock_cnn_for_window, count_parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=60)
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

    run_context = stage3_run_context_base(
        config,
        args.image_window,
        args.image_spec,
        args.return_horizon,
        args.run_seed,
    )
    experiment_name = run_context["stage3_experiment_name"]
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
    model = build_linear_stock_cnn_for_window(config, args.image_window)
    output_roots = experiment_output_roots(paths, experiment_name, args.run_seed)
    run_context.update(
        {
            "experiment_name": experiment_name,
            "device": str(device),
            "source_file": prepared.source_file,
            "num_parameters": count_parameters(model),
            "split_summary": split_summary(prepared.splits),
            "seed_info": seed_info,
        }
    )

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

