"""Run one Stage 4 context-conditioned BTC experiment.

This is the 4-I8 runner: it reuses Stage 2 BTC data/image/split/normalization,
builds Stage 4 context features, attaches context tensors to each batch, and
trains one of:

    concat, gating, film_gamma, film_full
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_context_config, get_stage4_model_config, validate_context_method
from stage4_film.runners import run_stage4_context_training
from stage4_film.runtime import select_device
from stage4_film.seed import set_global_seed


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
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run one Stage 4 context-conditioned training job."""

    args = parse_args()
    config = load_config(args.config)
    config = _apply_cli_overrides(config, args)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_method = validate_context_method(args.context_method)
    context_window = int(context_config["context_window"])

    training_config = config["training"]
    deterministic = bool(training_config.get("determinism", {}).get("enabled", True))
    seed_info = set_global_seed(int(args.run_seed), deterministic=deterministic)
    _apply_torch_determinism_flags(training_config)
    device = select_device(config)

    manifest = run_stage4_context_training(
        config=config,
        paths=paths,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_method=context_method,
        run_seed=int(args.run_seed),
        device=device,
        seed_info=seed_info,
        max_train_rows=_optional_positive_int(args.max_train_rows),
        max_validation_rows=_optional_positive_int(args.max_validation_rows),
        max_test_rows=_optional_positive_int(args.max_test_rows),
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_name": manifest["run_context"]["stage4_experiment_name"],
                "context_method": context_method,
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "context_window": context_window,
                "run_seed": int(args.run_seed),
                "device": device,
                "num_parameters": manifest["run_context"]["num_parameters"],
                "split_summary": manifest["run_context"]["split_summary"],
                "context_audit_warnings": manifest["run_context"]["context_audit"].get(
                    "warnings",
                    [],
                ),
                "result": manifest["result"],
                "normalization": manifest["normalization"],
                "context_artifacts": manifest["context"]["artifacts"],
                "run_manifest": manifest["run_manifest"],
            },
            indent=2,
        ),
        flush=True,
    )


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply non-persistent CLI overrides to the loaded config dictionary."""

    if int(args.max_epochs) > 0:
        config["training"]["max_epochs"] = int(args.max_epochs)
    return config


def _optional_positive_int(value: int) -> int | None:
    """Convert CLI row limit values to optional positive integers."""

    integer = int(value)
    return integer if integer > 0 else None


def _apply_torch_determinism_flags(training_config: dict[str, Any]) -> None:
    """Apply cudnn deterministic/benchmark flags from config when CUDA exists."""

    determinism = training_config.get("determinism", {})
    if not isinstance(determinism, dict):
        return
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(
            determinism.get("cudnn_deterministic", True)
        )
        torch.backends.cudnn.benchmark = bool(determinism.get("cudnn_benchmark", False))


if __name__ == "__main__":
    main()
