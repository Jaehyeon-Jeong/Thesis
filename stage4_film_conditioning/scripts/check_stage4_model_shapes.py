"""Check Stage 4 context-conditioned model tensor shapes and parameter counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage2_btc.models.stock_cnn import VARIANT_SPECS, count_parameters
from stage4_film import build_stage4_paths, load_config
from stage4_film.config import (
    get_context_config,
    get_stage4_model_config,
    make_stage4_experiment_name,
    validate_context_method,
)
from stage4_film.context.features import make_context_output_name
from stage4_film.context.normalization import normalized_feature_columns
from stage4_film.models import (
    build_concat_context_stock_cnn_for_window,
    build_film_context_stock_cnn_for_window,
    build_gated_context_stock_cnn_for_window,
    expected_concat_context_parameter_count,
    expected_film_context_parameter_count,
    expected_gated_context_parameter_count,
)
from stage4_film.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument(
        "--model",
        default="concat",
        choices=["concat", "gating", "film_gamma", "film_full"],
    )
    parser.add_argument("--image-window", type=int, default=0)
    parser.add_argument("--image-spec", default="")
    parser.add_argument("--return-horizon", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument(
        "--context-features",
        default="",
        help="Optional explicit context_features.csv path.",
    )
    return parser.parse_args()


def main() -> None:
    """Instantiate the model and validate dummy plus real-context paths."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    set_global_seed(int(args.run_seed))

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_window = int(args.context_window or context_config["context_window"])
    method = validate_context_method(str(args.model))
    batch_size = int(args.batch_size)

    if image_window not in VARIANT_SPECS:
        raise KeyError(f"Unsupported image_window: {image_window}")

    spec = VARIANT_SPECS[image_window]
    if method == "concat":
        model = build_concat_context_stock_cnn_for_window(config, image_window=image_window)
        expected_params = expected_concat_context_parameter_count(config, image_window)
    elif method == "gating":
        model = build_gated_context_stock_cnn_for_window(config, image_window=image_window)
        expected_params = expected_gated_context_parameter_count(config, image_window)
    elif method in {"film_gamma", "film_full"}:
        model = build_film_context_stock_cnn_for_window(
            config,
            image_window=image_window,
            mode=method,
        )
        expected_params = expected_film_context_parameter_count(
            config,
            image_window,
            method,
        )
    else:
        raise ValueError(f"Unsupported model checker method: {method}")
    model.eval()

    context_dim = int(stage4_model["context_dim"])
    embedding_dim = int(stage4_model["context_embedding_dim"])
    dummy_image = torch.randn(batch_size, *spec.input_shape)
    dummy_context = torch.randn(batch_size, context_dim)
    shapes = model.forward_with_shapes(dummy_image, dummy_context)
    with torch.no_grad():
        logits = model(dummy_image, dummy_context)

    _assert_shape("logits", logits.shape, (batch_size, 2))
    _assert_shape("context_embedding", shapes["context_embedding"], (batch_size, embedding_dim))
    _assert_shape("flatten", shapes["flatten"], (batch_size, spec.expected_flatten_dim))
    if method == "concat":
        _assert_shape(
            "concat_feature",
            shapes["concat_feature"],
            (batch_size, spec.expected_flatten_dim + embedding_dim),
        )
    elif method == "gating":
        final_channels = int(spec.channels[-1])
        _assert_shape("raw_gate", shapes["raw_gate"], (batch_size, final_channels))
        _assert_shape("gate", shapes["gate"], (batch_size, final_channels))
        _assert_shape("gated_feature_map", shapes["gated_feature_map"], shapes["final_feature_map"])
    elif method in {"film_gamma", "film_full"}:
        for block_index, channels in enumerate(spec.channels, start=1):
            _assert_shape(
                f"gamma{block_index}",
                shapes[f"gamma{block_index}"],
                (batch_size, int(channels)),
            )
            if method == "film_full":
                _assert_shape(
                    f"beta{block_index}",
                    shapes[f"beta{block_index}"],
                    (batch_size, int(channels)),
                )
            _assert_shape(
                f"layer{block_index}_film",
                shapes[f"layer{block_index}_film"],
                shapes[f"layer{block_index}_bn"],
            )
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{method} context model produced non-finite dummy logits.")

    actual_params = count_parameters(model)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Parameter mismatch: actual={actual_params}, expected={expected_params}"
        )

    features = [str(feature) for feature in context_config["primary_features"]]
    normalized_columns = normalized_feature_columns(features)
    context_file = _resolve_context_feature_file(
        args.context_features,
        paths,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_window=context_window,
        run_seed=int(args.run_seed),
    )
    real_context_check = _check_real_context_rows(
        context_file,
        normalized_columns,
        model,
        spec,
        method=method,
        batch_size=batch_size,
    )

    gate_identity_check = None
    film_identity_check = None
    if method == "gating":
        with torch.no_grad():
            details = model.forward_with_gate_values(dummy_image, dummy_context)
        gate = details["gate"]
        gate_identity_check = {
            "raw_gate_shape": list(details["raw_gate"].shape),
            "gate_shape": list(gate.shape),
            "gate_min": float(gate.min().item()),
            "gate_max": float(gate.max().item()),
            "allclose_to_one_at_initialization": bool(
                torch.allclose(gate, torch.ones_like(gate))
            ),
        }
    if method in {"film_gamma", "film_full"}:
        with torch.no_grad():
            details = model.forward_with_modulation_values(
                dummy_image,
                dummy_context,
                keep_feature_maps=True,
            )
        blocks = []
        for index, block in enumerate(details["film_parameters"], start=1):
            gamma = block["gamma"]
            beta = block["beta"]
            pre_film = details["pre_film_feature_maps"][index - 1]
            post_film = details["post_film_feature_maps"][index - 1]
            block_summary = {
                "block": index,
                "gamma_shape": list(gamma.shape),
                "gamma_min": float(gamma.min().item()),
                "gamma_max": float(gamma.max().item()),
                "delta_gamma_shape": list(block["delta_gamma"].shape),
                "delta_gamma_min": float(block["delta_gamma"].min().item()),
                "delta_gamma_max": float(block["delta_gamma"].max().item()),
                "pre_film_shape": list(pre_film.shape),
                "post_film_shape": list(post_film.shape),
                "post_film_equals_pre_film_at_initialization": bool(
                    torch.allclose(post_film, pre_film)
                ),
            }
            if beta is not None:
                block_summary.update(
                    {
                        "beta_shape": list(beta.shape),
                        "beta_min": float(beta.min().item()),
                        "beta_max": float(beta.max().item()),
                    }
                )
            blocks.append(block_summary)
        film_identity_check = {
            "mode": method,
            "blocks": blocks,
            "all_blocks_identity_at_initialization": all(
                block["post_film_equals_pre_film_at_initialization"] for block in blocks
            ),
        }

    experiment_name = make_stage4_experiment_name(
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_method=method,
        context_window=context_window,
        experiment_suffix=str(get_context_config(config).get("feature_set_name", "")),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_name": experiment_name,
                "model": method,
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "context_window": context_window,
                "spec": {
                    "stage2_variant": spec.name,
                    "input_shape": list(spec.input_shape),
                    "channels": list(spec.channels),
                    "expected_flatten_dim": spec.expected_flatten_dim,
                    "stage2_expected_num_params": spec.expected_num_params,
                },
                "context": {
                    "features": features,
                    "normalized_columns": normalized_columns,
                    "context_dim": context_dim,
                    "embedding_dim": embedding_dim,
                },
                "parameter_count": {
                    "actual": actual_params,
                    "expected": expected_params,
                    "delta_vs_stage2_baseline": actual_params - spec.expected_num_params,
                },
                "dummy_shapes": {key: list(value) for key, value in shapes.items()},
                "dummy_logits_shape": list(logits.shape),
                "dummy_logits_finite": bool(torch.isfinite(logits).all().item()),
                "gate_identity_check": gate_identity_check,
                "film_identity_check": film_identity_check,
                "real_context_check": real_context_check,
            },
            indent=2,
        )
    )


def _assert_shape(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    """Raise a clear error when a tensor shape differs from expected."""

    actual_tuple = tuple(int(value) for value in actual)
    if actual_tuple != expected:
        raise RuntimeError(f"{name} shape mismatch: actual={actual_tuple}, expected={expected}")


def _resolve_context_feature_file(
    explicit_path: str,
    paths: Any,
    *,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_window: int,
    run_seed: int,
) -> Path | None:
    """Find a context_features.csv file if available."""

    if explicit_path.strip():
        return Path(explicit_path).expanduser()
    context_name = make_context_output_name(
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_window=context_window,
        context_suffix=str(get_context_config(config).get("feature_set_name", "")),
    )
    candidate = paths.context_root / context_name / f"seed_{int(run_seed)}" / "context_features.csv"
    if candidate.exists():
        return candidate
    return None


def _check_real_context_rows(
    context_file: Path | None,
    normalized_columns: list[str],
    model: Any,
    spec: Any,
    *,
    method: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run the full model with real normalized context rows when available."""

    if context_file is None:
        return {
            "available": False,
            "reason": "context_features.csv not found; dummy tensor check was used",
        }
    if not context_file.exists():
        raise FileNotFoundError(f"Context feature file does not exist: {context_file}")

    frame = pd.read_csv(context_file, nrows=max(batch_size, 1))
    missing = [column for column in normalized_columns if column not in frame.columns]
    if missing:
        raise KeyError(
            "Context feature file missing normalized column(s): " + ", ".join(missing)
        )

    context = torch.tensor(
        frame.loc[:, normalized_columns].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    image = torch.zeros(len(context), *spec.input_shape)
    model.eval()
    with torch.no_grad():
        logits = model(image, context)
        shapes = model.forward_with_shapes(image, context)
        gate_summary = None
        film_summary = None
        if method == "gating":
            details = model.forward_with_gate_values(image, context)
            gate = details["gate"]
            gate_summary = {
                "shape": list(gate.shape),
                "min": float(gate.min().item()),
                "max": float(gate.max().item()),
                "allclose_to_one_at_initialization": bool(
                    torch.allclose(gate, torch.ones_like(gate))
                ),
            }
        if method in {"film_gamma", "film_full"}:
            details = model.forward_with_modulation_values(
                image,
                context,
                keep_feature_maps=False,
            )
            blocks = []
            for block in details["film_parameters"]:
                gamma = block["gamma"]
                beta = block["beta"]
                block_summary = {
                    "block": int(block["block"]),
                    "gamma_shape": list(gamma.shape),
                    "gamma_min": float(gamma.min().item()),
                    "gamma_max": float(gamma.max().item()),
                    "delta_gamma_min": float(block["delta_gamma"].min().item()),
                    "delta_gamma_max": float(block["delta_gamma"].max().item()),
                    "identity_at_initialization": bool(block["identity_at_initialization"]),
                }
                if beta is not None:
                    block_summary.update(
                        {
                            "beta_shape": list(beta.shape),
                            "beta_min": float(beta.min().item()),
                            "beta_max": float(beta.max().item()),
                        }
                    )
                blocks.append(block_summary)
            film_summary = {
                "mode": method,
                "blocks": blocks,
                "all_blocks_identity_at_initialization": all(
                    block["identity_at_initialization"] for block in blocks
                ),
            }
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite logits for real context rows.")

    result = {
        "available": True,
        "path": str(context_file),
        "rows_checked": int(len(frame)),
        "context_shape": list(context.shape),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
    }
    if method == "concat":
        result["concat_feature_shape"] = list(shapes["concat_feature"])
    if method == "gating":
        result["gated_feature_map_shape"] = list(shapes["gated_feature_map"])
        result["gate"] = gate_summary
    if method in {"film_gamma", "film_full"}:
        result["film"] = film_summary
    return result


if __name__ == "__main__":
    main()
