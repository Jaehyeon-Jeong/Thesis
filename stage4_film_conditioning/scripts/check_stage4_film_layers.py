"""Check Stage 4 FiLM layer and generator modules."""

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

from stage2_btc.models.stock_cnn import StockCNN, VARIANT_SPECS
from stage4_film import build_stage4_paths, load_config
from stage4_film.conditions import (
    build_context_encoder_from_config,
    build_film_generator_for_window,
    count_parameters as count_encoder_parameters,
    expected_film_generator_parameter_count,
)
from stage4_film.config import get_context_config, get_stage4_model_config
from stage4_film.context.features import make_context_output_name
from stage4_film.context.normalization import normalized_feature_columns
from stage4_film.layers import FeatureWiseAffineModulation
from stage4_film.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=0)
    parser.add_argument("--image-spec", default="")
    parser.add_argument("--return-horizon", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["film_gamma", "film_full"],
        choices=["film_gamma", "film_full"],
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--context-features", default="")
    return parser.parse_args()


def main() -> None:
    """Validate FiLM generators and the feature-wise affine layer."""

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
    batch_size = int(args.batch_size)

    spec = VARIANT_SPECS[image_window]
    context_encoder = build_context_encoder_from_config(config)
    film_layer = FeatureWiseAffineModulation()

    dummy_context = torch.randn(batch_size, int(stage4_model["context_dim"]))
    context_encoder.eval()
    with torch.no_grad():
        dummy_embedding = context_encoder(dummy_context)
    feature_maps = _make_stage2_feature_maps(spec, batch_size=batch_size)

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
        context_encoder,
        batch_size=batch_size,
    )

    mode_results: dict[str, Any] = {}
    for mode in args.modes:
        generator = build_film_generator_for_window(
            config,
            image_window=image_window,
            mode=mode,
        )
        generator.eval()
        with torch.no_grad():
            parameters = generator(dummy_embedding)
        mode_results[mode] = _check_mode(
            mode=mode,
            generator=generator,
            parameters=parameters,
            feature_maps=feature_maps,
            film_layer=film_layer,
            context_embedding=dummy_embedding,
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "image_window": image_window,
                "image_spec": image_spec,
                "return_horizon": return_horizon,
                "context_window": context_window,
                "spec": {
                    "stage2_variant": spec.name,
                    "input_shape": list(spec.input_shape),
                    "channels": list(spec.channels),
                },
                "context_encoder": {
                    "embedding_dim": int(stage4_model["context_embedding_dim"]),
                    "num_parameters": count_encoder_parameters(context_encoder),
                    "dummy_embedding_shape": list(dummy_embedding.shape),
                },
                "feature_map_shapes": [list(feature.shape) for feature in feature_maps],
                "modes": mode_results,
                "real_context_check": real_context_check,
            },
            indent=2,
        )
    )


def _make_stage2_feature_maps(spec: Any, *, batch_size: int) -> list[torch.Tensor]:
    """Run Stage 2 CNN blocks and collect post-block feature maps."""

    backbone = StockCNN(spec=spec, dropout=0.5)
    backbone.eval()
    image = torch.randn(batch_size, *spec.input_shape)
    feature_maps: list[torch.Tensor] = []
    with torch.no_grad():
        hidden = image
        for layer in backbone.layers:
            hidden = layer(hidden)
            feature_maps.append(hidden)
    return feature_maps


def _check_mode(
    *,
    mode: str,
    generator: Any,
    parameters: list[dict[str, torch.Tensor | None]],
    feature_maps: list[torch.Tensor],
    film_layer: FeatureWiseAffineModulation,
    context_embedding: torch.Tensor,
) -> dict[str, Any]:
    """Validate generated FiLM parameters and their application."""

    if len(parameters) != len(feature_maps):
        raise RuntimeError(
            f"{mode} block count mismatch: parameters={len(parameters)}, "
            f"feature_maps={len(feature_maps)}"
        )

    blocks: list[dict[str, Any]] = []
    for index, (block_params, feature_map) in enumerate(
        zip(parameters, feature_maps, strict=True),
        start=1,
    ):
        gamma = block_params["gamma"]
        beta = block_params["beta"]
        delta_gamma = block_params["delta_gamma"]
        if gamma is None or delta_gamma is None:
            raise RuntimeError(f"{mode} block {index} missing gamma or delta_gamma")
        expected_shape = (feature_map.shape[0], feature_map.shape[1])
        _assert_shape(f"{mode} block{index} gamma", gamma.shape, expected_shape)
        _assert_shape(f"{mode} block{index} delta_gamma", delta_gamma.shape, expected_shape)
        if mode == "film_gamma":
            if beta is not None:
                raise RuntimeError("film_gamma mode must not emit beta")
        else:
            if beta is None:
                raise RuntimeError("film_full mode must emit beta")
            _assert_shape(f"{mode} block{index} beta", beta.shape, expected_shape)

        output = film_layer(feature_map, gamma, beta)
        _assert_shape(f"{mode} block{index} output", output.shape, tuple(feature_map.shape))
        identity_ok = torch.allclose(output, feature_map)
        if not identity_ok:
            raise RuntimeError(f"{mode} block{index} is not identity at initialization")

        block_summary: dict[str, Any] = {
            "block": index,
            "feature_map_shape": list(feature_map.shape),
            "gamma_shape": list(gamma.shape),
            "delta_gamma_shape": list(delta_gamma.shape),
            "gamma_min": float(gamma.min().item()),
            "gamma_max": float(gamma.max().item()),
            "delta_gamma_min": float(delta_gamma.min().item()),
            "delta_gamma_max": float(delta_gamma.max().item()),
            "output_shape": list(output.shape),
            "identity_at_initialization": bool(identity_ok),
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

    expected_params = expected_film_generator_parameter_count(
        context_embedding_dim=int(context_embedding.shape[1]),
        block_channels=[int(feature.shape[1]) for feature in feature_maps],
        mode=mode,
    )
    actual_params = count_encoder_parameters(generator)
    if actual_params != expected_params:
        raise RuntimeError(
            f"{mode} generator parameter mismatch: actual={actual_params}, "
            f"expected={expected_params}"
        )
    return {
        "generator_spec": generator.spec.as_dict(),
        "parameter_count": {
            "actual": actual_params,
            "expected": expected_params,
        },
        "generator_shapes": generator.forward_with_shapes(context_embedding),
        "blocks": blocks,
    }


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
    )
    candidate = paths.context_root / context_name / f"seed_{int(run_seed)}" / "context_features.csv"
    if candidate.exists():
        return candidate
    return None


def _check_real_context_rows(
    context_file: Path | None,
    normalized_columns: list[str],
    context_encoder: Any,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Run the context encoder on real normalized rows when available."""

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
    context_encoder.eval()
    with torch.no_grad():
        embedding = context_encoder(context)
    if not torch.isfinite(embedding).all():
        raise RuntimeError("Context encoder produced non-finite real-row embedding.")
    return {
        "available": True,
        "path": str(context_file),
        "rows_checked": int(len(frame)),
        "context_shape": list(context.shape),
        "embedding_shape": list(embedding.shape),
        "embedding_finite": bool(torch.isfinite(embedding).all().item()),
    }


def _assert_shape(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    """Raise a clear error when a tensor shape differs from expected."""

    actual_tuple = tuple(int(value) for value in actual)
    if actual_tuple != expected:
        raise RuntimeError(f"{name} shape mismatch: actual={actual_tuple}, expected={expected}")


if __name__ == "__main__":
    main()
