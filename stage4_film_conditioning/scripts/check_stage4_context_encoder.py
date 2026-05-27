"""Check the Stage 4 shared context MLP encoder."""

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

from stage4_film import build_stage4_paths, load_config
from stage4_film.conditions import build_context_encoder_from_config, count_parameters
from stage4_film.config import get_context_config, get_stage4_model_config
from stage4_film.context.features import make_context_output_name
from stage4_film.context.normalization import normalized_feature_columns
from stage4_film.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument(
        "--context-features",
        default="",
        help="Optional explicit context_features.csv path.",
    )
    return parser.parse_args()


def main() -> None:
    """Instantiate and validate the shared context encoder."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    set_global_seed(int(args.run_seed))

    context_config = get_context_config(config)
    stage4_model = get_stage4_model_config(config)
    features = [str(feature) for feature in context_config["primary_features"]]
    normalized_columns = normalized_feature_columns(features)

    encoder = build_context_encoder_from_config(config)
    context_dim = int(stage4_model["context_dim"])
    embedding_dim = int(stage4_model["context_embedding_dim"])
    batch_size = int(args.batch_size)
    dummy = torch.randn(batch_size, context_dim)

    encoder.train()
    train_embedding = encoder(dummy)
    encoder.eval()
    eval_embedding = encoder(dummy)
    shapes = encoder.forward_with_shapes(dummy)

    if tuple(eval_embedding.shape) != (batch_size, embedding_dim):
        raise RuntimeError(
            f"Unexpected encoder output shape: {tuple(eval_embedding.shape)}"
        )
    if not torch.isfinite(eval_embedding).all():
        raise RuntimeError("Context encoder produced non-finite values for dummy input.")

    context_file = _resolve_context_feature_file(
        args.context_features,
        paths,
        config,
        run_seed=int(args.run_seed),
    )
    context_file_check = _check_context_file(
        context_file,
        normalized_columns,
        encoder,
        batch_size=batch_size,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "encoder_spec": encoder.spec.as_dict(),
                "num_parameters": count_parameters(encoder),
                "num_trainable_parameters": count_parameters(encoder, trainable_only=True),
                "dummy_shapes": {
                    key: list(value)
                    for key, value in shapes.items()
                },
                "dummy_train_embedding_shape": list(train_embedding.shape),
                "dummy_eval_embedding_shape": list(eval_embedding.shape),
                "dummy_eval_embedding_finite": bool(torch.isfinite(eval_embedding).all().item()),
                "normalized_context_columns": normalized_columns,
                "context_file_check": context_file_check,
            },
            indent=2,
        )
    )


def _resolve_context_feature_file(
    explicit_path: str,
    paths: Any,
    config: dict[str, Any],
    *,
    run_seed: int,
) -> Path | None:
    """Find a context_features.csv file if one exists."""

    if explicit_path.strip():
        return Path(explicit_path).expanduser()

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    context_name = make_context_output_name(
        image_window=int(stage4_model["primary_image_window"]),
        image_spec=str(stage4_model["primary_image_spec"]),
        return_horizon=int(stage4_model["primary_return_horizon"]),
        context_window=int(context_config["context_window"]),
        context_suffix=str(context_config.get("feature_set_name", "")),
    )
    candidate = paths.context_root / context_name / f"seed_{int(run_seed)}" / "context_features.csv"
    if candidate.exists():
        return candidate
    return None


def _check_context_file(
    context_file: Path | None,
    normalized_columns: list[str],
    encoder: Any,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Run the encoder on real normalized context rows when available."""

    if context_file is None:
        return {
            "available": False,
            "reason": "context_features.csv not found; dummy tensor check was used",
        }
    if not context_file.exists():
        raise FileNotFoundError(f"Context feature file does not exist: {context_file}")

    frame = pd.read_csv(context_file, nrows=max(batch_size, 1))
    missing_columns = [column for column in normalized_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(
            "Context feature file missing normalized column(s): "
            + ", ".join(missing_columns)
        )
    context_tensor = torch.tensor(
        frame.loc[:, normalized_columns].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    encoder.eval()
    with torch.no_grad():
        embedding = encoder(context_tensor)
    if not torch.isfinite(embedding).all():
        raise RuntimeError("Context encoder produced non-finite values for real context rows.")
    return {
        "available": True,
        "path": str(context_file),
        "rows_checked": int(len(frame)),
        "input_shape": list(context_tensor.shape),
        "embedding_shape": list(embedding.shape),
        "embedding_finite": bool(torch.isfinite(embedding).all().item()),
    }


if __name__ == "__main__":
    main()
