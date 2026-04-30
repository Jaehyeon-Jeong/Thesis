#!/usr/bin/env python3
"""Smoke-check the Stage 1 StockCNNI20 model implementation.

This script verifies the model contract without loading real data:
random input `(batch_size,1,64,60)` -> logits `(batch_size,2)`, expected
intermediate shapes, expected parameter count, and Grad-CAM target layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_stage1_src_to_path() -> Path:
    """Add local Stage 1 `src/` directory to `sys.path`."""

    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def parse_args(stage_root: Path) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=stage_root / "configs" / "env_local.yaml",
        help="Stage 1 environment config path.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run model shape and parameter-count checks."""

    add_stage1_src_to_path()
    args = parse_args(Path(__file__).resolve().parents[1])

    import torch  # pylint: disable=import-outside-toplevel

    from stage1_reimage.config import load_config  # pylint: disable=import-outside-toplevel
    from stage1_reimage.models import (  # pylint: disable=import-outside-toplevel
        EXPECTED_I20_PARAMETER_COUNT,
        build_stock_cnn_i20_from_config,
        count_parameters,
    )
    from stage1_reimage.seed import set_global_seed  # pylint: disable=import-outside-toplevel

    config = load_config(args.config)
    set_global_seed(args.seed)
    model = build_stock_cnn_i20_from_config(config)
    model.eval()

    # Synthetic batch with the same shape as real I20 images after DataLoader
    # collation: `(B, channel=1, height=64, width=60)`.
    sample = torch.rand(args.batch_size, 1, 64, 60, dtype=torch.float32)
    with torch.no_grad():
        # Forward pass returns raw class scores, not probabilities.
        logits = model(sample)
        softmax_of_logits = torch.softmax(logits, dim=1)

    parameter_count = count_parameters(model)
    trainable_parameter_count = count_parameters(model, trainable_only=True)
    model_config = config["model"]
    expected_parameter_count = int(
        model_config.get("expected_parameter_count", EXPECTED_I20_PARAMETER_COUNT)
    )
    if parameter_count != expected_parameter_count:
        raise AssertionError(
            f"Parameter count mismatch: expected={expected_parameter_count}, "
            f"actual={parameter_count}"
        )

    # `forward_with_shapes` runs the same model path and records the tensor
    # shape after each block so architecture mistakes are caught early.
    shapes = model.forward_with_shapes(sample)
    expected_shapes = {
        "input": (args.batch_size, 1, 64, 60),
        "layer1": (args.batch_size, 64, 13, 60),
        "layer2": (args.batch_size, 128, 5, 60),
        "layer3": (args.batch_size, 256, 3, 60),
        "flatten": (args.batch_size, 46_080),
        "logits": (args.batch_size, 2),
    }
    for name, expected_shape in expected_shapes.items():
        if shapes[name] != expected_shape:
            raise AssertionError(
                f"Shape mismatch for {name}: expected={expected_shape}, actual={shapes[name]}"
            )

    gradcam_layers = model.gradcam_target_layers()
    summary = {
        "status": "ok",
        "config_path": str(args.config),
        "model_name": model_config["name"],
        "reference_repo": model_config["reference_repo"],
        "reference_commit": model_config["reference_commit"],
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "shapes": {key: list(value) for key, value in shapes.items()},
        "logits_shape": list(logits.shape),
        "softmax_row_sums": [float(value) for value in softmax_of_logits.sum(dim=1)],
        "forward_returns_logits": True,
        "gradcam_target_layers": {
            name: layer.__class__.__name__ for name, layer in gradcam_layers.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
