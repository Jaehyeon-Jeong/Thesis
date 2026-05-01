#!/usr/bin/env python
"""3-I2: Check Stage 3 Linear adapter model shapes and parameter counts."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """Linear adapter model이 계획한 tensor shape와 parameter count를 갖는지 확인한다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import load_config
    from stage2_btc.models.stock_cnn import VARIANT_SPECS
    from stage3_linear import get_linear_adapter_config, stage3_expected_parameter_count
    from stage3_linear.models import build_linear_stock_cnn_for_window, count_parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    model = build_linear_stock_cnn_for_window(config, args.image_window)
    spec = VARIANT_SPECS[int(args.image_window)]
    dummy = torch.zeros(int(args.batch_size), *spec.input_shape)
    shapes = model.forward_with_shapes(dummy)
    result = {
        "status": "ok",
        "image_window": int(args.image_window),
        "adapter_config": dict(get_linear_adapter_config(config)),
        "stage2_baseline_expected_params": int(spec.expected_num_params),
        "stage3_linear_expected_params": stage3_expected_parameter_count(config, args.image_window),
        "stage3_linear_actual_params": count_parameters(model),
        "shapes": {key: list(value) for key, value in shapes.items()},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

