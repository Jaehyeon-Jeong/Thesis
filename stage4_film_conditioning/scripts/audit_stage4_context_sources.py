"""Audit Stage 4 BTC and Fear & Greed context sources.

This script checks only data availability and leakage policy. It does not train
or evaluate a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage2_btc.data import build_btc_samples, build_btc_splits, find_btc_ohlcv_source
from stage2_btc.data import load_btc_ohlcv
from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import get_context_config, get_stage4_model_config
from stage4_film.context import audit_context_sources, find_fear_greed_source
from stage4_film.context import load_fear_greed_index


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=None)
    parser.add_argument("--return-horizon", type=int, default=None)
    parser.add_argument(
        "--output",
        default="reports/tables/stage4_context_source_audit.json",
        help="Output JSON path relative to the Stage 4 project root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the source audit and write a JSON report."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_window = int(context_config["context_window"])

    btc_source = find_btc_ohlcv_source(config, paths)
    ohlcv = load_btc_ohlcv(btc_source)
    samples = build_btc_samples(
        ohlcv,
        config,
        image_window=image_window,
        return_horizon=return_horizon,
    )
    splits = build_btc_splits(samples, config)

    fg_source = find_fear_greed_source(config, paths)
    fg_config = context_config.get("fear_greed", {})
    fear_greed = load_fear_greed_index(
        fg_source,
        date_column=str(fg_config.get("date_column", "date")),
        value_column=str(fg_config.get("value_column", "value")),
        classification_column=str(fg_config.get("classification_column", "classification")),
    )

    audit = audit_context_sources(
        ohlcv=ohlcv,
        samples_by_split=splits,
        fear_greed=fear_greed,
        context_window=context_window,
        btc_source_file=btc_source,
        fear_greed_source_file=fg_source,
    )
    audit["stage4_run"] = {
        "image_window": image_window,
        "return_horizon": return_horizon,
        "context_window": context_window,
        "primary_features": [str(feature) for feature in context_config["primary_features"]],
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = paths.project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "image_window": image_window,
                "return_horizon": return_horizon,
                "context_window": context_window,
                "btc_source": str(btc_source),
                "fear_greed_source": str(fg_source),
                "split_counts": {
                    split: int(len(frame))
                    for split, frame in splits.items()
                },
                "written": str(output_path),
            },
            indent=2,
        )
    )


def _jsonable(value: Any) -> Any:
    """Convert common pandas/numpy scalar values to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    main()
