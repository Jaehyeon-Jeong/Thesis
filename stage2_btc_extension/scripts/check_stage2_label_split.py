#!/usr/bin/env python
"""2-I4: BTC label/split/normalization check."""

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
    from stage2_btc import build_stage2_paths, load_config
    from stage2_btc.runners.btc_baseline import prepare_btc_experiment_data, split_summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    prepared = prepare_btc_experiment_data(
        config,
        paths,
        args.image_window,
        args.image_spec,
        args.return_horizon,
    )
    result = {
        "status": "ok",
        "source_file": prepared.source_file,
        "num_samples": int(len(prepared.samples)),
        "split_summary": split_summary(prepared.splits),
        "normalization": prepared.normalization.as_dict(),
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
