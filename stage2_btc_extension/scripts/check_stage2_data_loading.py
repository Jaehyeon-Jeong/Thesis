#!/usr/bin/env python
"""2-I2: BTC OHLCV loader check."""

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
    from stage2_btc.data import find_btc_ohlcv_source, load_btc_ohlcv

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    source = find_btc_ohlcv_source(config, paths)
    frame = load_btc_ohlcv(source)
    result = {
        "status": "ok",
        "source_file": str(source),
        "num_rows": int(len(frame)),
        "date_min": str(frame["Date"].min().date()),
        "date_max": str(frame["Date"].max().date()),
        "columns": list(frame.columns),
        "head": frame.head(3).to_dict(orient="records"),
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
