#!/usr/bin/env python
"""2-I3: BTC sample image generation check.

생성 image 위치:
    기본값은 `reports/figures/sample_images/`다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def add_src_to_path() -> Path:
    stage_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage_root / "src"))
    return stage_root


def main() -> None:
    stage_root = add_src_to_path()
    from stage2_btc import build_stage2_paths, load_config
    from stage2_btc.data import build_btc_samples, find_btc_ohlcv_source, load_btc_ohlcv
    from stage2_btc.data.label_split import add_moving_average_column, generate_image_for_sample

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=20)
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/figures/sample_images")
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    source = find_btc_ohlcv_source(config, paths)
    ohlcv = load_btc_ohlcv(source)
    ohlcv = add_moving_average_column(ohlcv, args.image_window)
    samples = build_btc_samples(ohlcv, config, args.image_window, args.return_horizon)
    sample = samples.iloc[0]

    output_dir = stage_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for image_spec in ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]:
        image = generate_image_for_sample(ohlcv, sample, config, args.image_window, image_spec)
        output_path = output_dir / f"btc_i{args.image_window}_{image_spec}_sample.png"
        plt.imsave(output_path, image, cmap="gray", vmin=0, vmax=255)
        written[image_spec] = str(output_path)

    result = {
        "status": "ok",
        "image_window": args.image_window,
        "return_horizon": args.return_horizon,
        "sample_date": str(sample["Date"].date()),
        "output_dir": str(output_dir),
        "written": written,
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
