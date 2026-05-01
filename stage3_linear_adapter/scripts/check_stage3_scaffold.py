#!/usr/bin/env python
"""3-I1: Check Stage 3 config and Stage 2 dependency paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """Stage 3 실행 전에 필요한 config/source file/script 상태를 점검한다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import build_stage2_paths, load_config
    from stage2_btc.data import find_btc_ohlcv_source
    from stage3_linear.config import get_stage2_dependency_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    dependency = get_stage2_dependency_config(config)
    stage2_src = Path(str(dependency.get("src_path", stage_root.parent / "stage2_btc_extension" / "src")))
    source_file = find_btc_ohlcv_source(config, paths)
    required_scripts = [
        "run_stage3_linear.py",
        "evaluate_stage3_predictions.py",
        "evaluate_stage3_trading.py",
        "generate_stage3_gradcam.py",
        "check_stage3_outputs.py",
        "run_stage3_grid.py",
        "summarize_stage3_grid_results.py",
    ]
    checks = {
        "config": stage_root / args.config,
        "stage3_src": stage_root / "src",
        "stage2_src": stage2_src,
        "btc_source_file": source_file,
    }
    script_checks = {
        script: (stage_root / "scripts" / script).exists()
        for script in required_scripts
    }
    result = {
        "status": "ok" if all(path.exists() for path in checks.values()) and all(script_checks.values()) else "missing",
        "paths": {name: {"path": str(path), "exists": path.exists()} for name, path in checks.items()},
        "scripts": script_checks,
    }
    print(json.dumps(result, indent=2, default=str))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

