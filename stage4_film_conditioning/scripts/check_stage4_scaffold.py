#!/usr/bin/env python
"""4-I1: Check Stage 4 config, path scaffold, and Stage 2 dependency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


def _path_status(path: Path | None) -> dict[str, object]:
    """JSON에 넣기 쉬운 path 상태 dictionary를 만든다."""

    if path is None:
        return {"path": None, "exists": False}
    return {"path": str(path), "exists": path.exists()}


def main() -> None:
    """Stage 4 구현 전에 필요한 config/source file/script 상태를 점검한다."""

    stage_root = add_stage4_and_stage2_src_from_argv(sys.argv)

    from stage2_btc.data import find_btc_ohlcv_source
    from stage4_film import build_stage4_paths, load_config
    from stage4_film.config import (
        get_context_config,
        get_primary_context_features,
        get_stage2_dependency_config,
        get_stage4_model_config,
        make_stage4_experiment_name,
    )
    from stage4_film.paths import ensure_stage4_output_dirs
    from stage4_film.runtime import select_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create Stage 4 output directories while checking the scaffold.",
    )
    args = parser.parse_args()

    config_path = stage_root / args.config
    config = load_config(config_path)
    paths = build_stage4_paths(config)
    dependency = get_stage2_dependency_config(config)
    context_config = get_context_config(config)
    stage4_model = get_stage4_model_config(config)
    primary_features = get_primary_context_features(config)

    stage2_src = Path(str(dependency.get("src_path"))).expanduser()
    btc_source_file = find_btc_ohlcv_source(config, paths)

    fear_greed_file = paths.fear_greed_file
    if fear_greed_file is None:
        fear_greed_config = context_config.get("fear_greed", {})
        for key in ("local_file", "kaggle_file"):
            candidate = str(fear_greed_config.get(key, "")).strip()
            if candidate:
                fear_greed_file = Path(candidate).expanduser()
                break

    if args.create_output_dirs:
        ensured_dirs = ensure_stage4_output_dirs(paths)
    else:
        ensured_dirs = []

    required_files = {
        "config_py": stage_root / "src" / "stage4_film" / "config.py",
        "paths_py": stage_root / "src" / "stage4_film" / "paths.py",
        "runtime_py": stage_root / "src" / "stage4_film" / "runtime.py",
        "seed_py": stage_root / "src" / "stage4_film" / "seed.py",
        "script_utils": stage_root / "scripts" / "_stage4_script_utils.py",
        "this_checker": stage_root / "scripts" / "check_stage4_scaffold.py",
    }
    required_dirs = {
        "stage4_src": stage_root / "src",
        "stage4_package": stage_root / "src" / "stage4_film",
        "stage2_src": stage2_src,
        "scripts": stage_root / "scripts",
        "configs": stage_root / "configs",
        "reports_tables": stage_root / "reports" / "tables",
    }

    primary_window = int(stage4_model["primary_image_window"])
    primary_horizon = int(stage4_model["primary_return_horizon"])
    primary_spec = str(stage4_model["primary_image_spec"])
    context_window = int(context_config["context_window"])
    method_names = list(stage4_model["context_methods"])
    experiment_names = [
        make_stage4_experiment_name(
            primary_window,
            primary_spec,
            primary_horizon,
            method,
            context_window,
            experiment_suffix=str(context_config.get("feature_set_name", "")),
        )
        for method in method_names
    ]

    checks = {
        "config": _path_status(config_path),
        "btc_source_file": _path_status(btc_source_file),
        "fear_greed_file": _path_status(fear_greed_file),
        "stage2_src": _path_status(stage2_src),
        "required_files": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required_files.items()
        },
        "required_dirs": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required_dirs.items()
        },
        "created_output_dirs": [str(path) for path in ensured_dirs],
    }

    status_ok = (
        checks["config"]["exists"]
        and checks["btc_source_file"]["exists"]
        and checks["stage2_src"]["exists"]
        and all(item["exists"] for item in checks["required_files"].values())
        and all(item["exists"] for item in checks["required_dirs"].values())
    )

    result = {
        "status": "ok" if status_ok else "missing",
        "stage": "stage4",
        "config_path": str(config_path),
        "device": select_device(config),
        "primary_experiment": {
            "image_window": primary_window,
            "image_spec": primary_spec,
            "return_horizon": primary_horizon,
            "context_window": context_window,
            "context_methods": method_names,
            "experiment_names": experiment_names,
        },
        "context": {
            "primary_features": primary_features,
            "context_dim": int(stage4_model["context_dim"]),
            "context_embedding_dim": int(stage4_model["context_embedding_dim"]),
        },
        "paths": paths.as_dict(),
        "checks": checks,
    }
    print(json.dumps(result, indent=2, default=str))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
