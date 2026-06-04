#!/usr/bin/env python
"""Check one Stage 4 experiment/seed output bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _stage4_script_utils import add_stage4_and_stage2_src_from_argv


add_stage4_and_stage2_src_from_argv(sys.argv)

from stage4_film import build_stage4_paths, ensure_stage4_output_dirs, load_config
from stage4_film.config import (
    CONTEXT_METHODS,
    get_context_config,
    get_stage4_model_config,
    stage4_run_context_base,
    validate_context_method,
)
from stage4_film.context import make_context_output_name
from stage4_film.paths import experiment_output_roots


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=0)
    parser.add_argument("--image-spec", default="")
    parser.add_argument("--return-horizon", type=int, default=0)
    parser.add_argument(
        "--context-method",
        default="concat",
        choices=list(CONTEXT_METHODS),
    )
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--gradcam-samples-per-class", type=int, default=2)
    parser.add_argument("--min-predictions", type=int, default=1)
    parser.add_argument("--write-summary", default="")
    return parser.parse_args()


def main() -> None:
    """Check files, parse small metadata, and fail if anything is missing."""

    args = parse_args()
    config = load_config(args.config)
    paths = build_stage4_paths(config)
    ensure_stage4_output_dirs(paths)

    stage4_model = get_stage4_model_config(config)
    context_config = get_context_config(config)
    image_window = int(args.image_window or stage4_model["primary_image_window"])
    image_spec = str(args.image_spec or stage4_model["primary_image_spec"])
    return_horizon = int(args.return_horizon or stage4_model["primary_return_horizon"])
    context_method = validate_context_method(args.context_method)
    run_seed = int(args.run_seed)
    split = str(args.split)
    context_window = int(context_config["context_window"])

    run_context = stage4_run_context_base(
        config,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_method=context_method,
        run_seed=run_seed,
    )
    experiment_name = str(run_context["stage4_experiment_name"])
    output_roots = experiment_output_roots(paths, experiment_name, run_seed)
    context_name = _resolve_context_name(
        context_config=context_config,
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_window=context_window,
    )
    context_root = paths.context_root / context_name / f"seed_{run_seed}"
    manifest_root = paths.run_manifest_root / experiment_name / f"seed_{run_seed}"
    gradcam_root = output_roots["figures"] / "gradcam" / split

    checks = {
        "best_checkpoint": _file_check(output_roots["checkpoint"] / "best.pt"),
        "last_checkpoint": _file_check(output_roots["checkpoint"] / "last.pt"),
        "train_history": _csv_check(output_roots["metrics"] / "train_history.csv"),
        "train_metadata": _json_check(output_roots["metrics"] / "train_metadata.json"),
        "predictions": _csv_check(
            output_roots["predictions"] / f"{split}_predictions.csv",
            min_rows=int(args.min_predictions),
        ),
        "classification_metrics": _json_check(output_roots["metrics"] / f"{split}_metrics.json"),
        "trading_metrics": _json_check(output_roots["metrics"] / f"{split}_trading_metrics.json"),
        "gradcam": _file_check(
            gradcam_root / f"btc_context_gradcam_{split}_{int(args.gradcam_samples_per_class)}perclass.png"
        ),
        "gradcam_samples": _csv_check(gradcam_root / "samples.csv", min_rows=1),
        "modulation_summary": _csv_check(gradcam_root / "modulation_summary.csv", min_rows=1),
        "modulation_values": _json_list_check(gradcam_root / "modulation_values.json", min_items=1),
        "context_features": _csv_check(context_root / "context_features.csv", min_rows=1),
        "context_scaler": _json_check(context_root / "context_scaler.json"),
        "context_feature_audit": _json_check(context_root / "context_feature_audit.json"),
        "context_feature_summary": _csv_check(context_root / "context_feature_summary.csv", min_rows=1),
        "run_manifest": _json_check(manifest_root / "run_manifest.json"),
    }

    status = "ok" if all(item["ok"] for item in checks.values()) else "missing"
    summary = {
        "status": status,
        "experiment_name": experiment_name,
        "context_method": context_method,
        "image_window": image_window,
        "image_spec": image_spec,
        "return_horizon": return_horizon,
        "context_window": context_window,
        "run_seed": run_seed,
        "split": split,
        "checks": checks,
    }
    if args.write_summary:
        output_path = Path(args.write_summary).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if status != "ok":
        raise SystemExit(1)


def _file_check(path: Path) -> dict[str, Any]:
    """Check that a file exists and is non-empty."""

    exists = path.exists()
    size_bytes = int(path.stat().st_size) if exists and path.is_file() else 0
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": size_bytes,
        "ok": bool(exists and path.is_file() and size_bytes > 0),
    }


def _csv_check(path: Path, min_rows: int = 0) -> dict[str, Any]:
    """Check that a CSV exists, parses, and has enough rows."""

    base = _file_check(path)
    if not base["ok"]:
        base.update({"rows": None, "columns": None, "parse_ok": False})
        return base
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        base.update({"rows": None, "columns": None, "parse_ok": False, "error": str(exc)})
        base["ok"] = False
        return base
    rows = int(len(frame))
    base.update(
        {
            "rows": rows,
            "columns": [str(column) for column in frame.columns],
            "parse_ok": True,
            "ok": rows >= int(min_rows),
        }
    )
    return base


def _json_check(path: Path) -> dict[str, Any]:
    """Check that a JSON file exists and parses."""

    base = _file_check(path)
    if not base["ok"]:
        base.update({"parse_ok": False})
        return base
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        base.update({"parse_ok": False, "error": str(exc)})
        base["ok"] = False
        return base
    base.update({"parse_ok": True, "json_type": type(value).__name__})
    return base


def _json_list_check(path: Path, min_items: int = 0) -> dict[str, Any]:
    """Check that a JSON file parses to a list with enough items."""

    base = _json_check(path)
    if not base.get("parse_ok"):
        return base
    value = json.loads(path.read_text(encoding="utf-8"))
    item_count = len(value) if isinstance(value, list) else None
    base.update(
        {
            "items": item_count,
            "ok": isinstance(value, list) and int(item_count or 0) >= int(min_items),
        }
    )
    return base


def _resolve_context_name(
    *,
    context_config: dict[str, Any],
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_window: int,
) -> str:
    """Resolve structured or prebuilt context artifact name."""

    source = str(context_config.get("source", "structured")).strip().lower()
    if source in {"prebuilt", "prebuilt_news", "news_prebuilt"}:
        explicit = str(
            context_config.get("prebuilt_context_name")
            or context_config.get("context_name")
            or ""
        ).strip()
        if not explicit:
            raise ValueError("Prebuilt context source requires context.prebuilt_context_name.")
        return explicit
    return make_context_output_name(
        image_window=image_window,
        image_spec=image_spec,
        return_horizon=return_horizon,
        context_window=context_window,
        context_suffix=str(context_config.get("feature_set_name", "")),
    )


if __name__ == "__main__":
    main()
