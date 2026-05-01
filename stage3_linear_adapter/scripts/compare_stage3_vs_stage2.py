#!/usr/bin/env python
"""Compare Stage 3 Linear metrics with Stage 2 baseline metrics.

역할:
    같은 `image_window / return_horizon / image_spec / seed` 조합에서 Stage 2와
    Stage 3 결과가 모두 있으면 metric 차이를 CSV로 저장한다. 결과가 없으면 missing
    상태를 기록하고 학습을 다시 실행하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _stage3_script_utils import add_stage3_and_stage2_src_from_argv


def main() -> None:
    """Stage 2/Stage 3 metric JSON을 join해서 비교 table을 만든다."""

    stage_root = add_stage3_and_stage2_src_from_argv(sys.argv)
    from stage2_btc import build_stage2_paths, load_config
    from stage3_linear.config import (
        get_stage2_dependency_config,
        make_stage2_baseline_experiment_name,
        stage3_run_context_base,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-windows", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--return-horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--image-specs", nargs="+", default=["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"])
    parser.add_argument("--run-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-name", default="stage3_vs_stage2_comparison.csv")
    args = parser.parse_args()

    config = load_config(stage_root / args.config)
    paths = build_stage2_paths(config)
    dependency = get_stage2_dependency_config(config)
    stage2_root = Path(str(dependency.get("baseline_output_root", ""))).expanduser()
    rows: list[dict[str, Any]] = []

    for seed in args.run_seeds:
        for image_window in args.image_windows:
            for return_horizon in args.return_horizons:
                for image_spec in args.image_specs:
                    stage3_context = stage3_run_context_base(config, image_window, image_spec, return_horizon, seed)
                    stage3_name = stage3_context["stage3_experiment_name"]
                    stage2_name = make_stage2_baseline_experiment_name(image_window, image_spec, return_horizon)
                    seed_name = f"seed_{seed}"
                    stage3_metrics_path = paths.metrics_root / stage3_name / seed_name / f"{args.split}_metrics.json"
                    stage2_metrics_path = stage2_root / "metrics" / stage2_name / seed_name / f"{args.split}_metrics.json"
                    stage3_metrics = _read_json(stage3_metrics_path)
                    stage2_metrics = _read_json(stage2_metrics_path)
                    row = {
                        "image_window": image_window,
                        "return_horizon": return_horizon,
                        "image_spec": image_spec,
                        "run_seed": seed,
                        "stage2_experiment_name": stage2_name,
                        "stage3_experiment_name": stage3_name,
                        "stage2_available": stage2_metrics is not None,
                        "stage3_available": stage3_metrics is not None,
                    }
                    for metric_name in ("accuracy", "roc_auc", "f1", "brier_score", "log_loss"):
                        s2 = None if stage2_metrics is None else stage2_metrics.get(metric_name)
                        s3 = None if stage3_metrics is None else stage3_metrics.get(metric_name)
                        row[f"stage2_{metric_name}"] = s2
                        row[f"stage3_{metric_name}"] = s3
                        row[f"delta_{metric_name}"] = None if s2 is None or s3 is None else s3 - s2
                    rows.append(row)

    table = pd.DataFrame(rows)
    output_path = paths.tables_root / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": int(len(table)),
                "both_available": int((table["stage2_available"] & table["stage3_available"]).sum()),
                "written": str(output_path),
            },
            indent=2,
        )
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    """JSON이 있으면 읽고 없으면 None을 반환한다."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

