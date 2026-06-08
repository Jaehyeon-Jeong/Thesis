#!/usr/bin/env python
"""Build the N14-B1 conditional-regime merge table.

This script does not train a model. It joins Stage 2 visual-only predictions,
Stage 4 context-FiLM predictions, and the corresponding context feature table
into one long decision-level table. Later N14-B steps use this table to create
predefined regime buckets such as extreme F&G, high volatility, high funding,
high CFTC open-interest change, news-intense days, and Stage2 uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_kaggle.yaml")
    parser.add_argument("--stage2-experiment", default="stage2_i60_ohlc_vb_r20")
    parser.add_argument(
        "--stage4-experiment",
        default=(
            "stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_"
            "n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02"
        ),
    )
    parser.add_argument(
        "--context-name",
        default="stage4_derivatives_context_i60_ohlc_vb_r20_n16d_funding_plus_cftc_oi",
    )
    parser.add_argument("--analysis-name", default="n16_ohlc_vb_funding_plus_cftc_oi")
    parser.add_argument("--stage2-output-root", default="")
    parser.add_argument("--stage4-output-root", default="")
    parser.add_argument("--run-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-prefix", default="stage4_n14b1_conditional_merge")
    return parser.parse_args()


def main() -> None:
    """Build and write the N14-B1 merged decision table."""

    args = parse_args()
    cfg = _load_config(args.config)
    stage2_output_root = _resolve_output_root(
        explicit=args.stage2_output_root,
        cfg_value=_nested_get(cfg, ["stage2_dependency", "baseline_output_root"]),
        fallback=Path("../stage2_btc_extension/outputs/stage2"),
        expected_child="predictions",
    )
    stage4_output_root = _resolve_output_root(
        explicit=args.stage4_output_root,
        cfg_value=_nested_get(cfg, ["paths", "output_root"]),
        fallback=Path("outputs/stage4"),
        expected_child="predictions",
    )
    tables_root = Path(
        args.output_dir
        or _nested_get(cfg, ["paths", "tables_root"])
        or Path("reports/tables")
    ).expanduser()
    tables_root.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    missing_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for run_seed in args.run_seeds:
        seed = int(run_seed)
        stage2_path = (
            stage2_output_root
            / "predictions"
            / str(args.stage2_experiment)
            / f"seed_{seed}"
            / f"{args.split}_predictions.csv"
        )
        stage4_path = (
            stage4_output_root
            / "predictions"
            / str(args.stage4_experiment)
            / f"seed_{seed}"
            / f"{args.split}_predictions.csv"
        )
        context_path = (
            stage4_output_root
            / "context"
            / str(args.context_name)
            / f"seed_{seed}"
            / "context_features.csv"
        )
        missing = _missing_inputs(
            {
                "stage2_predictions": stage2_path,
                "stage4_predictions": stage4_path,
                "context_features": context_path,
            },
            run_seed=seed,
        )
        if missing:
            missing_rows.extend(missing)
            continue

        stage2 = _prepare_prediction_frame(pd.read_csv(stage2_path), prefix="stage2")
        stage4 = _prepare_prediction_frame(pd.read_csv(stage4_path), prefix="stage4")
        context = _prepare_context_frame(pd.read_csv(context_path))
        feature_rows.extend(_feature_inventory(context, run_seed=seed))

        merged = stage2.merge(stage4, on="sample_index", how="inner", validate="one_to_one")
        merged = merged.merge(context, on="sample_index", how="left", validate="one_to_one")
        if merged.empty:
            missing_rows.append(
                {
                    "run_seed": seed,
                    "input_name": "merged",
                    "path": f"{stage2_path} :: {stage4_path} :: {context_path}",
                    "reason": "no_common_sample_index",
                }
            )
            continue

        merged.insert(0, "analysis_name", str(args.analysis_name))
        merged.insert(1, "run_seed", seed)
        merged["label"] = _coalesce(merged, "label_stage4", "label_stage2").astype(int)
        merged["Date"] = _coalesce(merged, "Date_stage4", "Date_stage2")
        merged["future_return"] = _coalesce(
            merged,
            "future_return_stage4",
            "future_return_stage2",
        )
        merged["stage2_correct"] = _correct_column(merged, "stage2")
        merged["stage4_correct"] = _correct_column(merged, "stage4")
        merged["transition_type"] = _transition_column(merged)
        merged["stage2_margin"] = (
            pd.to_numeric(merged["prob_up_stage2"], errors="raise") - 0.5
        ).abs()
        merged["stage4_margin"] = (
            pd.to_numeric(merged["prob_up_stage4"], errors="raise") - 0.5
        ).abs()
        merged["prob_up_delta"] = (
            pd.to_numeric(merged["prob_up_stage4"], errors="raise")
            - pd.to_numeric(merged["prob_up_stage2"], errors="raise")
        )
        merged["stage2_true_prob"] = _true_class_probability(merged, "stage2")
        merged["stage4_true_prob"] = _true_class_probability(merged, "stage4")
        merged["true_prob_delta"] = merged["stage4_true_prob"] - merged["stage2_true_prob"]
        merged["stage2_uncertain_45_55"] = merged["prob_up_stage2"].between(0.45, 0.55)
        merged["stage2_uncertain_40_60"] = merged["prob_up_stage2"].between(0.40, 0.60)
        merged["stage2_confident_up_70"] = merged["prob_up_stage2"].ge(0.70)
        merged["stage2_confident_down_30"] = merged["prob_up_stage2"].le(0.30)
        frames.append(merged)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    seed_summary = _seed_summary(all_df)
    transition_summary = _transition_summary(all_df)
    missing_df = pd.DataFrame(missing_rows)
    feature_inventory = pd.DataFrame(feature_rows).drop_duplicates() if feature_rows else pd.DataFrame()

    paths = {
        "merged_decisions": tables_root / f"{args.output_prefix}_merged_decisions.csv",
        "seed_summary": tables_root / f"{args.output_prefix}_seed_summary.csv",
        "transition_summary": tables_root / f"{args.output_prefix}_transition_summary.csv",
        "context_feature_inventory": tables_root
        / f"{args.output_prefix}_context_feature_inventory.csv",
        "missing_inputs": tables_root / f"{args.output_prefix}_missing_inputs.csv",
        "report": tables_root / f"{args.output_prefix}_report.md",
        "manifest": tables_root / f"{args.output_prefix}_manifest.json",
    }
    _write_csv(all_df, paths["merged_decisions"])
    _write_csv(seed_summary, paths["seed_summary"])
    _write_csv(transition_summary, paths["transition_summary"])
    _write_csv(feature_inventory, paths["context_feature_inventory"])
    _write_csv(missing_df, paths["missing_inputs"])
    paths["manifest"].write_text(
        json.dumps(
            {
                "analysis_name": str(args.analysis_name),
                "stage2_experiment": str(args.stage2_experiment),
                "stage4_experiment": str(args.stage4_experiment),
                "context_name": str(args.context_name),
                "split": str(args.split),
                "run_seeds": [int(seed) for seed in args.run_seeds],
                "num_rows": int(len(all_df)),
                "num_missing_inputs": int(len(missing_df)),
                "stage2_output_root": str(stage2_output_root),
                "stage4_output_root": str(stage4_output_root),
                "written": {key: str(path) for key, path in paths.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["report"].write_text(
        _build_report(
            args=args,
            all_df=all_df,
            seed_summary=seed_summary,
            transition_summary=transition_summary,
            missing_df=missing_df,
            feature_inventory=feature_inventory,
            paths=paths,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok" if missing_df.empty else "partial",
                "analysis_name": str(args.analysis_name),
                "num_rows": int(len(all_df)),
                "num_missing_inputs": int(len(missing_df)),
                "written": {key: str(path) for key, path in paths.items()},
            },
            indent=2,
        )
    )


def _load_config(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_output_root(
    *,
    explicit: str,
    cfg_value: Any,
    fallback: Path,
    expected_child: str,
) -> Path:
    text = str(explicit or cfg_value or fallback).strip()
    root = Path(text).expanduser()
    candidates = [
        root,
        root / "outputs/stage2",
        root / "outputs/stage4",
        root / "stage2",
        root / "stage4",
    ]
    for candidate in candidates:
        if (candidate / expected_child).exists():
            return candidate
    return root


def _missing_inputs(paths: dict[str, Path], *, run_seed: int) -> list[dict[str, Any]]:
    rows = []
    for name, path in paths.items():
        if not path.exists():
            rows.append(
                {
                    "run_seed": int(run_seed),
                    "input_name": name,
                    "path": str(path),
                    "reason": "missing_file",
                }
            )
    return rows


def _prepare_prediction_frame(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    required = {"sample_index", "label", "prob_up", "pred_class"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{prefix} prediction CSV missing columns: {missing}")
    keep = [
        column
        for column in [
            "sample_index",
            "Date",
            "label",
            "future_return",
            "entry_close",
            "exit_close",
            "prob_down",
            "prob_up",
            "pred_class",
            "correct",
            "logit_down",
            "logit_up",
            "checkpoint_path",
            "experiment_name",
            "stage4_experiment_name",
        ]
        if column in frame.columns
    ]
    prepared = frame.loc[:, keep].copy()
    prepared["sample_index"] = pd.to_numeric(prepared["sample_index"], errors="raise").astype(int)
    prepared = prepared.drop_duplicates("sample_index", keep="first")
    return prepared.rename(
        columns={column: f"{column}_{prefix}" for column in keep if column != "sample_index"}
    )


def _prepare_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "sample_index" not in frame.columns:
        raise KeyError("context_features.csv missing sample_index")
    prepared = frame.copy()
    prepared["sample_index"] = pd.to_numeric(prepared["sample_index"], errors="raise").astype(int)
    prepared = prepared.drop_duplicates("sample_index", keep="first")
    rename = {
        column: f"context__{column}"
        for column in prepared.columns
        if column != "sample_index"
    }
    return prepared.rename(columns=rename)


def _feature_inventory(context_frame: pd.DataFrame, *, run_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in context_frame.columns:
        if column == "sample_index":
            continue
        raw_name = column.removeprefix("context__")
        if raw_name.endswith("_normalized"):
            feature_type = "normalized"
        elif raw_name.endswith("_missing"):
            feature_type = "missing_indicator"
        elif raw_name.endswith("_transformed"):
            feature_type = "transformed"
        elif raw_name.endswith("_imputed_clipped"):
            feature_type = "imputed_clipped"
        else:
            feature_type = "raw_or_metadata"
        rows.append(
            {
                "run_seed": int(run_seed),
                "column": column,
                "raw_name": raw_name,
                "feature_type": feature_type,
            }
        )
    return rows


def _coalesce(frame: pd.DataFrame, first: str, second: str) -> pd.Series:
    if first in frame.columns and second in frame.columns:
        return frame[first].combine_first(frame[second])
    if first in frame.columns:
        return frame[first]
    if second in frame.columns:
        return frame[second]
    raise KeyError(f"Missing both columns: {first}, {second}")


def _correct_column(frame: pd.DataFrame, prefix: str) -> pd.Series:
    correct_column = f"correct_{prefix}"
    if correct_column in frame.columns:
        return pd.to_numeric(frame[correct_column], errors="coerce").fillna(0).astype(int).eq(1)
    return (
        pd.to_numeric(frame[f"pred_class_{prefix}"], errors="raise").astype(int)
        == pd.to_numeric(frame["label"], errors="raise").astype(int)
    )


def _transition_column(frame: pd.DataFrame) -> pd.Series:
    stage2_correct = frame["stage2_correct"].astype(bool)
    stage4_correct = frame["stage4_correct"].astype(bool)
    transition = pd.Series("both_wrong", index=frame.index)
    transition.loc[stage2_correct & stage4_correct] = "both_correct"
    transition.loc[~stage2_correct & stage4_correct] = "correction"
    transition.loc[stage2_correct & ~stage4_correct] = "regression"
    return transition


def _true_class_probability(frame: pd.DataFrame, prefix: str) -> pd.Series:
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    prob_up = pd.to_numeric(frame[f"prob_up_{prefix}"], errors="raise")
    prob_down_column = f"prob_down_{prefix}"
    prob_down = (
        pd.to_numeric(frame[prob_down_column], errors="raise")
        if prob_down_column in frame.columns
        else 1.0 - prob_up
    )
    return prob_up.where(labels.eq(1), prob_down)


def _seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (analysis_name, run_seed), group in frame.groupby(["analysis_name", "run_seed"]):
        correction = int(group["transition_type"].eq("correction").sum())
        regression = int(group["transition_type"].eq("regression").sum())
        rows.append(
            {
                "analysis_name": analysis_name,
                "run_seed": int(run_seed),
                "num_decisions": int(len(group)),
                "stage2_accuracy": float(group["stage2_correct"].mean()),
                "stage4_accuracy": float(group["stage4_correct"].mean()),
                "delta_accuracy": float(group["stage4_correct"].mean() - group["stage2_correct"].mean()),
                "stage2_predicted_up_rate": float(group["pred_class_stage2"].mean()),
                "stage4_predicted_up_rate": float(group["pred_class_stage4"].mean()),
                "mean_prob_up_delta": float(group["prob_up_delta"].mean()),
                "mean_true_prob_delta": float(group["true_prob_delta"].mean()),
                "correction_count": correction,
                "regression_count": regression,
                "net_correction": correction - regression,
                "changed_decision_rate": float(
                    group["transition_type"].isin(["correction", "regression"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _transition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["analysis_name", "run_seed", "transition_type"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = (
        frame.groupby(["analysis_name", "run_seed"], dropna=False)
        .size()
        .reset_index(name="seed_total")
    )
    grouped = grouped.merge(totals, on=["analysis_name", "run_seed"], how="left")
    grouped["rate"] = grouped["count"] / grouped["seed_total"]
    return grouped.sort_values(["analysis_name", "run_seed", "transition_type"]).reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_report(
    *,
    args: argparse.Namespace,
    all_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    missing_df: pd.DataFrame,
    feature_inventory: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    lines = [
        "# 4-N14-B1 Conditional Merge Table",
        "",
        "Status: prepared by joining Stage 2 predictions, Stage 4 predictions, and context features.",
        "",
        "## Inputs",
        "",
        f"- Analysis name: `{args.analysis_name}`",
        f"- Stage 2 experiment: `{args.stage2_experiment}`",
        f"- Stage 4 experiment: `{args.stage4_experiment}`",
        f"- Context artifact: `{args.context_name}`",
        f"- Split: `{args.split}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.run_seeds)}`",
        "",
        "## Output",
        "",
        f"- Merged decisions: `{paths['merged_decisions']}`",
        f"- Seed summary: `{paths['seed_summary']}`",
        f"- Transition summary: `{paths['transition_summary']}`",
        f"- Context feature inventory: `{paths['context_feature_inventory']}`",
        "",
    ]
    if not missing_df.empty:
        lines.extend(
            [
                "## Missing Inputs",
                "",
                missing_df.to_markdown(index=False),
                "",
            ]
        )
    if not seed_summary.empty:
        lines.extend(
            [
                "## Seed Summary",
                "",
                seed_summary.to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    if not transition_summary.empty:
        compact = transition_summary.pivot_table(
            index=["analysis_name", "run_seed"],
            columns="transition_type",
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        lines.extend(["## Transition Counts", "", compact.to_markdown(index=False), ""])
    if not feature_inventory.empty:
        feature_counts = (
            feature_inventory.groupby("feature_type")
            .size()
            .reset_index(name="num_columns")
            .sort_values("feature_type")
        )
        lines.extend(["## Context Feature Inventory", "", feature_counts.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Next Step",
            "",
            "Use the merged decision table for N14-B2 bucket construction. The first",
            "recommended buckets are high funding, high CFTC OI change, Stage2 uncertainty,",
            "and transition groups (`correction`, `regression`, `both_correct`, `both_wrong`).",
            "",
        ]
    )
    if not all_df.empty:
        lines.extend(
            [
                "## Required Columns For N14-B2",
                "",
                "- `stage2_correct`, `stage4_correct`, `transition_type`",
                "- `prob_up_stage2`, `prob_up_stage4`, `prob_up_delta`",
                "- `stage2_uncertain_45_55`, `stage2_uncertain_40_60`",
                "- `context__*_normalized` and raw context columns",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
