#!/usr/bin/env python
"""Compare Stage 2 baseline predictions against a Stage 4 context-FiLM run.

This script is intended for N10-style interpretation. It does not train a
model. It reads full prediction CSVs, finds samples where the context-FiLM
model corrects or breaks the Stage 2 visual baseline, and writes compact tables
for targeted Grad-CAM/news/gamma-beta follow-up.
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
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_kaggle.yaml")
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--context-window", type=int, default=60)
    parser.add_argument("--context-method", default="film_full_bounded_last_block")
    parser.add_argument(
        "--stage4-experiment-suffix",
        default="news_tfidf_svd32_w7_20_60_pretrained_frozen_s0p02",
    )
    parser.add_argument(
        "--context-name",
        default="",
        help=(
            "Optional Stage 4 context artifact name. If omitted, the script "
            "infers the N10 SVD32 news context name."
        ),
    )
    parser.add_argument("--stage2-output-root", default="")
    parser.add_argument("--stage4-output-root", default="")
    parser.add_argument("--run-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--output-prefix",
        default="stage4_n10_stage2_vs_n10_correction_analysis",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Stage 2 versus Stage 4 correction analysis."""

    args = parse_args()
    cfg = _load_config_if_available(args.config)
    stage2_output_root = _resolve_output_root(
        explicit=args.stage2_output_root,
        cfg_value=_nested_get(cfg, ["stage2_dependency", "baseline_output_root"]),
        fallback=Path("outputs/stage2"),
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

    stage2_experiment = _stage2_experiment_name(
        args.image_window,
        args.image_spec,
        args.return_horizon,
    )
    stage4_experiment = _stage4_experiment_name(
        args.image_window,
        args.image_spec,
        args.return_horizon,
        args.context_method,
        args.context_window,
        args.stage4_experiment_suffix,
    )
    context_name = args.context_name or _default_news_context_name(
        args.image_window,
        args.image_spec,
        args.return_horizon,
        args.stage4_experiment_suffix,
    )

    all_frames: list[pd.DataFrame] = []
    missing_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for run_seed in args.run_seeds:
        stage2_path = (
            stage2_output_root
            / "predictions"
            / stage2_experiment
            / f"seed_{int(run_seed)}"
            / f"{args.split}_predictions.csv"
        )
        stage4_path = (
            stage4_output_root
            / "predictions"
            / stage4_experiment
            / f"seed_{int(run_seed)}"
            / f"{args.split}_predictions.csv"
        )
        missing = []
        if not stage2_path.exists():
            missing.append(("stage2", stage2_path))
        if not stage4_path.exists():
            missing.append(("stage4", stage4_path))
        if missing:
            for source, path in missing:
                missing_rows.append(
                    {
                        "run_seed": int(run_seed),
                        "source": source,
                        "path": str(path),
                        "reason": "missing_prediction_csv",
                    }
                )
            continue

        stage2 = _prepare_prediction_frame(pd.read_csv(stage2_path), "stage2")
        stage4 = _prepare_prediction_frame(pd.read_csv(stage4_path), "stage4")
        merged = stage2.merge(stage4, on="sample_index", how="inner", validate="one_to_one")
        if merged.empty:
            missing_rows.append(
                {
                    "run_seed": int(run_seed),
                    "source": "merge",
                    "path": f"{stage2_path} :: {stage4_path}",
                    "reason": "no_common_sample_index",
                }
            )
            continue
        merged.insert(0, "run_seed", int(run_seed))
        merged["label"] = merged["label_stage4"].fillna(merged["label_stage2"]).astype(int)
        merged["Date"] = merged["Date_stage4"].fillna(merged["Date_stage2"])
        merged["future_return"] = merged["future_return_stage4"].fillna(
            merged["future_return_stage2"]
        )
        merged["stage2_correct"] = _correct_column(merged, "stage2")
        merged["stage4_correct"] = _correct_column(merged, "stage4")
        merged["stage2_margin"] = (merged["prob_up_stage2"] - 0.5).abs()
        merged["stage4_margin"] = (merged["prob_up_stage4"] - 0.5).abs()
        merged["prob_up_delta_stage4_minus_stage2"] = (
            merged["prob_up_stage4"] - merged["prob_up_stage2"]
        )
        merged["stage2_true_prob"] = _true_class_probability(merged, "stage2")
        merged["stage4_true_prob"] = _true_class_probability(merged, "stage4")
        merged["true_prob_delta_stage4_minus_stage2"] = (
            merged["stage4_true_prob"] - merged["stage2_true_prob"]
        )
        merged["transition"] = _transition_column(merged)
        merged = _merge_context_features(
            merged,
            context_root=stage4_output_root / "context",
            context_name=context_name,
            run_seed=int(run_seed),
        )
        all_frames.append(merged)
        seed_rows.append(_seed_summary(merged, run_seed=int(run_seed)))

    all_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    missing_df = pd.DataFrame(missing_rows)
    seed_summary_df = pd.DataFrame(seed_rows)
    transition_summary_df = _transition_summary(all_df)
    correction_df = _selected_transition(
        all_df,
        transition="stage2_wrong_to_stage4_correct",
        top_k=int(args.top_k),
    )
    regression_df = _selected_transition(
        all_df,
        transition="stage2_correct_to_stage4_wrong",
        top_k=int(args.top_k),
    )
    selected_df = pd.concat(
        [
            correction_df.assign(analysis_group="stage2_wrong_to_stage4_correct"),
            regression_df.assign(analysis_group="stage2_correct_to_stage4_wrong"),
        ],
        ignore_index=True,
    )

    written = {
        "all_transitions": tables_root / f"{args.output_prefix}_all_transitions.csv",
        "seed_summary": tables_root / f"{args.output_prefix}_seed_summary.csv",
        "transition_summary": tables_root / f"{args.output_prefix}_transition_summary.csv",
        "top_corrections": tables_root / f"{args.output_prefix}_top_corrections.csv",
        "top_regressions": tables_root / f"{args.output_prefix}_top_regressions.csv",
        "selected_for_gradcam": tables_root / f"{args.output_prefix}_selected_for_gradcam.csv",
        "selected_indices_json": tables_root / f"{args.output_prefix}_selected_indices_by_seed.json",
        "missing_predictions": tables_root / f"{args.output_prefix}_missing_predictions.csv",
        "markdown_report": tables_root / f"{args.output_prefix}_report.md",
    }
    _write_csv(all_df, written["all_transitions"])
    _write_csv(seed_summary_df, written["seed_summary"])
    _write_csv(transition_summary_df, written["transition_summary"])
    _write_csv(correction_df, written["top_corrections"])
    _write_csv(regression_df, written["top_regressions"])
    _write_csv(selected_df, written["selected_for_gradcam"])
    _write_csv(missing_df, written["missing_predictions"])
    written["selected_indices_json"].write_text(
        json.dumps(_selected_indices_by_seed(selected_df), indent=2),
        encoding="utf-8",
    )
    written["markdown_report"].write_text(
        _build_markdown_report(
            args=args,
            stage2_output_root=stage2_output_root,
            stage4_output_root=stage4_output_root,
            stage2_experiment=stage2_experiment,
            stage4_experiment=stage4_experiment,
            context_name=context_name,
            seed_summary_df=seed_summary_df,
            transition_summary_df=transition_summary_df,
            correction_df=correction_df,
            regression_df=regression_df,
            missing_df=missing_df,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok" if missing_df.empty else "partial",
                "stage2_experiment": stage2_experiment,
                "stage4_experiment": stage4_experiment,
                "context_name": context_name,
                "num_compared_rows": int(len(all_df)),
                "num_missing_rows": int(len(missing_df)),
                "num_top_corrections": int(len(correction_df)),
                "num_top_regressions": int(len(regression_df)),
                "written": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


def _load_config_if_available(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
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


def _stage2_experiment_name(image_window: int, image_spec: str, return_horizon: int) -> str:
    return f"stage2_i{int(image_window)}_{image_spec}_r{int(return_horizon)}"


def _stage4_experiment_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
    context_method: str,
    context_window: int,
    suffix: str,
) -> str:
    base = (
        f"stage4_{context_method}_i{int(image_window)}_"
        f"{image_spec}_r{int(return_horizon)}_c{int(context_window)}"
    )
    return f"{base}_{suffix}" if suffix else base


def _default_news_context_name(
    image_window: int,
    image_spec: str,
    return_horizon: int,
    suffix: str,
) -> str:
    feature_set = "tfidf_svd32_w7_20_60"
    for token in suffix.split("_"):
        if token.startswith("svd") and token[3:].isdigit():
            feature_set = f"tfidf_{token}_w7_20_60"
            break
    return (
        f"stage4_news_context_i{int(image_window)}_{image_spec}_"
        f"r{int(return_horizon)}_{feature_set}"
    )


def _prepare_prediction_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {"sample_index", "label", "prob_up", "pred_class"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{prefix} predictions missing columns: {missing}")
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
        ]
        if column in frame.columns
    ]
    prepared = frame.loc[:, keep].copy()
    prepared["sample_index"] = pd.to_numeric(
        prepared["sample_index"], errors="raise"
    ).astype(int)
    if prepared["sample_index"].duplicated().any():
        prepared = prepared.drop_duplicates("sample_index", keep="first")
    rename = {column: f"{column}_{prefix}" for column in keep if column != "sample_index"}
    return prepared.rename(columns=rename)


def _correct_column(frame: pd.DataFrame, prefix: str) -> pd.Series:
    column = f"correct_{prefix}"
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int).eq(1)
    return (
        pd.to_numeric(frame[f"pred_class_{prefix}"], errors="raise").astype(int)
        == pd.to_numeric(frame["label"], errors="raise").astype(int)
    )


def _true_class_probability(frame: pd.DataFrame, prefix: str) -> pd.Series:
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    prob_up = pd.to_numeric(frame[f"prob_up_{prefix}"], errors="raise")
    if f"prob_down_{prefix}" in frame.columns:
        prob_down = pd.to_numeric(frame[f"prob_down_{prefix}"], errors="raise")
    else:
        prob_down = 1.0 - prob_up
    return prob_up.where(labels.eq(1), prob_down)


def _transition_column(frame: pd.DataFrame) -> pd.Series:
    stage2_correct = frame["stage2_correct"].astype(bool)
    stage4_correct = frame["stage4_correct"].astype(bool)
    transition = pd.Series("both_wrong", index=frame.index)
    transition.loc[stage2_correct & stage4_correct] = "both_correct"
    transition.loc[~stage2_correct & stage4_correct] = "stage2_wrong_to_stage4_correct"
    transition.loc[stage2_correct & ~stage4_correct] = "stage2_correct_to_stage4_wrong"
    return transition


def _merge_context_features(
    frame: pd.DataFrame,
    *,
    context_root: Path,
    context_name: str,
    run_seed: int,
) -> pd.DataFrame:
    context_path = context_root / context_name / f"seed_{int(run_seed)}" / "context_features.csv"
    if not context_path.exists():
        frame["context_features_path"] = str(context_path)
        frame["context_features_available"] = False
        return frame
    context = pd.read_csv(context_path)
    if "sample_index" not in context.columns:
        frame["context_features_path"] = str(context_path)
        frame["context_features_available"] = False
        return frame
    keep = [
        column
        for column in context.columns
        if column == "sample_index"
        or column.startswith("news_")
        or column.endswith("_normalized")
        or column.endswith("_missing")
    ]
    context = context.loc[:, keep].copy()
    context["sample_index"] = pd.to_numeric(context["sample_index"], errors="raise").astype(int)
    context = context.drop_duplicates("sample_index", keep="first")
    merged = frame.merge(context, on="sample_index", how="left")
    merged["context_features_path"] = str(context_path)
    merged["context_features_available"] = True
    return merged


def _seed_summary(frame: pd.DataFrame, *, run_seed: int) -> dict[str, Any]:
    return {
        "run_seed": int(run_seed),
        "num_samples": int(len(frame)),
        "stage2_accuracy": float(frame["stage2_correct"].mean()),
        "stage4_accuracy": float(frame["stage4_correct"].mean()),
        "accuracy_delta_stage4_minus_stage2": float(
            frame["stage4_correct"].mean() - frame["stage2_correct"].mean()
        ),
        "stage2_predicted_positive_rate": float(
            pd.to_numeric(frame["pred_class_stage2"], errors="raise").mean()
        ),
        "stage4_predicted_positive_rate": float(
            pd.to_numeric(frame["pred_class_stage4"], errors="raise").mean()
        ),
        "mean_true_prob_delta_stage4_minus_stage2": float(
            frame["true_prob_delta_stage4_minus_stage2"].mean()
        ),
        "stage2_wrong_to_stage4_correct": int(
            frame["transition"].eq("stage2_wrong_to_stage4_correct").sum()
        ),
        "stage2_correct_to_stage4_wrong": int(
            frame["transition"].eq("stage2_correct_to_stage4_wrong").sum()
        ),
        "both_correct": int(frame["transition"].eq("both_correct").sum()),
        "both_wrong": int(frame["transition"].eq("both_wrong").sum()),
    }


def _transition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["run_seed", "transition"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = frame.groupby("run_seed").size().rename("seed_total").reset_index()
    grouped = grouped.merge(totals, on="run_seed", how="left")
    grouped["rate"] = grouped["count"] / grouped["seed_total"]
    return grouped.sort_values(["run_seed", "transition"]).reset_index(drop=True)


def _selected_transition(frame: pd.DataFrame, *, transition: str, top_k: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    selected = frame.loc[frame["transition"].eq(transition)].copy()
    if selected.empty:
        return selected
    selected["selection_score"] = selected["true_prob_delta_stage4_minus_stage2"].abs()
    if transition == "stage2_wrong_to_stage4_correct":
        sort_columns = [
            "true_prob_delta_stage4_minus_stage2",
            "stage4_true_prob",
            "stage2_margin",
        ]
        ascending = [False, False, True]
    else:
        sort_columns = [
            "true_prob_delta_stage4_minus_stage2",
            "stage2_true_prob",
            "stage4_margin",
        ]
        ascending = [True, False, True]
    return (
        selected.sort_values(sort_columns, ascending=ascending)
        .head(int(top_k))
        .reset_index(drop=True)
    )


def _selected_indices_by_seed(frame: pd.DataFrame) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    if frame.empty:
        return result
    for run_seed, seed_frame in frame.groupby("run_seed"):
        result[str(int(run_seed))] = [
            int(value) for value in seed_frame["sample_index"].dropna().astype(int).tolist()
        ]
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_markdown_report(
    *,
    args: argparse.Namespace,
    stage2_output_root: Path,
    stage4_output_root: Path,
    stage2_experiment: str,
    stage4_experiment: str,
    context_name: str,
    seed_summary_df: pd.DataFrame,
    transition_summary_df: pd.DataFrame,
    correction_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> str:
    lines = [
        "# Stage 2 vs N10 Correction Analysis",
        "",
        "Purpose: identify samples where the N10 news-FiLM model corrects or breaks "
        "the fixed Stage 2 visual baseline.",
        "",
        "## Inputs",
        "",
        f"- Stage 2 output root: `{stage2_output_root}`",
        f"- Stage 4 output root: `{stage4_output_root}`",
        f"- Stage 2 experiment: `{stage2_experiment}`",
        f"- Stage 4 experiment: `{stage4_experiment}`",
        f"- Context artifact: `{context_name}`",
        f"- Split: `{args.split}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.run_seeds)}`",
        "",
    ]
    if not missing_df.empty:
        lines += [
            "## Missing Inputs",
            "",
            missing_df.to_markdown(index=False),
            "",
            "The analysis is partial until the missing prediction CSVs are generated.",
            "",
        ]
    if not seed_summary_df.empty:
        lines += [
            "## Seed Summary",
            "",
            seed_summary_df.to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    if not transition_summary_df.empty:
        compact = transition_summary_df.pivot_table(
            index="run_seed",
            columns="transition",
            values="count",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()
        lines += [
            "## Transition Counts",
            "",
            compact.to_markdown(index=False),
            "",
        ]
    lines += [
        "## Top Stage2 Wrong -> N10 Correct",
        "",
        _compact_selected_table(correction_df),
        "",
        "## Top Stage2 Correct -> N10 Wrong",
        "",
        _compact_selected_table(regression_df),
        "",
        "## How To Use For Grad-CAM",
        "",
        "Use `selected_for_gradcam.csv` or `selected_indices_by_seed.json` to "
        "generate targeted Stage 2 and N10 Grad-CAM figures. The important group "
        "for the thesis claim is `stage2_wrong_to_stage4_correct`: same chart "
        "image, Stage 2 wrong, context-FiLM correct.",
        "",
    ]
    return "\n".join(lines)


def _compact_selected_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No samples found._"
    columns = [
        column
        for column in [
            "run_seed",
            "sample_index",
            "Date",
            "label",
            "future_return",
            "pred_class_stage2",
            "prob_up_stage2",
            "pred_class_stage4",
            "prob_up_stage4",
            "true_prob_delta_stage4_minus_stage2",
            "stage2_margin",
            "stage4_margin",
        ]
        if column in frame.columns
    ]
    return frame.loc[:, columns].to_markdown(index=False, floatfmt=".6f")


if __name__ == "__main__":
    main()
