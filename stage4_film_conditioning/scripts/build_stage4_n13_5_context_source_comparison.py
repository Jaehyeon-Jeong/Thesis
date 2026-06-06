"""Build the Stage 4 N13-5 context-source comparison tables.

N13-5 is an aggregation step. It does not train a new model; it compares the
completed Stage 2 frozen context-FiLM sources under one baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


TABLE_ROOT_DEFAULT = Path("reports/tables")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table is missing: {path}")
    return pd.read_csv(path)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _row(
    *,
    comparison_group: str,
    model: str,
    context_source: str,
    context_family: str,
    protocol: str,
    status: str,
    note: str,
    source_file: Path | str,
    accuracy_mean: Any,
    accuracy_std: Any = None,
    roc_auc_mean: Any = None,
    roc_auc_std: Any = None,
    f1_mean: Any = None,
    brier_score_mean: Any = None,
    predicted_positive_rate_mean: Any = None,
    long_flat_sharpe_net_mean: Any = None,
    long_short_sharpe_net_mean: Any = None,
    modulation_scale: Any = None,
    context_dim: Any = None,
    seed_count: Any = None,
    collapse_warning_count: Any = None,
    correction_count: Any = None,
    regression_count: Any = None,
    net_correction: Any = None,
) -> dict[str, Any]:
    return {
        "comparison_group": comparison_group,
        "model": model,
        "context_source": context_source,
        "context_family": context_family,
        "protocol": protocol,
        "status": status,
        "source_file": str(source_file),
        "note": note,
        "accuracy_mean": _float(accuracy_mean),
        "accuracy_std": _float(accuracy_std),
        "roc_auc_mean": _float(roc_auc_mean),
        "roc_auc_std": _float(roc_auc_std),
        "f1_mean": _float(f1_mean),
        "brier_score_mean": _float(brier_score_mean),
        "predicted_positive_rate_mean": _float(predicted_positive_rate_mean),
        "long_flat_sharpe_net_mean": _float(long_flat_sharpe_net_mean),
        "long_short_sharpe_net_mean": _float(long_short_sharpe_net_mean),
        "modulation_scale": _float(modulation_scale),
        "context_dim": _float(context_dim),
        "seed_count": _float(seed_count),
        "collapse_warning_count": _float(collapse_warning_count),
        "correction_count": _float(correction_count),
        "regression_count": _float(regression_count),
        "net_correction": _float(net_correction),
    }


def _find(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    matches = df[df[column].astype(str).eq(value)]
    if matches.empty:
        raise ValueError(f"Could not find {column}={value}")
    return matches.iloc[0]


def build_comparison(table_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    n10_path = table_root / "stage4_n10_metric_comparison.csv"
    n10 = _read_csv(n10_path)
    stage2 = _find(n10, "model", "Stage2 baseline")
    baseline = _row(
        comparison_group="baseline",
        model="Stage2 frozen baseline",
        context_source="visual-only ohlc_ma_vb",
        context_family="visual_only",
        protocol="Stage2 I60/R20 checkpoint reload, no context",
        status="completed",
        source_file=n10_path,
        note="Reference baseline for all Stage 2 frozen context-FiLM comparisons.",
        accuracy_mean=stage2["accuracy_mean"],
        accuracy_std=stage2["accuracy_std"],
        roc_auc_mean=stage2["roc_auc_mean"],
        roc_auc_std=stage2["roc_auc_std"],
        f1_mean=stage2["f1_mean"],
        brier_score_mean=stage2["brier_score_mean"],
        predicted_positive_rate_mean=stage2["predicted_positive_rate_mean"],
        long_flat_sharpe_net_mean=stage2["long_flat_sharpe_net_mean"],
        long_short_sharpe_net_mean=stage2["long_short_sharpe_net_mean"],
        seed_count=5,
        collapse_warning_count=0,
    )
    rows.append(baseline)

    n8b_path = table_root / "stage4_n8b_fg_only_pretrained_frozen_bounded_film_mean_std_results.csv"
    n8b = _read_csv(n8b_path)
    for _, item in n8b.iterrows():
        scale = item["modulation_scale"]
        rows.append(
            _row(
                comparison_group="main",
                model=f"N8-B F&G-only bounded FiLM s{scale:.2f}",
                context_source="F&G-only external crypto regime",
                context_family="fear_greed",
                protocol="Stage2 CNN/classifier frozen, bounded last-block FiLM",
                status="completed",
                source_file=n8b_path,
                note=(
                    "Best compact numeric context candidate."
                    if abs(scale - 0.02) < 1e-9
                    else "Higher F&G scale diagnostic."
                ),
                accuracy_mean=item["accuracy_mean"],
                accuracy_std=item["accuracy_std"],
                roc_auc_mean=item["roc_auc_mean"],
                roc_auc_std=item["roc_auc_std"],
                f1_mean=item["f1_mean"],
                brier_score_mean=item["brier_score_mean"],
                predicted_positive_rate_mean=item["predicted_positive_rate_mean"],
                long_flat_sharpe_net_mean=item["long_flat_sharpe_net_mean"],
                long_short_sharpe_net_mean=item["long_short_sharpe_net_mean"],
                modulation_scale=scale,
                context_dim=8,
                seed_count=item["seed_count"],
                collapse_warning_count=0,
            )
        )

    for model_name in [
        "N9-A news SVD8 s0.02",
        "N9 grid news SVD32 s0.02",
        "N9 grid news SVD8 s0.05",
    ]:
        item = _find(n10, "model", model_name)
        context_dim = 8 if "SVD8" in model_name else 32
        scale = 0.05 if "s0.05" in model_name else 0.02
        rows.append(
            _row(
                comparison_group="main",
                model=model_name,
                context_source="headline news TF-IDF/SVD",
                context_family="news",
                protocol="Stage2 CNN/classifier frozen, bounded last-block FiLM",
                status="completed",
                source_file=n10_path,
                note=(
                    "Best news accuracy/interpretability row."
                    if model_name == "N9 grid news SVD32 s0.02"
                    else "News ranking/calibration diagnostic."
                ),
                accuracy_mean=item["accuracy_mean"],
                accuracy_std=item["accuracy_std"],
                roc_auc_mean=item["roc_auc_mean"],
                roc_auc_std=item["roc_auc_std"],
                f1_mean=item["f1_mean"],
                brier_score_mean=item["brier_score_mean"],
                predicted_positive_rate_mean=item["predicted_positive_rate_mean"],
                long_flat_sharpe_net_mean=item["long_flat_sharpe_net_mean"],
                long_short_sharpe_net_mean=item["long_short_sharpe_net_mean"],
                modulation_scale=scale,
                context_dim=context_dim,
                seed_count=5,
                collapse_warning_count=0,
                correction_count=27 if model_name == "N9 grid news SVD32 s0.02" else None,
                regression_count=24 if model_name == "N9 grid news SVD32 s0.02" else None,
                net_correction=3 if model_name == "N9 grid news SVD32 s0.02" else None,
            )
        )

    n12c_path = table_root / "stage4_n12c_technical_only_pretrained_frozen_bounded_film_mean_std_results.csv"
    n12c = _read_csv(n12c_path)
    for _, item in n12c.iterrows():
        scale = item["modulation_scale"]
        rows.append(
            _row(
                comparison_group="main",
                model=f"N12-C technical-only bounded FiLM s{scale:.2f}",
                context_source="BB60/MFI60/RV60 chart-derived technical context",
                context_family="technical",
                protocol="Stage2 CNN/classifier frozen, bounded last-block FiLM",
                status="completed",
                source_file=n12c_path,
                note="Mostly redundant with ohlc_ma_vb chart information.",
                accuracy_mean=item["accuracy_mean"],
                accuracy_std=item["accuracy_std"],
                roc_auc_mean=item["roc_auc_mean"],
                roc_auc_std=item["roc_auc_std"],
                f1_mean=item["f1_mean"],
                brier_score_mean=item["brier_score_mean"],
                predicted_positive_rate_mean=item["predicted_positive_rate_mean"],
                long_flat_sharpe_net_mean=item["long_flat_sharpe_net_mean"],
                long_short_sharpe_net_mean=item["long_short_sharpe_net_mean"],
                modulation_scale=scale,
                context_dim=3,
                seed_count=item["seed_count"],
                collapse_warning_count=0,
            )
        )

    n13_2_path = table_root / "stage4_n13_2_fsi_only_pretrained_frozen_bounded_film_mean_std_results.csv"
    n13_2 = _read_csv(n13_2_path)
    for _, item in n13_2.iterrows():
        key = item["context_feature_set_key"]
        rows.append(
            _row(
                comparison_group="macro",
                model=f"N13-2 FSI-only {key}",
                context_source="OFR FSI financial-stress context",
                context_family="fsi",
                protocol="Stage2 CNN/classifier frozen, bounded last-block FiLM",
                status="completed",
                source_file=n13_2_path,
                note="Official macro stress proxy; stable but weak hard-decision gain.",
                accuracy_mean=item["accuracy_mean"],
                accuracy_std=item["accuracy_std"],
                roc_auc_mean=item["roc_auc_mean"],
                roc_auc_std=item["roc_auc_std"],
                f1_mean=item["f1_mean"],
                brier_score_mean=item["brier_score_mean"],
                predicted_positive_rate_mean=item["predicted_positive_rate_mean"],
                long_flat_sharpe_net_mean=item["long_flat_sharpe_net_mean"],
                long_short_sharpe_net_mean=item["long_short_sharpe_net_mean"],
                modulation_scale=item["modulation_scale"],
                context_dim=item["context_dim"],
                seed_count=item["seed_count"],
                collapse_warning_count=item["collapse_warning_count"],
                correction_count=item["stage2_wrong_context_correct_mean"] * item["seed_count"],
                regression_count=item["stage2_correct_context_wrong_mean"] * item["seed_count"],
                net_correction=item["net_correction_mean"] * item["seed_count"],
            )
        )

    n13_4_path = table_root / "stage4_n13_4_roro_only_pretrained_frozen_bounded_film_mean_std_results.csv"
    n13_4 = _read_csv(n13_4_path)
    for _, item in n13_4.iterrows():
        key = item["context_feature_set_key"]
        rows.append(
            _row(
                comparison_group="macro",
                model=f"N13-4 RORO-only {key}",
                context_source="KC Fed-inspired public-data RORO proxy",
                context_family="roro",
                protocol="Stage2 CNN/classifier frozen, bounded last-block FiLM",
                status="completed",
                source_file=n13_4_path,
                note="Public-data risk-off proxy; stable but mostly preserves Stage2 decisions.",
                accuracy_mean=item["accuracy_mean"],
                accuracy_std=item["accuracy_std"],
                roc_auc_mean=item["roc_auc_mean"],
                roc_auc_std=item["roc_auc_std"],
                f1_mean=item["f1_mean"],
                brier_score_mean=item["brier_score_mean"],
                predicted_positive_rate_mean=item["predicted_positive_rate_mean"],
                long_flat_sharpe_net_mean=item["long_flat_sharpe_net_mean"],
                long_short_sharpe_net_mean=item["long_short_sharpe_net_mean"],
                modulation_scale=item["modulation_scale"],
                context_dim=item["context_dim"],
                seed_count=item["seed_count"],
                collapse_warning_count=item["collapse_warning_count"],
                correction_count=item["stage2_wrong_context_correct_mean"] * item["seed_count"],
                regression_count=item["stage2_correct_context_wrong_mean"] * item["seed_count"],
                net_correction=item["net_correction_mean"] * item["seed_count"],
            )
        )

    df = pd.DataFrame(rows)
    for metric in ["accuracy_mean", "roc_auc_mean", "f1_mean", "brier_score_mean", "predicted_positive_rate_mean", "long_flat_sharpe_net_mean", "long_short_sharpe_net_mean"]:
        df[f"{metric}_delta_vs_stage2"] = df[metric] - baseline[metric]

    df["accuracy_rank"] = df["accuracy_mean"].rank(ascending=False, method="min")
    df["roc_auc_rank"] = df["roc_auc_mean"].rank(ascending=False, method="min")
    df["brier_rank"] = df["brier_score_mean"].rank(ascending=True, method="min")
    df["no_collapse"] = df["collapse_warning_count"].fillna(0).eq(0)

    ordered = df.sort_values(["accuracy_mean", "roc_auc_mean"], ascending=[False, False]).reset_index(drop=True)

    best_by_family = (
        ordered[ordered["context_family"].ne("visual_only")]
        .sort_values(["context_family", "accuracy_mean", "roc_auc_mean"], ascending=[True, False, False])
        .groupby("context_family", as_index=False)
        .head(1)
        .copy()
    )
    compact = pd.concat([ordered[ordered["context_family"].eq("visual_only")], best_by_family], ignore_index=True)
    compact = compact.sort_values(["accuracy_mean", "roc_auc_mean"], ascending=[False, False]).reset_index(drop=True)

    recommendation_rows = [
        {
            "rank": 1,
            "recommendation": "Use Stage 2 frozen baseline as the main benchmark.",
            "reason": "All completed context sources only move five-seed metrics by tiny margins.",
        },
        {
            "rank": 2,
            "recommendation": "Keep N8-B F&G-only s0.02 as the best compact context-FiLM candidate.",
            "reason": "It has the best five-seed accuracy among completed context rows, but the gain is only about +0.001.",
        },
        {
            "rank": 3,
            "recommendation": "Use news SVD32 s0.02 for targeted interpretability examples.",
            "reason": "It gives the clearest correction/regression table and modest ROC/Brier improvement, but weak hard-decision gain.",
        },
        {
            "rank": 4,
            "recommendation": "Do not select FSI-only or RORO-only as final models.",
            "reason": "Both macro sources are stable and conceptually useful, but they do not materially beat F&G or the visual baseline.",
        },
        {
            "rank": 5,
            "recommendation": "Proceed to N13-5A cross-context feature audit before selected-combo FiLM.",
            "reason": "A selected combo should be based on train-only redundancy/error-correlation diagnostics, not by stacking every context source.",
        },
    ]
    recommendation = pd.DataFrame(recommendation_rows)
    return ordered, compact, recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=TABLE_ROOT_DEFAULT)
    parser.add_argument("--output-prefix", default="stage4_n13_5_context_source_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_root = args.table_root
    table_root.mkdir(parents=True, exist_ok=True)
    full, compact, recommendation = build_comparison(table_root)

    full_path = table_root / f"{args.output_prefix}.csv"
    compact_path = table_root / f"{args.output_prefix}_compact.csv"
    recommendation_path = table_root / f"{args.output_prefix}_recommendation.csv"

    full.to_csv(full_path, index=False)
    compact.to_csv(compact_path, index=False)
    recommendation.to_csv(recommendation_path, index=False)

    print(
        {
            "status": "ok",
            "full_rows": int(len(full)),
            "compact_rows": int(len(compact)),
            "full": str(full_path),
            "compact": str(compact_path),
            "recommendation": str(recommendation_path),
        }
    )


if __name__ == "__main__":
    main()
