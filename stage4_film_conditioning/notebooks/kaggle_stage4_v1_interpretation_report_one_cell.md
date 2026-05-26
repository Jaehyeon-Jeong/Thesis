# Kaggle Stage 4 v1 Interpretation Report One Cell

Copy this Python cell **below the Stage 4 five-seed runner output cell**. It does
not retrain models. It reads the five-seed result tables plus saved
prediction/Grad-CAM/modulation artifacts and writes a diagnostic interpretation
report.

Purpose:
- Confirm whether Stage 4 v1 improves over the Stage 2 selected baseline.
- Identify the best and worst `film_full` seeds.
- Compare prediction collapse, Grad-CAM samples, and gamma/beta summaries.
- Produce the rationale for a Stage 4 v2 stabilization experiment.

```python
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

# ============================================================
# User settings
# ============================================================
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/stage4"
TABLES_ROOT = PROJECT_ROOT / "reports/tables"
REPORT_ROOT = PROJECT_ROOT / "reports/stage4_v1_interpretation"
REPORT_TABLE_ROOT = REPORT_ROOT / "tables"
REPORT_FIGURE_ROOT = REPORT_ROOT / "figures"

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
EVAL_SPLIT = "test"
MAIN_METHOD = "film_full"
GRADCAM_SAMPLES_PER_CLASS = 2

# Stage 2 selected baseline from the prior five-seed check:
# I60/R20/ohlc_ma_vb, seeds 42-46.
STAGE2_SELECTED_BASELINE = {
    "accuracy_mean": 0.5793199167244969,
    "accuracy_std": 0.0182183337383599,
    "roc_auc_mean": 0.584861903099422,
    "roc_auc_std": 0.0232503983423101,
    "f1_mean": 0.6510711809366792,
    "f1_std": 0.0069724431115118,
    "brier_score_mean": 0.2743367621571529,
    "brier_score_std": 0.0089566268480153,
    "long_flat_sharpe_net_mean": 3.442312147251912,
    "long_short_sharpe_net_mean": 2.407759221482054,
}

SEED_RESULTS_CSV = TABLES_ROOT / "stage4_four_ablation_five_seed_seed_results.csv"
MEAN_STD_CSV = TABLES_ROOT / "stage4_four_ablation_five_seed_mean_std_results.csv"


def experiment_name(context_method: str) -> str:
    return (
        f"stage4_{context_method}_i{IMAGE_WINDOW}_"
        f"{IMAGE_SPEC}_r{RETURN_HORIZON}_c{CONTEXT_WINDOW}"
    )


def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def to_markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" for _ in headers]) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col], digits=digits) for col in headers) + " |")
    return "\n".join(lines)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table missing: {path}")
    return pd.read_csv(path)


def prediction_path(context_method: str, seed: int) -> Path:
    return (
        OUTPUT_ROOT
        / "predictions"
        / experiment_name(context_method)
        / f"seed_{seed}"
        / f"{EVAL_SPLIT}_predictions.csv"
    )


def gradcam_dir(context_method: str, seed: int) -> Path:
    return (
        OUTPUT_ROOT
        / "figures"
        / experiment_name(context_method)
        / f"seed_{seed}"
        / "gradcam"
        / EVAL_SPLIT
    )


def read_predictions(context_method: str, seed: int) -> pd.DataFrame | None:
    path = prediction_path(context_method, seed)
    if not path.exists():
        return None
    return pd.read_csv(path)


def prediction_diagnostics(context_method: str, seed: int) -> dict:
    predictions = read_predictions(context_method, seed)
    row = {"context_method": context_method, "run_seed": seed, "predictions_available": predictions is not None}
    if predictions is None or predictions.empty:
        return row

    pred_class = predictions["pred_class"].astype(int)
    label = predictions["label"].astype(int)
    prob_up = pd.to_numeric(predictions["prob_up"], errors="coerce")
    correct = predictions["correct"].astype(int) if "correct" in predictions else (pred_class == label).astype(int)
    row.update(
        {
            "num_predictions": int(len(predictions)),
            "predicted_positive_rate": float(pred_class.mean()),
            "true_positive_rate": float(label.mean()),
            "correct_rate": float(correct.mean()),
            "prob_up_mean": float(prob_up.mean()),
            "prob_up_std": float(prob_up.std(ddof=1)),
            "prob_up_q05": float(prob_up.quantile(0.05)),
            "prob_up_q50": float(prob_up.quantile(0.50)),
            "prob_up_q95": float(prob_up.quantile(0.95)),
            "correct_up_count": int(((pred_class == 1) & (label == 1)).sum()),
            "incorrect_up_count": int(((pred_class == 1) & (label == 0)).sum()),
            "correct_down_count": int(((pred_class == 0) & (label == 0)).sum()),
            "incorrect_down_count": int(((pred_class == 0) & (label == 1)).sum()),
        }
    )
    return row


def select_case_rows(context_method: str, seed: int, n: int = 3) -> pd.DataFrame:
    predictions = read_predictions(context_method, seed)
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["pred_class"] = frame["pred_class"].astype(int)
    frame["label"] = frame["label"].astype(int)
    if "correct" not in frame:
        frame["correct"] = (frame["pred_class"] == frame["label"]).astype(int)
    frame["prob_down"] = pd.to_numeric(frame.get("prob_down", 1.0 - frame["prob_up"]), errors="coerce")
    frame["prob_up"] = pd.to_numeric(frame["prob_up"], errors="coerce")

    cases = [
        ("correct_up", (frame["pred_class"] == 1) & (frame["label"] == 1), "prob_up", False),
        ("incorrect_up", (frame["pred_class"] == 1) & (frame["label"] == 0), "prob_up", False),
        ("correct_down", (frame["pred_class"] == 0) & (frame["label"] == 0), "prob_down", False),
        ("incorrect_down", (frame["pred_class"] == 0) & (frame["label"] == 1), "prob_down", False),
    ]
    rows = []
    keep_cols = [
        "sample_index",
        "Date",
        "label_end_date",
        "future_return",
        "label",
        "pred_class",
        "correct",
        "prob_down",
        "prob_up",
    ]
    keep_cols = [col for col in keep_cols if col in frame.columns]
    for case_name, mask, score_col, ascending in cases:
        selected = frame.loc[mask].sort_values(score_col, ascending=ascending).head(n)
        for _, row in selected.iterrows():
            record = row[keep_cols].to_dict()
            record["case"] = case_name
            record["context_method"] = context_method
            record["run_seed"] = seed
            rows.append(record)
    return pd.DataFrame(rows)


def read_modulation_summary(context_method: str, seed: int) -> pd.DataFrame | None:
    path = gradcam_dir(context_method, seed) / "modulation_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["context_method"] = context_method
    frame["run_seed"] = seed
    return frame


def summarize_modulation(context_method: str, seed: int) -> dict:
    frame = read_modulation_summary(context_method, seed)
    row = {"context_method": context_method, "run_seed": seed, "modulation_available": frame is not None}
    if frame is None or frame.empty:
        return row
    metric_cols = [
        col for col in frame.columns
        if (
            col.startswith("block")
            and (
                col.endswith("_mean")
                or col.endswith("_std")
                or col.endswith("_min")
                or col.endswith("_max")
                or col.endswith("_l2")
            )
        )
    ]
    for col in metric_cols:
        row[f"{col}_sample_mean"] = float(pd.to_numeric(frame[col], errors="coerce").mean())
    return row


def copy_gradcam_figure(context_method: str, seed: int, label: str) -> Path | None:
    source = gradcam_dir(context_method, seed) / (
        f"btc_context_gradcam_{EVAL_SPLIT}_{GRADCAM_SAMPLES_PER_CLASS}perclass.png"
    )
    if not source.exists():
        return None
    REPORT_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = REPORT_FIGURE_ROOT / f"{label}_{context_method}_seed{seed}_gradcam.png"
    shutil.copy2(source, destination)
    return destination


def compact_method_summary(mean_std: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "context_method",
        "seed_count",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "f1_mean",
        "f1_std",
        "brier_score_mean",
        "long_flat_sharpe_net_mean",
        "long_short_sharpe_net_mean",
    ]
    cols = [col for col in cols if col in mean_std.columns]
    compact = mean_std[cols].copy()
    if "accuracy_mean" in compact.columns:
        compact = compact.sort_values("accuracy_mean", ascending=False)
    return compact


# ============================================================
# 1. Load result tables and identify good/bad film_full seeds
# ============================================================
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

seed_results = read_csv_required(SEED_RESULTS_CSV)
mean_std = read_csv_required(MEAN_STD_CSV)

method_summary = compact_method_summary(mean_std)
film_seed_rows = seed_results[seed_results["context_method"].eq(MAIN_METHOD)].copy()
if film_seed_rows.empty:
    raise ValueError(f"No rows found for MAIN_METHOD={MAIN_METHOD!r}")

film_seed_rows["accuracy"] = pd.to_numeric(film_seed_rows["accuracy"], errors="coerce")
film_seed_rows["roc_auc"] = pd.to_numeric(film_seed_rows["roc_auc"], errors="coerce")
film_seed_rows["f1"] = pd.to_numeric(film_seed_rows["f1"], errors="coerce")

good_row = film_seed_rows.sort_values(["accuracy", "roc_auc"], ascending=False).iloc[0]
bad_row = film_seed_rows.sort_values(["f1", "accuracy"], ascending=True).iloc[0]
good_seed = int(good_row["run_seed"])
bad_seed = int(bad_row["run_seed"])


# ============================================================
# 2. Prediction, sample, and modulation diagnostics
# ============================================================
diagnostic_rows = [
    prediction_diagnostics(MAIN_METHOD, int(seed))
    for seed in sorted(film_seed_rows["run_seed"].astype(int).unique())
]
diagnostics = pd.DataFrame(diagnostic_rows)

case_rows = pd.concat(
    [
        select_case_rows(MAIN_METHOD, good_seed, n=3),
        select_case_rows(MAIN_METHOD, bad_seed, n=3),
    ],
    ignore_index=True,
)

modulation_rows = [
    summarize_modulation(MAIN_METHOD, good_seed),
    summarize_modulation(MAIN_METHOD, bad_seed),
]
modulation_compare = pd.DataFrame(modulation_rows)

good_figure = copy_gradcam_figure(MAIN_METHOD, good_seed, "good_seed")
bad_figure = copy_gradcam_figure(MAIN_METHOD, bad_seed, "bad_seed")


# ============================================================
# 3. Write tables
# ============================================================
method_summary_csv = REPORT_TABLE_ROOT / "stage4_v1_interpretation_method_summary.csv"
diagnostics_csv = REPORT_TABLE_ROOT / "stage4_v1_interpretation_film_full_seed_diagnostics.csv"
cases_csv = REPORT_TABLE_ROOT / "stage4_v1_interpretation_film_full_good_bad_cases.csv"
modulation_csv = REPORT_TABLE_ROOT / "stage4_v1_interpretation_film_full_good_bad_modulation.csv"

method_summary.to_csv(method_summary_csv, index=False)
diagnostics.to_csv(diagnostics_csv, index=False)
case_rows.to_csv(cases_csv, index=False)
modulation_compare.to_csv(modulation_csv, index=False)


# ============================================================
# 4. Write Markdown interpretation report
# ============================================================
film_mean = mean_std[mean_std["context_method"].eq(MAIN_METHOD)].iloc[0]
film_vs_stage2 = {
    "accuracy_delta": float(film_mean["accuracy_mean"]) - STAGE2_SELECTED_BASELINE["accuracy_mean"],
    "roc_auc_delta": float(film_mean["roc_auc_mean"]) - STAGE2_SELECTED_BASELINE["roc_auc_mean"],
    "f1_delta": float(film_mean["f1_mean"]) - STAGE2_SELECTED_BASELINE["f1_mean"],
}

stage2_comparison = pd.DataFrame(
    [
        {
            "model": "Stage 2 selected baseline",
            "accuracy_mean": STAGE2_SELECTED_BASELINE["accuracy_mean"],
            "accuracy_std": STAGE2_SELECTED_BASELINE["accuracy_std"],
            "roc_auc_mean": STAGE2_SELECTED_BASELINE["roc_auc_mean"],
            "roc_auc_std": STAGE2_SELECTED_BASELINE["roc_auc_std"],
            "f1_mean": STAGE2_SELECTED_BASELINE["f1_mean"],
            "f1_std": STAGE2_SELECTED_BASELINE["f1_std"],
        },
        {
            "model": "Stage 4 v1 film_full",
            "accuracy_mean": film_mean["accuracy_mean"],
            "accuracy_std": film_mean["accuracy_std"],
            "roc_auc_mean": film_mean["roc_auc_mean"],
            "roc_auc_std": film_mean["roc_auc_std"],
            "f1_mean": film_mean["f1_mean"],
            "f1_std": film_mean["f1_std"],
        },
    ]
)

good_bad_summary = pd.DataFrame(
    [
        {
            "role": "good_seed",
            "seed": good_seed,
            "accuracy": good_row["accuracy"],
            "roc_auc": good_row["roc_auc"],
            "f1": good_row["f1"],
        },
        {
            "role": "bad_or_collapse_seed",
            "seed": bad_seed,
            "accuracy": bad_row["accuracy"],
            "roc_auc": bad_row["roc_auc"],
            "f1": bad_row["f1"],
        },
    ]
)

report = f"""# Stage 4 v1 Interpretation Report

## Setup

- Model family: `I{IMAGE_WINDOW}/R{RETURN_HORIZON}/{IMAGE_SPEC}`
- Context window: `{CONTEXT_WINDOW}`
- Main interpretation target: `{MAIN_METHOD}`
- Context features: F&G value, F&G 60-day summaries, BB60 %B/bandwidth, MFI60, RV60
- This report does not retrain models. It reads saved five-seed output artifacts.

## Stage 4 v1 Result Summary

{to_markdown_table(method_summary)}

## Stage 2 Baseline Comparison

{to_markdown_table(stage2_comparison)}

Delta for `film_full` versus Stage 2 selected baseline:

- Accuracy delta: `{film_vs_stage2["accuracy_delta"]:.4f}`
- ROC-AUC delta: `{film_vs_stage2["roc_auc_delta"]:.4f}`
- F1 delta: `{film_vs_stage2["f1_delta"]:.4f}`

Interpretation:

- Stage 4 v1 does not robustly beat the Stage 2 selected baseline.
- `film_full` is still the most relevant Stage 4 method because it is the best
  FiLM variant and showed promising individual seeds.
- The main issue is instability, not only low average performance.

## Good Seed vs Bad Seed

{to_markdown_table(good_bad_summary)}

Prediction diagnostics for all `film_full` seeds:

{to_markdown_table(diagnostics)}

Interpretation:

- A seed with very low F1 or extreme predicted-positive rate indicates class
  collapse.
- If the bad seed shows a much more one-sided prediction distribution than the
  good seed, the v1 FiLM path is likely overpowering or destabilizing image
  evidence.
- This should be treated as diagnostic evidence, not as a market interpretation.

## Correct/Incorrect Case Table

The table below selects high-confidence correct/incorrect Up/Down cases from
the best and worst `film_full` seeds. Use these sample IDs when inspecting
Grad-CAM images and modulation exports.

{to_markdown_table(case_rows)}

## Modulation Summary

The table written to `{modulation_csv.name}` compares saved gamma/beta summaries
from the existing Grad-CAM samples for the good and bad seeds.

Key interpretation rule:

- Grad-CAM explains where the chart image affected the predicted class.
- FiLM gamma/beta summaries explain how market context modulated CNN feature
  channels.
- Because Stage 4 v1 underperformed, gamma/beta should be interpreted as a
  failure diagnostic first, not as validated alpha signal.

## v2 Design Rationale

Stage 4 v2 should be justified from this v1 diagnosis:

1. If the bad seed collapses toward one class, reduce the modulation strength.
2. If gamma/beta summaries are more volatile in the bad seed, use bounded FiLM:
   `gamma = 1 + s_gamma * tanh(raw_gamma)`, `beta = s_beta * tanh(raw_beta)`.
3. If early-layer modulation appears unstable, test last-block-only FiLM.
4. Keep the same data and context features first, so v2 is a structural
   stabilization experiment rather than a different dataset experiment.

Recommended v2 candidates:

- `film_full_bounded_all_blocks`
- `film_full_bounded_last_block`

## Artifacts

- Method summary: `{method_summary_csv}`
- Film seed diagnostics: `{diagnostics_csv}`
- Good/bad sample cases: `{cases_csv}`
- Good/bad modulation summary: `{modulation_csv}`
- Good seed Grad-CAM copy: `{good_figure}`
- Bad seed Grad-CAM copy: `{bad_figure}`
"""

report_path = REPORT_ROOT / "stage4_v1_interpretation_report.md"
report_path.write_text(report, encoding="utf-8")

archive_base = PROJECT_ROOT / "stage4_v1_interpretation_report"
archive_path = shutil.make_archive(str(archive_base), "zip", REPORT_ROOT)


# ============================================================
# 5. Display report and figures
# ============================================================
display(Markdown(report))
if good_figure is not None:
    display(Markdown(f"## Good seed Grad-CAM: seed {good_seed}"))
    display(Image(filename=str(good_figure)))
if bad_figure is not None:
    display(Markdown(f"## Bad/collapse seed Grad-CAM: seed {bad_seed}"))
    display(Image(filename=str(bad_figure)))

print("\nDONE", flush=True)
print("Report:", report_path, flush=True)
print("Report zip:", archive_path, flush=True)
print("Method summary:", method_summary_csv, flush=True)
print("Diagnostics:", diagnostics_csv, flush=True)
print("Cases:", cases_csv, flush=True)
print("Modulation comparison:", modulation_csv, flush=True)
```
