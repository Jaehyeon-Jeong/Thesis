# Kaggle Stage 2 Results Viewer

## English

Use this cell after running Stage 2 single-seed or five-seed grid experiments.
It reads output JSON files and backup zips, then shows compact tables.

## 한국어

Stage 2 single-seed 또는 five-seed grid 실험을 돌린 뒤 이 cell을 실행합니다.
현재 output JSON과 backup zip을 읽어서 보기 쉬운 표로 보여줍니다.

```python
from pathlib import Path
import subprocess
import sys

import pandas as pd
from IPython.display import display, Image, Markdown

# ============================================================
# User settings
# ============================================================
PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
BACKUP_ROOT = Path("/kaggle/working/stage2_saved_outputs")

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]
IMAGE_SPECS = ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]

# 현재 가지고 있는 seed에 맞게 둔다.
# single-seed 결과만 있으면 [42], five-seed까지 있으면 [42, 43, 44, 45, 46].
RUN_SEEDS = [42]

EVAL_SPLIT = "test"
OUTPUT_PREFIX = "stage2_grid_view"


def run(cmd, cwd=PROJECT_ROOT):
    """Run one command and stream output."""
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


summary_cmd = [
    sys.executable, "-u",
    "scripts/summarize_stage2_grid_results.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-windows", *map(str, IMAGE_WINDOWS),
    "--return-horizons", *map(str, RETURN_HORIZONS),
    "--image-specs", *IMAGE_SPECS,
    "--run-seeds", *map(str, RUN_SEEDS),
    "--split", EVAL_SPLIT,
    "--backup-root", str(BACKUP_ROOT),
    "--include-backup-zips",
    "--output-prefix", OUTPUT_PREFIX,
]
run(summary_cmd)

seed_csv = PROJECT_ROOT / f"reports/tables/{OUTPUT_PREFIX}_seed_results.csv"
summary_csv = PROJECT_ROOT / f"reports/tables/{OUTPUT_PREFIX}_mean_std_results.csv"
seed_df = pd.read_csv(seed_csv)
summary_df = pd.read_csv(summary_csv)

display(Markdown("# Stage 2 Seed-level Results"))
display(
    seed_df.sort_values(["return_horizon", "image_window", "image_spec", "run_seed"])
)

compact_cols = [
    col for col in [
        "image_window",
        "return_horizon",
        "image_spec",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_minus_majority_mean",
        "roc_auc_mean",
        "f1_mean",
        "brier_score_mean",
        "long_flat_sharpe_net_mean",
        "long_short_sharpe_net_mean",
        "long_flat_annualized_return_net_mean",
        "long_short_annualized_return_net_mean",
    ] if col in summary_df.columns
]

display(Markdown("# Stage 2 Mean/Std Summary"))
display(summary_df[compact_cols].sort_values("accuracy_mean", ascending=False))

display(Markdown("# Top Accuracy"))
display(summary_df[compact_cols].sort_values("accuracy_mean", ascending=False).head(20))

if "roc_auc_mean" in summary_df.columns:
    display(Markdown("# Top ROC-AUC"))
    display(summary_df[compact_cols].sort_values("roc_auc_mean", ascending=False).head(20))

if "long_flat_sharpe_net_mean" in summary_df.columns:
    display(Markdown("# Top Long/Flat Sharpe Net"))
    display(summary_df[compact_cols].sort_values("long_flat_sharpe_net_mean", ascending=False).head(20))

display(Markdown("# Pivot: Accuracy Mean"))
try:
    pivot = summary_df.pivot_table(
        index=["image_window", "return_horizon"],
        columns="image_spec",
        values="accuracy_mean",
    )
    display(pivot)
except Exception as exc:
    print("pivot failed:", exc)

display(Markdown("# Available Grad-CAM Figures"))
gradcam_files = sorted((PROJECT_ROOT / "outputs/stage2/figures").rglob("btc_gradcam_test_2perclass.png"))
for path in gradcam_files[:12]:
    print(path)

# 원하는 Grad-CAM 하나를 바로 보고 싶으면 아래 값을 바꾼다.
SHOW_GRADCAM = True
GRADCAM_EXPERIMENT = "stage2_i20_ohlc_ma_vb_r20"
GRADCAM_SEED = 42
if SHOW_GRADCAM:
    gradcam = (
        PROJECT_ROOT
        / f"outputs/stage2/figures/{GRADCAM_EXPERIMENT}/seed_{GRADCAM_SEED}/gradcam/test/btc_gradcam_test_2perclass.png"
    )
    samples = gradcam.parent / "samples.csv"
    if gradcam.exists():
        display(Markdown(f"## Grad-CAM: {GRADCAM_EXPERIMENT}, seed {GRADCAM_SEED}"))
        display(Image(filename=str(gradcam)))
        if samples.exists():
            sample_df = pd.read_csv(samples)
            view_cols = [
                col for col in [
                    "gradcam_panel_label",
                    "Date",
                    "pred_class",
                    "label",
                    "correct",
                    "prob_down",
                    "prob_up",
                    "future_return",
                    "sample_index",
                ] if col in sample_df.columns
            ]
            display(Markdown("### Grad-CAM sample metadata"))
            display(sample_df[view_cols].sort_values(["gradcam_panel_label", "prob_up"], ascending=[False, False]))
    else:
        print("Grad-CAM missing:", gradcam)
```
