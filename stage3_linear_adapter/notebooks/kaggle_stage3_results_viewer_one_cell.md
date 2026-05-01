# Kaggle Stage 3 Results Viewer One Cell

## English

Use this after a Stage 3 run to rebuild result tables and display available
Grad-CAM files.

## 한국어

Stage 3 run 이후 결과표를 다시 만들고 저장된 Grad-CAM 파일을 확인할 때 사용합니다.

```python
from pathlib import Path
import subprocess
import sys

import pandas as pd
from IPython.display import display, Image, Markdown

PROJECT_ROOT = Path("/kaggle/working/stage3_linear_adapter")
BACKUP_ROOT = Path("/kaggle/working/stage3_saved_outputs")

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]
IMAGE_SPECS = ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]
RUN_SEEDS = [42]
ADAPTER_DIM = 128
EVAL_SPLIT = "test"
OUTPUT_PREFIX = "stage3_grid_view"


def run(cmd, cwd=PROJECT_ROOT):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


summary_cmd = [
    sys.executable, "-u", "scripts/summarize_stage3_grid_results.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-windows", *map(str, IMAGE_WINDOWS),
    "--return-horizons", *map(str, RETURN_HORIZONS),
    "--image-specs", *IMAGE_SPECS,
    "--run-seeds", *map(str, RUN_SEEDS),
    "--adapter-dim", str(ADAPTER_DIM),
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

compact_cols = [
    col for col in [
        "image_window",
        "return_horizon",
        "image_spec",
        "adapter_dim",
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

display(Markdown("# Stage 3 Seed-level Results"))
display(seed_df.sort_values(["return_horizon", "image_window", "image_spec", "run_seed"]))
display(Markdown("# Stage 3 Summary Sorted by Accuracy"))
display(summary_df[compact_cols].sort_values("accuracy_mean", ascending=False))

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

display(Markdown("# Available Stage 3 Grad-CAM Figures"))
gradcam_files = sorted((PROJECT_ROOT / "outputs/stage3/figures").rglob("btc_linear_gradcam_test_2perclass.png"))
for path in gradcam_files[:20]:
    print(path)

SHOW_GRADCAM = True
GRADCAM_EXPERIMENT = "stage3_linear_i60_ohlc_ma_vb_r20_a128"
GRADCAM_SEED = 42
if SHOW_GRADCAM:
    gradcam = (
        PROJECT_ROOT
        / f"outputs/stage3/figures/{GRADCAM_EXPERIMENT}/seed_{GRADCAM_SEED}/gradcam/test/btc_linear_gradcam_test_2perclass.png"
    )
    if gradcam.exists():
        display(Markdown(f"## Grad-CAM: {GRADCAM_EXPERIMENT}, seed {GRADCAM_SEED}"))
        display(Image(filename=str(gradcam)))
    else:
        print("Grad-CAM missing:", gradcam)
```

