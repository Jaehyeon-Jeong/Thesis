# Kaggle Stage 2 Selected Robustness Runner - I20/I60 R20 Five Seeds

## English

This cell runs a selected Stage 2 BTC baseline robustness check with five seeds.

Rationale:
- The full five-seed grid is `180` model runs.
- For the next report, the priority is checking whether the strongest R20
  configurations are stable across random seeds.
- Therefore this selected runner only keeps `I20/R20` and `I60/R20`, while
  still comparing all four image specifications.

Grid:
- image windows: `I20`, `I60`
- return horizon: `R20`
- image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seeds: `42`, `43`, `44`, `45`, `46`

Total model runs: `2 x 1 x 4 x 5 = 40`.

Default:
- `RUN_GRADCAM = False`
- This keeps the selected robustness pass focused on metrics and trading
  outputs first. Generate Grad-CAM later for the final selected configuration.

## 한국어

이 cell은 Stage 2 BTC baseline의 선별된 robustness check를 5개 seed로 실행합니다.

선택 이유:
- 전체 five-seed grid는 `180`개 model run입니다.
- 다음 보고에서는 가장 강하게 나온 R20 configuration이 random seed를 바꿔도
  안정적인지 확인하는 것이 우선입니다.
- 따라서 이 selected runner는 `I20/R20`과 `I60/R20`만 유지하되, 네 가지 image
  specification은 모두 비교합니다.

Grid:
- image window: `I20`, `I60`
- return horizon: `R20`
- image spec: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seed: `42`, `43`, `44`, `45`, `46`

총 model run 수는 `2 x 1 x 4 x 5 = 40`입니다.

기본값:
- `RUN_GRADCAM = False`
- 우선 selected robustness pass는 metric과 trading output에 집중합니다.
- 최종 selected configuration을 확정한 뒤 해당 configuration만 Grad-CAM을 다시
  생성합니다.

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import yaml

import pandas as pd
from IPython.display import display, Markdown

# ============================================================
# User settings
# ============================================================
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage2_saved_outputs")

SOURCE_FILE = "/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv"

# Selected robustness grid:
#   I20/R20 and I60/R20 across all image specs.
#   This is a 40-run stability check, not the full 180-run grid.
IMAGE_WINDOWS = [20, 60]
RETURN_HORIZONS = [20]
IMAGE_SPECS = ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]
RUN_SEEDS = [42, 43, 44, 45, 46]

EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2

# Full interpretability run이면 True. 여기서는 40개 selected robustness run의
# metric/trading result를 먼저 얻는 것이 목적이므로 기본값은 False로 둔다.
# 최종 selected configuration만 Grad-CAM을 다시 생성한다.
RUN_GRADCAM = False

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
SUMMARY_NAME = "stage2_i20_i60_r20_five_seed"


def copy_or_extract_input(src: Path, dst: Path, expected_child: str | None = None):
    """Copy a Kaggle input folder or extract a zip file."""
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        candidate = src / expected_child if expected_child and (src / expected_child).exists() else src
        shutil.copytree(candidate, dst)
        return
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = dst.parent / f"{dst.name}_unzipped"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as archive:
            archive.extractall(tmp)
        candidate = tmp / expected_child if expected_child and (tmp / expected_child).exists() else tmp
        if not (candidate / "scripts").exists() and expected_child:
            nested = next(tmp.rglob(expected_child), None)
            if nested is not None:
                candidate = nested
        shutil.copytree(candidate, dst)
        shutil.rmtree(tmp)
        return
    raise FileNotFoundError(f"Input must be a folder or .zip file: {src}")


def run(cmd, cwd=PROJECT_ROOT):
    """Run one command and stream output."""
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def assert_latest_stage2_code(project_root: Path):
    """Kaggle에 붙인 Stage 2 code snapshot이 최신 grid code인지 확인한다.

    이 check가 필요한 이유:
        Kaggle input dataset은 한 번 업로드하면 read-only snapshot으로 고정된다.
        로컬/GitHub에서 새 script를 추가해도 Kaggle dataset을 다시 upload하지 않으면
        notebook은 이전 snapshot을 복사한다. 그러면 학습 전에 `run_stage2_grid.py`
        같은 파일이 없어서 몇 분 뒤 에러가 난다. 여기서 초반에 바로 실패시킨다.
    """

    required_files = [
        "scripts/run_stage2_grid.py",
        "scripts/summarize_stage2_grid_results.py",
        "scripts/run_stage2_btc_baseline.py",
        "scripts/evaluate_stage2_predictions.py",
        "scripts/evaluate_stage2_trading.py",
        "scripts/generate_stage2_gradcam.py",
        "src/stage2_btc/models/stock_cnn.py",
    ]
    missing = [path for path in required_files if not (project_root / path).exists()]
    if missing:
        available_scripts = sorted(str(path.relative_to(project_root)) for path in (project_root / "scripts").glob("*.py"))
        raise RuntimeError(
            "The attached Kaggle Stage 2 code dataset is old or incomplete. "
            "Upload the latest `stage2_btc_extension` folder/zip as a Kaggle dataset, "
            "then update CODE_INPUT if the path changed. "
            f"Missing: {missing}. Available scripts: {available_scripts}"
        )


copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_latest_stage2_code(PROJECT_ROOT)
print(f"Code copied to: {PROJECT_ROOT}", flush=True)

config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
cfg = yaml.safe_load(config_path.read_text())
cfg["paths"]["data_root"] = str(DATA_ROOT)
cfg["paths"]["source_file"] = SOURCE_FILE
cfg["data"]["source_file"] = SOURCE_FILE
cfg["runtime"]["num_workers"] = NUM_WORKERS
cfg["runtime"]["pin_memory"] = True
cfg["training"]["batch_size"] = BATCH_SIZE
cfg["evaluation"]["batch_size"] = BATCH_SIZE
cfg["training"]["mixed_precision"] = MIXED_PRECISION
cfg["training"]["data_parallel"] = DATA_PARALLEL
cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

print(
    "Selected Stage 2 robustness grid:",
    {
        "image_windows": IMAGE_WINDOWS,
        "return_horizons": RETURN_HORIZONS,
        "image_specs": IMAGE_SPECS,
        "run_seeds": RUN_SEEDS,
        "total_runs": len(IMAGE_WINDOWS) * len(RETURN_HORIZONS) * len(IMAGE_SPECS) * len(RUN_SEEDS),
        "run_gradcam": RUN_GRADCAM,
    },
    flush=True,
)

run([
    sys.executable, "-u",
    "scripts/audit_btc_ohlcv.py",
    "--data-root", str(DATA_ROOT),
    "--output-dir", "reports/data_audit",
])

grid_cmd = [
    sys.executable, "-u",
    "scripts/run_stage2_grid.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-windows", *map(str, IMAGE_WINDOWS),
    "--return-horizons", *map(str, RETURN_HORIZONS),
    "--image-specs", *IMAGE_SPECS,
    "--run-seeds", *map(str, RUN_SEEDS),
    "--split", EVAL_SPLIT,
    "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
    "--backup-root", str(BACKUP_ROOT),
    "--summary-name", SUMMARY_NAME,
]
if SKIP_COMPLETED:
    grid_cmd.append("--skip-completed")
if CONTINUE_ON_ERROR:
    grid_cmd.append("--continue-on-error")
if not RUN_GRADCAM:
    grid_cmd.append("--skip-gradcam")
if SMOKE_TEST:
    grid_cmd.extend(["--max-epochs", "2", "--max-train-rows", "128", "--max-validation-rows", "64", "--max-test-rows", "64"])

run(grid_cmd)

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
    "--output-prefix", SUMMARY_NAME,
]
run(summary_cmd)

seed_csv = PROJECT_ROOT / f"reports/tables/{SUMMARY_NAME}_seed_results.csv"
summary_csv = PROJECT_ROOT / f"reports/tables/{SUMMARY_NAME}_mean_std_results.csv"

display(Markdown("## Selected I20/I60 R20 Five-Seed Results"))
seed_df = pd.read_csv(seed_csv)
display(seed_df.sort_values(["return_horizon", "image_window", "image_spec", "run_seed"]))

display(Markdown("## Mean/Std Summary"))
summary_df = pd.read_csv(summary_csv)
display(summary_df.sort_values("accuracy_mean", ascending=False))

compact_cols = [
    col for col in [
        "image_window",
        "return_horizon",
        "image_spec",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_minus_majority_mean",
        "roc_auc_mean",
        "roc_auc_std",
        "long_flat_sharpe_net_mean",
        "long_short_sharpe_net_mean",
        "long_flat_annualized_return_net_mean",
        "long_short_annualized_return_net_mean",
    ] if col in summary_df.columns
]
display(Markdown("## Compact Summary"))
display(summary_df[compact_cols].sort_values("accuracy_mean", ascending=False))

print("\nDONE", flush=True)
print("Seed results:", seed_csv, flush=True)
print("Mean/std results:", summary_csv, flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```
