# Kaggle Stage 2 BTC Grid Runner - Five Seeds

## English

This cell runs the full Stage 2 BTC baseline grid with five seeds.

Grid:
- image windows: `I5`, `I20`, `I60`
- return horizons: `R5`, `R20`, `R60`
- image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seeds: `42`, `43`, `44`, `45`, `46`

Total model runs: `3 x 3 x 4 x 5 = 180`.

## 한국어

이 cell은 Stage 2 BTC baseline grid를 5개 seed로 실행합니다.

Grid:
- image window: `I5`, `I20`, `I60`
- return horizon: `R5`, `R20`, `R60`
- image spec: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seed: `42`, `43`, `44`, `45`, `46`

총 model run 수는 `3 x 3 x 4 x 5 = 180`입니다.

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

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]
IMAGE_SPECS = ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]
RUN_SEEDS = [42, 43, 44, 45, 46]

EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2

# Full interpretability run이면 True. 너무 오래 걸리면 False로 두고 metric/trading을
# 먼저 만든 뒤, 선택된 configuration만 Grad-CAM을 다시 생성한다.
RUN_GRADCAM = True

SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
SMOKE_TEST = False

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False


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


copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage2_btc_extension")
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
    "--summary-name", "stage2_grid_five_seed",
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
    "--output-prefix", "stage2_grid_five_seed",
]
run(summary_cmd)

seed_csv = PROJECT_ROOT / "reports/tables/stage2_grid_five_seed_seed_results.csv"
summary_csv = PROJECT_ROOT / "reports/tables/stage2_grid_five_seed_mean_std_results.csv"

display(Markdown("## Seed-level Results"))
seed_df = pd.read_csv(seed_csv)
display(seed_df.sort_values(["return_horizon", "image_window", "image_spec", "run_seed"]))

display(Markdown("## Mean/Std Summary"))
summary_df = pd.read_csv(summary_csv)
display(summary_df.sort_values("accuracy_mean", ascending=False))

print("\nDONE", flush=True)
print("Seed results:", seed_csv, flush=True)
print("Mean/std results:", summary_csv, flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```
