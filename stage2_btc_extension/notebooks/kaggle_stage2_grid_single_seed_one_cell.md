# Kaggle Stage 2 BTC Grid Runner - Single Seed

## English

This cell runs the full Stage 2 BTC baseline grid once with `seed=42`.

Grid:
- image windows: `I5`, `I20`, `I60`
- return horizons: `R5`, `R20`, `R60`
- image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seeds: `42`

Total model runs: `3 x 3 x 4 x 1 = 36`.

## 한국어

이 cell은 Stage 2 BTC baseline grid를 `seed=42` 한 번으로 실행합니다.

Grid:
- image window: `I5`, `I20`, `I60`
- return horizon: `R5`, `R20`, `R60`
- image spec: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seed: `42`

총 model run 수는 `3 x 3 x 4 x 1 = 36`입니다.

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

# Public BTC dataset path. If Kaggle path changes, run the discovery cell and edit this.
SOURCE_FILE = "/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv"

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]
IMAGE_SPECS = ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]
RUN_SEEDS = [42]

EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
RUN_GRADCAM = True
SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True

# Smoke check only. For full grid, keep False.
SMOKE_TEST = False

# Strict Stage 2 default. Do not change unless recording a speed diagnostic.
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
    "--summary-name", "stage2_grid_single_seed",
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
    "--output-prefix", "stage2_grid_single_seed",
]
run(summary_cmd)

seed_csv = PROJECT_ROOT / "reports/tables/stage2_grid_single_seed_seed_results.csv"
summary_csv = PROJECT_ROOT / "reports/tables/stage2_grid_single_seed_mean_std_results.csv"

display(Markdown("## Seed-level Results"))
seed_df = pd.read_csv(seed_csv)
display(seed_df.sort_values(["return_horizon", "image_window", "image_spec"]))

display(Markdown("## Mean/Std Summary"))
summary_df = pd.read_csv(summary_csv)
display(summary_df.sort_values("accuracy_mean", ascending=False))

print("\nDONE", flush=True)
print("Seed results:", seed_csv, flush=True)
print("Mean/std results:", summary_csv, flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```
