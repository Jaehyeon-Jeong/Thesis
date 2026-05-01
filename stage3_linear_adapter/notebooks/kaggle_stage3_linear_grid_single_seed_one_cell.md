# Kaggle Stage 3 Linear Grid Single Seed One Cell

## English

Runs the Stage 3 single-seed grid:
`3 image windows x 3 return horizons x 4 image specs x seed 42 = 36 runs`.

## 한국어

Stage 3 single-seed grid를 실행합니다:
`3 image window x 3 return horizon x 4 image spec x seed 42 = 36개 run`.

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import yaml

import pandas as pd
from IPython.display import display, Markdown

STAGE3_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage3/stage3_linear_adapter")
STAGE2_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
STAGE3_ROOT = Path("/kaggle/working/stage3_linear_adapter")
STAGE2_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage3_saved_outputs")
SOURCE_FILE = "/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv"

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]
IMAGE_SPECS = ["ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb"]
RUN_SEEDS = [42]
ADAPTER_DIM = 128
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
RUN_GRADCAM = True
SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True

BATCH_SIZE = 128
NUM_WORKERS = 2
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False


def copy_or_extract_input(src: Path, dst: Path, expected_child: str | None = None):
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
        if expected_child and not (candidate / "scripts").exists():
            nested = next(tmp.rglob(expected_child), None)
            if nested is not None:
                candidate = nested
        shutil.copytree(candidate, dst)
        shutil.rmtree(tmp)
        return
    raise FileNotFoundError(f"Input must be a folder or .zip file: {src}")


def run(cmd, cwd=STAGE3_ROOT):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


copy_or_extract_input(STAGE3_CODE_INPUT, STAGE3_ROOT, expected_child="stage3_linear_adapter")
copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_ROOT, expected_child="stage2_btc_extension")
required = [
    "scripts/run_stage3_grid.py",
    "scripts/summarize_stage3_grid_results.py",
    "scripts/run_stage3_linear.py",
]
missing = [path for path in required if not (STAGE3_ROOT / path).exists()]
if missing:
    raise RuntimeError("Stage 3 Kaggle code snapshot is stale. Missing: " + ", ".join(missing))

config_path = STAGE3_ROOT / "configs" / "env_kaggle.yaml"
cfg = yaml.safe_load(config_path.read_text())
cfg["paths"]["project_root"] = str(STAGE3_ROOT)
cfg["paths"]["data_root"] = str(DATA_ROOT)
cfg["paths"]["source_file"] = SOURCE_FILE
cfg["paths"]["output_root"] = str(STAGE3_ROOT / "outputs/stage3")
cfg["paths"]["checkpoint_root"] = str(STAGE3_ROOT / "outputs/stage3/checkpoints")
cfg["paths"]["metrics_root"] = str(STAGE3_ROOT / "outputs/stage3/metrics")
cfg["paths"]["predictions_root"] = str(STAGE3_ROOT / "outputs/stage3/predictions")
cfg["paths"]["figures_root"] = str(STAGE3_ROOT / "outputs/stage3/figures")
cfg["paths"]["run_manifest_root"] = str(STAGE3_ROOT / "outputs/stage3/run_manifests")
cfg["paths"]["reports_root"] = str(STAGE3_ROOT / "reports")
cfg["paths"]["tables_root"] = str(STAGE3_ROOT / "reports/tables")
cfg["data"]["source_file"] = SOURCE_FILE
cfg["stage2_dependency"]["project_root"] = str(STAGE2_ROOT)
cfg["stage2_dependency"]["src_path"] = str(STAGE2_ROOT / "src")
cfg["stage2_dependency"]["baseline_output_root"] = str(STAGE2_ROOT / "outputs/stage2")
cfg["linear_adapter"]["adapter_dim"] = ADAPTER_DIM
cfg["runtime"]["num_workers"] = NUM_WORKERS
cfg["runtime"]["pin_memory"] = True
cfg["runtime"]["persistent_workers"] = NUM_WORKERS > 0
cfg["training"]["batch_size"] = BATCH_SIZE
cfg["evaluation"]["batch_size"] = BATCH_SIZE
cfg["training"]["mixed_precision"] = MIXED_PRECISION
cfg["training"]["data_parallel"] = DATA_PARALLEL
cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

grid_cmd = [
    sys.executable, "-u", "scripts/run_stage3_grid.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-windows", *map(str, IMAGE_WINDOWS),
    "--return-horizons", *map(str, RETURN_HORIZONS),
    "--image-specs", *IMAGE_SPECS,
    "--run-seeds", *map(str, RUN_SEEDS),
    "--adapter-dim", str(ADAPTER_DIM),
    "--split", EVAL_SPLIT,
    "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
    "--backup-root", str(BACKUP_ROOT),
    "--summary-name", "stage3_grid_single_seed",
]
if SKIP_COMPLETED:
    grid_cmd.append("--skip-completed")
if CONTINUE_ON_ERROR:
    grid_cmd.append("--continue-on-error")
if not RUN_GRADCAM:
    grid_cmd.append("--skip-gradcam")
run(grid_cmd)

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
    "--output-prefix", "stage3_grid_single_seed",
]
run(summary_cmd)

seed_csv = STAGE3_ROOT / "reports/tables/stage3_grid_single_seed_seed_results.csv"
summary_csv = STAGE3_ROOT / "reports/tables/stage3_grid_single_seed_mean_std_results.csv"
display(Markdown("## Stage 3 Seed-level Results"))
display(pd.read_csv(seed_csv).sort_values(["return_horizon", "image_window", "image_spec"]))
display(Markdown("## Stage 3 Mean/Std Summary"))
summary_df = pd.read_csv(summary_csv)
display(summary_df.sort_values("accuracy_mean", ascending=False))

print("\nDONE", flush=True)
print("Seed results:", seed_csv, flush=True)
print("Mean/std results:", summary_csv, flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```

