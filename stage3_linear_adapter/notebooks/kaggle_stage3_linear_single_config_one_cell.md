# Kaggle Stage 3 Linear Single Config One Cell

## English

Paste the Python cell below into Kaggle. It runs one Stage 3 Linear adapter
experiment end-to-end: data audit path check, training, prediction metrics,
trading metrics, Grad-CAM, output check, and backup zip.

Default run:
- `I60 / R20 / ohlc_ma_vb`
- seed `42`
- adapter dim `128`
- strict batch size `128`

## 한국어

아래 Python cell을 Kaggle에 그대로 붙여넣으면 Stage 3 Linear adapter 실험 하나를
끝까지 실행합니다. 학습, prediction metric, trading metric, Grad-CAM, output check,
backup zip 저장까지 포함합니다.

기본 실행:
- `I60 / R20 / ohlc_ma_vb`
- seed `42`
- adapter dim `128`
- strict batch size `128`

```python
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
import zipfile
import yaml

# ============================================================
# User settings
# ============================================================
STAGE3_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage3/stage3_linear_adapter")
STAGE2_CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
STAGE3_ROOT = Path("/kaggle/working/stage3_linear_adapter")
STAGE2_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage3_saved_outputs")

# Public BTC dataset path. If Kaggle changes the path, run a discovery cell and edit this.
SOURCE_FILE = "/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv"

# Optional: if you uploaded Stage 2 outputs as a dataset/zip, set the path here.
# If left None, Stage 3 still runs. Baseline-vs-Linear Grad-CAM comparison is skipped
# unless Stage 2 outputs already exist under STAGE2_ROOT/outputs/stage2.
STAGE2_OUTPUT_INPUT = None

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEED = 42
ADAPTER_DIM = 128
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
SAVE_BACKUP_ZIPS = True

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
        if expected_child and not (candidate / "scripts").exists():
            nested = next(tmp.rglob(expected_child), None)
            if nested is not None:
                candidate = nested
        shutil.copytree(candidate, dst)
        shutil.rmtree(tmp)
        return
    raise FileNotFoundError(f"Input must be a folder or .zip file: {src}")


def run(cmd, cwd=STAGE3_ROOT):
    """Run one command and stream output."""
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def backup_outputs(phase: str):
    """Save current Stage 3 outputs outside the project folder."""
    outputs_root = STAGE3_ROOT / "outputs" / "stage3"
    if not SAVE_BACKUP_ZIPS or not outputs_root.exists():
        return None
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"stage3_linear_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}_a{ADAPTER_DIM}"
    archive_base = BACKUP_ROOT / f"{name}_seed{RUN_SEED}_{phase}_{timestamp}_outputs"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", outputs_root))
    receipt = {
        "experiment": name,
        "run_seed": RUN_SEED,
        "phase": phase,
        "created_utc": timestamp,
        "archive_path": str(archive_path),
        "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 3),
    }
    receipt_path = BACKUP_ROOT / f"{name}_seed{RUN_SEED}_{phase}_{timestamp}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[backup:{phase}] saved {archive_path}, receipt={receipt_path}", flush=True)
    return archive_path


def run_step(phase: str, cmd):
    """Run a step and save a backup zip after it finishes or fails."""
    try:
        run(cmd)
    finally:
        backup_outputs(phase)


def assert_required_scripts(root: Path):
    required = [
        "scripts/run_stage3_linear.py",
        "scripts/evaluate_stage3_predictions.py",
        "scripts/evaluate_stage3_trading.py",
        "scripts/generate_stage3_gradcam.py",
        "scripts/check_stage3_outputs.py",
    ]
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise RuntimeError("Stage 3 code snapshot is stale. Missing: " + ", ".join(missing))


copy_or_extract_input(STAGE3_CODE_INPUT, STAGE3_ROOT, expected_child="stage3_linear_adapter")
copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_ROOT, expected_child="stage2_btc_extension")
assert_required_scripts(STAGE3_ROOT)
print("Stage 3 code:", STAGE3_ROOT, flush=True)
print("Stage 2 code:", STAGE2_ROOT, flush=True)

if STAGE2_OUTPUT_INPUT is not None:
    copy_or_extract_input(Path(STAGE2_OUTPUT_INPUT), STAGE2_ROOT / "outputs" / "stage2")

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

run([sys.executable, "-u", "scripts/check_stage3_scaffold.py", "--config", "configs/env_kaggle.yaml"])
run([sys.executable, "-u", "scripts/check_stage3_model.py", "--config", "configs/env_kaggle.yaml", "--image-window", str(IMAGE_WINDOW)])

run_step("after_train", [
    sys.executable, "-u", "scripts/run_stage3_linear.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
])

run_step("after_prediction_eval", [
    sys.executable, "-u", "scripts/evaluate_stage3_predictions.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
])

run_step("after_trading_eval", [
    sys.executable, "-u", "scripts/evaluate_stage3_trading.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
])

run_step("after_gradcam", [
    sys.executable, "-u", "scripts/generate_stage3_gradcam.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
    "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
    "--write-report-copy",
])

run_step("after_output_check", [
    sys.executable, "-u", "scripts/check_stage3_outputs.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
    "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
])

print("\nDONE", flush=True)
print("Outputs:", STAGE3_ROOT / "outputs/stage3", flush=True)
print("Backup zips:", BACKUP_ROOT, flush=True)
```

