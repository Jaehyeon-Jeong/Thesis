# Kaggle Stage 1 Single-horizon One-cell Runner

## English

Copy this cell into Kaggle when you want a clean one-block execution for one
horizon. It keeps the thesis code modular, but the Kaggle notebook stays simple.

Default:
- horizon: `stage1_i20_r20`
- seed: `42`
- split: `test`
- Grad-CAM year: `2019`
- quick Grad-CAM samples: `2` per class

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import yaml

# ============================================================
# User settings
# ============================================================
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage1/stage1_reimage_reproduction")
DATA_INPUT = Path("/kaggle/input/datasets/moskow/baseline-dataset")
PROJECT_ROOT = Path("/kaggle/working/stage1_reimage_reproduction")
DATA_WORK = Path("/tmp/baseline-dataset")

HORIZON = "stage1_i20_r20"
RUN_SEED = 42
EVAL_SPLIT = "test"
GRADCAM_YEAR = 2019
GRADCAM_SAMPLES_PER_CLASS = 2

# Fast Kaggle setting:
#   - 1024 uses fewer optimizer steps than the paper-style 128 batch.
#   - If Kaggle raises CUDA OOM, set this to 512 and rerun.
#   - For strict reproduction later, set BATCH_SIZE=128, MIXED_PRECISION=False,
#     DATA_PARALLEL=False, FAST_CUDNN=False.
BATCH_SIZE = 1024
NUM_WORKERS = 4
LOG_EVERY_BATCHES = 20
PRELOAD_TRAIN_VAL_IMAGES = True
PRELOAD_CHUNK_SIZE = 8192
MIXED_PRECISION = True
DATA_PARALLEL = True
FAST_CUDNN = True


def run(cmd, cwd=PROJECT_ROOT):
    """Run one shell command and stream output immediately."""
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def copy_or_extract_input(src: Path, dst: Path, expected_child: str | None = None):
    """Copy a Kaggle input folder or extract a Kaggle input zip file."""
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
        if not (candidate / "configs").exists() and expected_child:
            nested = next(tmp.rglob(expected_child), None)
            if nested is not None:
                candidate = nested
        shutil.copytree(candidate, dst)
        shutil.rmtree(tmp)
        return
    raise FileNotFoundError(f"Input must be a folder or .zip file: {src}")


def assert_latest_fast_code(project_root: Path):
    """Fail early if the attached Kaggle code dataset is an old snapshot."""
    required_markers = {
        "src/stage1_reimage/data/label_split.py": "PreloadedHorizonSplitImageDataset",
        "src/stage1_reimage/training/loop.py": "mixed_precision=",
        "src/stage1_reimage/runners/stage1_baseline.py": "preload_train_val_images",
    }
    missing = []
    for rel_path, marker in required_markers.items():
        file_path = project_root / rel_path
        if not file_path.exists() or marker not in file_path.read_text(encoding="utf-8"):
            missing.append(f"{rel_path} missing marker {marker!r}")
    if missing:
        raise RuntimeError(
            "The Kaggle CODE_INPUT snapshot is old. Upload the latest "
            "stage1_reimage_reproduction folder/zip, then rerun. Problems: "
            + "; ".join(missing)
        )


# ============================================================
# 1. Copy code snapshot into /kaggle/working
# ============================================================
copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage1_reimage_reproduction")
assert_latest_fast_code(PROJECT_ROOT)
print(f"Code copied to: {PROJECT_ROOT}", flush=True)


# ============================================================
# 2. Copy data to local temporary disk if possible
# ============================================================
copy_or_extract_input(DATA_INPUT, DATA_WORK, expected_child="baseline-dataset")
print(f"Data copied to: {DATA_WORK}", flush=True)


# ============================================================
# 3. Patch Kaggle config paths and runtime knobs
# ============================================================
config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
cfg = yaml.safe_load(config_path.read_text())

cfg["paths"]["data_root"] = str(DATA_WORK)
cfg["data"]["monthly20_root"] = str(DATA_WORK)
cfg["runtime"]["num_workers"] = NUM_WORKERS
cfg["runtime"]["pin_memory"] = True
cfg["runtime"]["persistent_workers"] = True
cfg["runtime"]["preload_train_val_images"] = PRELOAD_TRAIN_VAL_IMAGES
cfg["runtime"]["preload_chunk_size"] = PRELOAD_CHUNK_SIZE
cfg["training"]["batch_size"] = BATCH_SIZE
cfg["evaluation"]["batch_size"] = BATCH_SIZE
cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES
cfg["training"]["mixed_precision"] = MIXED_PRECISION
cfg["training"]["data_parallel"] = DATA_PARALLEL
cfg["training"]["determinism"]["enabled"] = not FAST_CUDNN
cfg["training"]["determinism"]["cudnn_deterministic"] = not FAST_CUDNN
cfg["training"]["determinism"]["cudnn_benchmark"] = FAST_CUDNN

config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Config patched:", config_path, flush=True)
print(
    "Fast settings:",
    {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "preload_train_val_images": PRELOAD_TRAIN_VAL_IMAGES,
        "mixed_precision": MIXED_PRECISION,
        "data_parallel": DATA_PARALLEL,
        "fast_cudnn": FAST_CUDNN,
    },
    flush=True,
)


# ============================================================
# 4. Data loading check
# ============================================================
run([
    sys.executable, "-u",
    "scripts/check_data_loading.py",
    "--config", "configs/env_kaggle.yaml",
    "--sample-indices", "0", "-1",
])


# ============================================================
# 5. Train one horizon
# ============================================================
run([
    sys.executable, "-u",
    "scripts/run_stage1_baseline.py",
    "--config", "configs/env_kaggle.yaml",
    "--run-mode", "full_single_seed",
    "--horizons", HORIZON,
    "--run-seeds", str(RUN_SEED),
])


# ============================================================
# 6. Evaluate one horizon
# ============================================================
run([
    sys.executable, "-u",
    "scripts/evaluate_stage1_predictions.py",
    "--config", "configs/env_kaggle.yaml",
    "--horizon", HORIZON,
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
])


# ============================================================
# 7. Quick Grad-CAM for visual check
# ============================================================
run([
    sys.executable, "-u",
    "scripts/generate_stage1_gradcam.py",
    "--config", "configs/env_kaggle.yaml",
    "--horizon", HORIZON,
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
    "--year", str(GRADCAM_YEAR),
    "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
    "--write-report-copy",
])

print("\nDONE", flush=True)
print("Outputs:", PROJECT_ROOT / "outputs", flush=True)
```

## 한국어

Kaggle에서 horizon 하나를 깔끔하게 한 셀로 실행하고 싶을 때 이 셀을 복붙합니다.
논문용 core code는 모듈 구조를 유지하지만, Kaggle Notebook에서는 한 블록만 실행하면
됩니다.

기본값:
- horizon: `stage1_i20_r20`
- seed: `42`
- split: `test`
- Grad-CAM year: `2019`
- 빠른 Grad-CAM sample: class당 `2`

위 Python cell을 그대로 Kaggle Notebook에 붙여 실행하면 됩니다.

주의:
- report용 Grad-CAM은 나중에 `GRADCAM_SAMPLES_PER_CLASS = 10`으로 바꿔 다시
  생성합니다.
- `BATCH_SIZE`를 256/512로 키우면 빨라질 수 있지만, 정확한 baseline config와
  달라지므로 speed diagnostic으로 기록해야 합니다.
