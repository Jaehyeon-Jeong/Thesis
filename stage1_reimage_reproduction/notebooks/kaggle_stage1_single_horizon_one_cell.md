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

# Keep batch_size=128 for closest Stage 1 config.
# For speed diagnostics only, try 256 or 512 and record that it is not the
# exact baseline config.
BATCH_SIZE = 128
NUM_WORKERS = 4
LOG_EVERY_BATCHES = 20


def run(cmd, cwd=PROJECT_ROOT):
    """Run one shell command and stream output immediately."""
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


# ============================================================
# 1. Copy code snapshot into /kaggle/working
# ============================================================
if PROJECT_ROOT.exists():
    shutil.rmtree(PROJECT_ROOT)
shutil.copytree(CODE_INPUT, PROJECT_ROOT)
print(f"Code copied to: {PROJECT_ROOT}", flush=True)


# ============================================================
# 2. Copy data to local temporary disk if possible
# ============================================================
if DATA_WORK.exists():
    shutil.rmtree(DATA_WORK)
shutil.copytree(DATA_INPUT, DATA_WORK)
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
cfg["training"]["batch_size"] = BATCH_SIZE
cfg["evaluation"]["batch_size"] = BATCH_SIZE
cfg["training"]["log_every_batches"] = LOG_EVERY_BATCHES

config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Config patched:", config_path, flush=True)


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
