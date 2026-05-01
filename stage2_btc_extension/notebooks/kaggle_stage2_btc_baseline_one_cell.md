# Kaggle Stage 2 BTC Baseline One-cell Runner

## English

This is the Stage 2 baseline runner interface.

Current status:
- executable after `2-I10` code update.
- The same cell can run a smoke check or one full BTC baseline tuple.

The design follows the Stage 1 one-cell runner:
- copy or extract the code snapshot into `/kaggle/working`
- locate the BTC daily CSV
- patch config paths and runtime knobs
- run one BTC experiment tuple
- evaluate predictions, trading metrics, and quick Grad-CAM

## 한국어

이 문서는 Stage 2 baseline runner interface입니다.

현재 상태:
- `2-I10` code update 이후 실행 가능한 runner입니다.
- 같은 cell로 smoke check 또는 BTC baseline tuple 하나를 full run할 수 있습니다.

설계는 Stage 1 one-cell runner를 따릅니다.
- code snapshot을 `/kaggle/working`으로 복사 또는 압축 해제
- BTC daily CSV 찾기
- config path와 runtime knob patch
- BTC experiment tuple 하나 실행
- prediction, trading metric, quick Grad-CAM 평가

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
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect btc_1d_data_2018_to_2025.csv under /kaggle/input.
SOURCE_FILE = ""

IMAGE_WINDOW = 20
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEED = 42
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
SMOKE_TEST = False

# Strict Stage 2 default.
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


def assert_required_scripts(project_root: Path):
    """Fail early if the code snapshot is missing Stage 2 implementation scripts."""
    required = [
        "scripts/audit_btc_ohlcv.py",
        "scripts/run_stage2_btc_baseline.py",
        "scripts/evaluate_stage2_predictions.py",
        "scripts/evaluate_stage2_trading.py",
        "scripts/generate_stage2_gradcam.py",
        "scripts/check_stage2_outputs.py",
    ]
    missing = [path for path in required if not (project_root / path).exists()]
    if missing:
        raise RuntimeError(
            "The attached Stage 2 code snapshot is incomplete. "
            "Missing scripts: " + ", ".join(missing)
        )


copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_required_scripts(PROJECT_ROOT)
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

smoke_train_args = []
smoke_data_args = []
if SMOKE_TEST:
    smoke_train_args = [
        "--max-epochs", "2",
        "--max-train-rows", "128",
        "--max-validation-rows", "64",
        "--max-test-rows", "64",
    ]
    smoke_data_args = [
        "--max-train-rows", "128",
        "--max-validation-rows", "64",
        "--max-test-rows", "64",
    ]

run([
    sys.executable, "-u",
    "scripts/audit_btc_ohlcv.py",
    "--data-root", str(DATA_ROOT),
    "--output-dir", "reports/data_audit",
])

run([
    sys.executable, "-u",
    "scripts/run_stage2_btc_baseline.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
] + smoke_train_args)

run([
    sys.executable, "-u",
    "scripts/evaluate_stage2_predictions.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
] + smoke_data_args)

run([
    sys.executable, "-u",
    "scripts/evaluate_stage2_trading.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
])

run([
    sys.executable, "-u",
    "scripts/generate_stage2_gradcam.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
    "--samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
    "--write-report-copy",
] + smoke_data_args)

run([
    sys.executable, "-u",
    "scripts/check_stage2_outputs.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
    "--gradcam-samples-per-class", str(GRADCAM_SAMPLES_PER_CLASS),
])

print("\nDONE", flush=True)
print("Outputs:", PROJECT_ROOT / "outputs" / "stage2", flush=True)
```
