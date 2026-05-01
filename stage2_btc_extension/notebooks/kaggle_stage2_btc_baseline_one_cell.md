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

Kaggle setup direction:
1. Create a new Kaggle Notebook.
2. Turn on GPU. T4 is enough for Stage 2 first runs because the BTC dataset is
   much smaller than the Stage 1 stock shard.
3. Add a Stage 2 code dataset. This dataset must contain the
   `stage2_btc_extension` folder with `configs/`, `src/`, `scripts/`, and
   `notebooks/`.
4. Add the BTC OHLCV data. Either attach the public Kaggle dataset
   `novandraanugrah/bitcoin-historical-datasets-2018-2024`, or upload a small
   private Kaggle dataset that contains `btc_1d_data_2018_to_2025.csv`.
5. Do not upload a separate MA file. Stage 2 computes the 5/20/60-day simple
   moving average from the BTC `Close` column inside the code.
6. Run the input discovery cell below, then set `CODE_INPUT` and, only if
   needed, `SOURCE_FILE`.

Input discovery cell:

```python
from pathlib import Path

for p in Path("/kaggle/input").glob("*"):
    print("\nINPUT:", p)
    for child in list(p.glob("*"))[:15]:
        print(" ", child)
```

How to set paths:
- If discovery shows `/kaggle/input/thesis-stage2-code/stage2_btc_extension`,
  set `CODE_INPUT = Path("/kaggle/input/thesis-stage2-code/stage2_btc_extension")`.
- If discovery shows the current workspace-style path
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`, the default
  `CODE_INPUT` below is already correct.
- Keep `DATA_ROOT = Path("/kaggle/input")`.
- Keep `SOURCE_FILE = ""` unless auto-detection fails. If it fails, set the
  exact CSV path printed by the discovery cell.
- First run with `SMOKE_TEST = True`. After that passes, set
  `SMOKE_TEST = False` for the full run.

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

Kaggle 실행 지시:
1. Kaggle에서 새 Notebook을 만듭니다.
2. GPU를 켭니다. Stage 2는 BTC sample 수가 Stage 1 주식 shard보다 훨씬 작아서
   첫 실행은 T4로 충분합니다.
3. Stage 2 code dataset을 추가합니다. 이 dataset 안에는
   `stage2_btc_extension` 폴더가 있어야 하고, 그 안에 `configs/`, `src/`,
   `scripts/`, `notebooks/`가 있어야 합니다.
4. BTC OHLCV data를 추가합니다. 방법은 둘 중 하나입니다.
   - Kaggle public dataset `novandraanugrah/bitcoin-historical-datasets-2018-2024`
     를 Notebook input으로 attach합니다.
   - 또는 로컬에 받은 `btc_1d_data_2018_to_2025.csv`를 작은 private Kaggle
     dataset으로 업로드한 뒤 attach합니다.
5. MA 파일은 따로 업로드하지 않습니다. Stage 2 코드는 BTC `Close` column으로
   5/20/60-day simple moving average를 직접 계산합니다.
6. 아래 input discovery cell을 먼저 실행한 뒤, 출력된 경로에 맞춰 `CODE_INPUT`과
   필요 시 `SOURCE_FILE`만 고칩니다.

Input discovery cell:

```python
from pathlib import Path

for p in Path("/kaggle/input").glob("*"):
    print("\nINPUT:", p)
    for child in list(p.glob("*"))[:15]:
        print(" ", child)
```

경로 설정법:
- discovery 결과가 `/kaggle/input/thesis-stage2-code/stage2_btc_extension`이면
  `CODE_INPUT = Path("/kaggle/input/thesis-stage2-code/stage2_btc_extension")`로
  바꿉니다.
- discovery 결과가 현재 작업 예시처럼
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`이면 아래 기본값을
  그대로 쓰면 됩니다.
- `DATA_ROOT = Path("/kaggle/input")`는 그대로 둡니다.
- `SOURCE_FILE = ""`는 자동 탐색입니다. 자동 탐색이 실패할 때만 discovery에 찍힌
  정확한 CSV path로 바꿉니다.
- 처음에는 `SMOKE_TEST = True`로 실행합니다. 통과하면 `SMOKE_TEST = False`로
  full run을 실행합니다.

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
# Kaggle input path that contains the Stage 2 code snapshot.
# 먼저 위 discovery cell로 실제 Kaggle input path를 확인한 뒤 필요하면 바꿉니다.
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage2_saved_outputs")

# Leave empty to auto-detect btc_1d_data_2018_to_2025.csv under /kaggle/input.
# MA는 별도 파일을 읽지 않습니다. BTC Close에서 window별 SMA를 코드가 계산합니다.
SOURCE_FILE = ""

IMAGE_WINDOW = 20
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEED = 42
EVAL_SPLIT = "test"
GRADCAM_SAMPLES_PER_CLASS = 2
SMOKE_TEST = False
SAVE_BACKUP_ZIPS = True

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


def experiment_name() -> str:
    """Stage 2 experiment directory name을 runner 설정값에서 만든다."""

    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def backup_stage2_outputs(phase: str):
    """현재 Stage 2 output을 `PROJECT_ROOT` 밖에 zip으로 저장한다.

    왜 필요한가:
        Kaggle one-cell runner는 실행 시작 시 code snapshot을 `/kaggle/working`으로
        다시 복사한다. 이후 다른 model/window/spec을 실행하면서 project folder를
        새로 만들면 이전 output이 사라질 수 있다. 이 함수는 backup zip을
        `/kaggle/working/stage2_saved_outputs`에 저장해서 결과를 보존한다.
    """

    if not SAVE_BACKUP_ZIPS:
        return None

    outputs_root = PROJECT_ROOT / "outputs" / "stage2"
    if not outputs_root.exists():
        print(f"[backup:{phase}] skip: outputs directory does not exist yet", flush=True)
        return None

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = experiment_name()
    archive_base = BACKUP_ROOT / f"{name}_seed{RUN_SEED}_{phase}_{timestamp}_outputs"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", outputs_root))
    receipt = {
        "experiment": name,
        "image_window": IMAGE_WINDOW,
        "image_spec": IMAGE_SPEC,
        "return_horizon": RETURN_HORIZON,
        "run_seed": RUN_SEED,
        "phase": phase,
        "created_utc": timestamp,
        "archive_path": str(archive_path),
        "archive_size_mb": round(archive_path.stat().st_size / (1024 * 1024), 3),
        "project_root": str(PROJECT_ROOT),
        "outputs_root": str(outputs_root),
    }
    receipt_path = BACKUP_ROOT / f"{name}_seed{RUN_SEED}_{phase}_{timestamp}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        f"[backup:{phase}] saved {archive_path} "
        f"({receipt['archive_size_mb']} MB), receipt={receipt_path}",
        flush=True,
    )
    return archive_path


def run_step(step_name: str, cmd):
    """단계 실행 후 성공/실패와 관계없이 현재 outputs를 backup한다."""

    try:
        run(cmd)
    finally:
        backup_stage2_outputs(step_name)


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

run_step("after_train", [
    sys.executable, "-u",
    "scripts/run_stage2_btc_baseline.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
] + smoke_train_args)

run_step("after_prediction_eval", [
    sys.executable, "-u",
    "scripts/evaluate_stage2_predictions.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
] + smoke_data_args)

run_step("after_trading_eval", [
    sys.executable, "-u",
    "scripts/evaluate_stage2_trading.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
])

run_step("after_gradcam", [
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

run_step("after_output_check", [
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
print("Backup zips:", BACKUP_ROOT, flush=True)
```
