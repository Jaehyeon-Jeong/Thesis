# Kaggle Stage 2 Best-Config Grad-CAM - 10 Per Predicted Class

## English

Use this cell after the Stage 2 single-seed grid has finished. It generates a
Figure-13-style Grad-CAM figure for the best single-seed configuration:

`I60 / R20 / ohlc_ma_vb / seed 42`

The output contains:
- `Predicted Up`: 10 images
- `Predicted Down`: 10 images
- Total: 20 original chart images plus layer-wise Grad-CAM rows

## 한국어

Stage 2 single-seed grid가 끝난 뒤 이 cell을 실행합니다. Single-seed 기준 best
configuration인 아래 조합에 대해 Figure 13 스타일 Grad-CAM을 생성합니다.

`I60 / R20 / ohlc_ma_vb / seed 42`

Output 구성:
- `Predicted Up`: 10개 이미지
- `Predicted Down`: 10개 이미지
- 총 20개 원본 chart image와 layer별 Grad-CAM row

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import yaml
from IPython.display import display, Image, Markdown

# ============================================================
# User settings
# ============================================================
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")
PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
BACKUP_ROOT = Path("/kaggle/working/stage2_saved_outputs")
DATA_ROOT = Path("/kaggle/input")
SOURCE_FILE = "/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv"

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEED = 42
EVAL_SPLIT = "test"
SAMPLES_PER_CLASS = 10

EXPERIMENT = f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def copy_or_extract_input(src: Path, dst: Path, expected_child: str | None = None):
    """Copy the Stage 2 code snapshot into `/kaggle/working` if needed."""

    if dst.exists():
        return
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


def patch_kaggle_config():
    """Make sure the copied Stage 2 config points at the Kaggle BTC input."""

    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
    cfg["data"]["source_file"] = SOURCE_FILE
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Config patched:", config_path)


def run(cmd, cwd=PROJECT_ROOT):
    """Run one command and stream output."""

    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def restore_experiment_from_backup_if_needed():
    """Restore checkpoint/prediction files from stage2_saved_outputs if needed.

    Kaggle working folders can be reset, while `stage2_saved_outputs` may still
    contain per-experiment zip backups. This function extracts the latest backup
    for the selected experiment when checkpoint or prediction files are missing.
    """

    checkpoint = PROJECT_ROOT / f"outputs/stage2/checkpoints/{EXPERIMENT}/seed_{RUN_SEED}/best.pt"
    predictions = PROJECT_ROOT / f"outputs/stage2/predictions/{EXPERIMENT}/seed_{RUN_SEED}/{EVAL_SPLIT}_predictions.csv"
    if checkpoint.exists() and predictions.exists():
        print("checkpoint/prediction already available")
        return

    candidates = sorted(
        BACKUP_ROOT.glob(f"{EXPERIMENT}_seed{RUN_SEED}_*_outputs.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Missing checkpoint/prediction and no backup zip found. "
            f"Expected checkpoint={checkpoint}, predictions={predictions}, "
            f"backup pattern={BACKUP_ROOT}/{EXPERIMENT}_seed{RUN_SEED}_*_outputs.zip"
        )

    latest = candidates[0]
    print("restoring from backup:", latest)
    with zipfile.ZipFile(latest) as archive:
        archive.extractall(PROJECT_ROOT)

    if not checkpoint.exists() or not predictions.exists():
        raise FileNotFoundError(
            "Backup was extracted, but required files are still missing. "
            f"checkpoint_exists={checkpoint.exists()}, predictions_exists={predictions.exists()}"
        )


copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage2_btc_extension")
patch_kaggle_config()
restore_experiment_from_backup_if_needed()

run([
    sys.executable, "-u",
    "scripts/generate_stage2_gradcam.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--split", EVAL_SPLIT,
    "--samples-per-class", str(SAMPLES_PER_CLASS),
    "--write-report-copy",
])

figure = (
    PROJECT_ROOT
    / f"outputs/stage2/figures/{EXPERIMENT}/seed_{RUN_SEED}/gradcam/{EVAL_SPLIT}/btc_gradcam_{EVAL_SPLIT}_{SAMPLES_PER_CLASS}perclass.png"
)
samples = figure.parent / "samples.csv"

report_dir = PROJECT_ROOT / "reports/figures/gradcam"
report_dir.mkdir(parents=True, exist_ok=True)
report_figure = report_dir / "stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass.png"
report_samples = report_dir / "stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass_samples.csv"
shutil.copy2(figure, report_figure)
shutil.copy2(samples, report_samples)

bundle_dir = Path("/kaggle/working/stage2_best_gradcam_10perclass")
bundle_zip = Path("/kaggle/working/stage2_best_gradcam_10perclass.zip")
if bundle_dir.exists():
    shutil.rmtree(bundle_dir)
bundle_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(report_figure, bundle_dir / report_figure.name)
shutil.copy2(report_samples, bundle_dir / report_samples.name)

if bundle_zip.exists():
    bundle_zip.unlink()
shutil.make_archive(str(bundle_zip).replace(".zip", ""), "zip", bundle_dir)

display(Markdown(f"# Grad-CAM 10 per class: {EXPERIMENT}, seed {RUN_SEED}"))
display(Image(filename=str(report_figure)))
display(Markdown("## Sample metadata"))
display(pd.read_csv(report_samples))

print("figure:", report_figure)
print("samples:", report_samples)
print("download zip:", bundle_zip)
print("zip size MB:", round(bundle_zip.stat().st_size / 1024 / 1024, 3))
```
