# Kaggle Stage 2 BTC OHLCV Audit One Cell

## English

Paste this Python cell into Kaggle after attaching:
- the Stage 2 code snapshot dataset
- the BTC OHLCV Kaggle dataset

It copies the Stage 2 code into `/kaggle/working`, scans `/kaggle/input` for the
BTC daily CSV, and writes small audit outputs under:

```text
/kaggle/working/stage2_btc_extension/reports/data_audit/
```

## 한국어

Kaggle에서 아래 Python cell을 그대로 붙여넣습니다. 먼저 Kaggle Notebook에 다음을
attach해야 합니다.
- Stage 2 code snapshot dataset
- BTC OHLCV Kaggle dataset

이 cell은 Stage 2 코드를 `/kaggle/working`으로 복사하고, `/kaggle/input` 아래에서
BTC daily CSV를 찾은 뒤, 작은 audit output을 아래에 저장합니다.

```text
/kaggle/working/stage2_btc_extension/reports/data_audit/
```

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# ============================================================
# User settings
# ============================================================
# Stage 2 folder or zip inside Kaggle input.
# If your uploaded code dataset path is different, update this value.
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension")

PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect btc_1d_data_2018_to_2025.csv under /kaggle/input.
SOURCE_FILE = ""


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

cmd = [
    sys.executable,
    "-u",
    "scripts/audit_btc_ohlcv.py",
    "--data-root",
    str(DATA_ROOT),
    "--output-dir",
    "reports/data_audit",
]
if SOURCE_FILE:
    cmd += ["--source-file", SOURCE_FILE]

run(cmd)

report_path = PROJECT_ROOT / "reports" / "data_audit" / "btc_ohlcv_audit.md"
print("\n===== AUDIT REPORT =====\n", flush=True)
print(report_path.read_text(), flush=True)
```
