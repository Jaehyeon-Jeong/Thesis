# Kaggle Stage 4 N13-3 Public RORO Context Features One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
- Stage 2 BTC code snapshot:
  `/kaggle/input/datasets/moskow/stage2/stage2_btc_extension`
- BTC OHLCV Kaggle dataset:
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`

This runner executes 4-N13-3 only:

```text
public macro sources
-> risk-off aligned components
-> train-only PCA/SVD first component
-> RORO proxy trailing 20/60 features
-> train-only imputation/clipping/z-score
-> prebuilt context artifact for N13-4
```

The KC Fed official RORO daily/weekly files are kept in
`data_inventory/roro_public/kc_fed_official/`, but they currently start in
June 2023 and therefore do not cover the Stage 4 train split. N13-3 uses the
longer-history public-source proxy when the FRED files are available.

```python
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
import yaml
from IPython.display import display, Markdown

# ============================================================
# User settings
# ============================================================
CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n13_3_latest.zip"),
]
STAGE2_CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage4/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension"),
]
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect BTC OHLCV under /kaggle/input.
SOURCE_FILE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEED = 42

# Conservative no-leakage default:
# use macro source date <= BTC image end date t - 1 day.
ASOF_LAG_DAYS = 1

# Include raw risk-off component features beside the six proxy features.
# N13-4 can later filter to proxy-only or component-only sets.
INCLUDE_COMPONENT_FEATURES = True

# Set True only if you want Kaggle to refresh official public CSVs.
# If Kaggle internet is disabled, keep False and attach a snapshot that already
# contains data_inventory/roro_public/raw/*.csv.
REFRESH_DOWNLOADS = False

OUTPUT_PREFIX = "stage4_roro_context"
FEATURE_SET_NAME = "public_roro_pca_lag1_w20_60"
BUNDLE_PATH = Path("/kaggle/working/stage4_n13_3_public_roro_context_features_bundle.zip")


def first_existing(candidates):
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in candidates))


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


def run(cmd, cwd=PROJECT_ROOT, capture=False, check=True):
    cmd = [str(item) for item in cmd]
    if not capture:
        print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def assert_latest_code():
    required = {
        "scripts/build_stage4_roro_context_features.py": "KC Fed-inspired public-data RORO proxy builder",
        "src/stage4_film/runners/context_experiment.py": "prebuilt_context_name",
    }
    missing = []
    for rel_path, marker in required.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            missing.append(f"{rel_path} missing")
            continue
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            missing.append(f"{rel_path} missing marker {marker!r}")
    if missing:
        raise RuntimeError(
            "Stage 4 code snapshot is stale/incomplete. Upload the latest "
            "stage4_film_conditioning folder/zip. Problems: " + "; ".join(missing)
        )


def patch_kaggle_config():
    config_path = PROJECT_ROOT / "configs" / "env_kaggle.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cfg["paths"]["project_root"] = str(PROJECT_ROOT)
    cfg["paths"]["data_root"] = str(DATA_ROOT)
    cfg["paths"]["source_file"] = SOURCE_FILE
    cfg["paths"]["output_root"] = str(PROJECT_ROOT / "outputs/stage4")
    cfg["paths"]["checkpoint_root"] = str(PROJECT_ROOT / "outputs/stage4/checkpoints")
    cfg["paths"]["metrics_root"] = str(PROJECT_ROOT / "outputs/stage4/metrics")
    cfg["paths"]["predictions_root"] = str(PROJECT_ROOT / "outputs/stage4/predictions")
    cfg["paths"]["figures_root"] = str(PROJECT_ROOT / "outputs/stage4/figures")
    cfg["paths"]["context_root"] = str(PROJECT_ROOT / "outputs/stage4/context")
    cfg["paths"]["run_manifest_root"] = str(PROJECT_ROOT / "outputs/stage4/run_manifests")
    cfg["paths"]["reports_root"] = str(PROJECT_ROOT / "reports")
    cfg["paths"]["tables_root"] = str(PROJECT_ROOT / "reports/tables")

    cfg["stage2_dependency"]["project_root"] = str(STAGE2_PROJECT_ROOT)
    cfg["stage2_dependency"]["src_path"] = str(STAGE2_PROJECT_ROOT / "src")
    cfg["stage2_dependency"]["baseline_output_root"] = str(STAGE2_PROJECT_ROOT / "outputs/stage2")

    cfg["data"]["source_file"] = SOURCE_FILE
    cfg["stage4_model"]["primary_image_window"] = IMAGE_WINDOW
    cfg["stage4_model"]["primary_image_spec"] = IMAGE_SPEC
    cfg["stage4_model"]["primary_return_horizon"] = RETURN_HORIZON

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Config patched:", config_path, flush=True)


def context_name():
    return f"{OUTPUT_PREFIX}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}_{FEATURE_SET_NAME}"


def zip_outputs():
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "outputs/stage4/context" / context_name() / f"seed_{RUN_SEED}",
        PROJECT_ROOT / "data_inventory/roro_public",
    ]
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    print("Bundle:", BUNDLE_PATH, round(BUNDLE_PATH.stat().st_size / 1024 / 1024, 3), "MB")


# ============================================================
# 1. Copy code snapshots and patch config
# ============================================================
CODE_INPUT = first_existing(CODE_INPUT_CANDIDATES)
STAGE2_CODE_INPUT = first_existing(STAGE2_CODE_INPUT_CANDIDATES)
copy_or_extract_input(CODE_INPUT, PROJECT_ROOT, expected_child="stage4_film_conditioning")
copy_or_extract_input(STAGE2_CODE_INPUT, STAGE2_PROJECT_ROOT, expected_child="stage2_btc_extension")
assert_latest_code()
patch_kaggle_config()


# ============================================================
# 2. Build public RORO context feature artifact
# ============================================================
cmd = [
    sys.executable, "-u",
    "scripts/build_stage4_roro_context_features.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--run-seed", str(RUN_SEED),
    "--asof-lag-days", str(ASOF_LAG_DAYS),
    "--roro-data-dir", "data_inventory/roro_public/raw",
    "--output-prefix", OUTPUT_PREFIX,
    "--feature-set-name", FEATURE_SET_NAME,
    "--write-report-copy",
]
if INCLUDE_COMPONENT_FEATURES:
    cmd.append("--include-component-features")
if REFRESH_DOWNLOADS:
    cmd.append("--refresh-downloads")

run(cmd)


# ============================================================
# 3. Display compact audit and zip results
# ============================================================
report_prefix = f"{context_name()}_seed{RUN_SEED}"
audit_path = PROJECT_ROOT / "reports/tables" / f"{report_prefix}_roro_context_feature_audit.json"
summary_path = PROJECT_ROOT / "reports/tables" / f"{report_prefix}_roro_context_feature_summary.csv"

display(Markdown("# Stage 4 N13-3 Public RORO Context Audit"))
display(json.loads(audit_path.read_text(encoding="utf-8")))

display(Markdown("# Stage 4 N13-3 Public RORO Feature Summary"))
try:
    summary = pd.read_csv(summary_path)
    display(summary)
except Exception as exc:
    print("summary display failed:", exc)

zip_outputs()
print("\nDONE", flush=True)
print("Context:", PROJECT_ROOT / "outputs/stage4/context" / context_name() / f"seed_{RUN_SEED}")
print("Bundle:", BUNDLE_PATH)
```
