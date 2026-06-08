# Kaggle Stage 4 N16-1 Derivatives Context Features One Cell

Copy the Python cell below into Kaggle after attaching:

- Stage 4 code snapshot containing `data_inventory/crypto_derivatives/`
- Stage 2 BTC code snapshot
- BTC OHLCV Kaggle dataset

This cell only builds N16 derivatives/leverage prebuilt context artifacts. It
does not train a model.

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
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n16_snapshot.zip"),
]
STAGE2_CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage22/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage2/stage2_btc_extension"),
    Path("/kaggle/input/datasets/moskow/stage4/stage2_btc_extension"),
]
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")
DATA_ROOT = Path("/kaggle/input")

# Leave empty to auto-detect BTC OHLCV under /kaggle/input.
SOURCE_FILE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
RUN_SEEDS = [42, 43, 44, 45, 46]
DAILY_ASOF_LAG_DAYS = 1

FEATURE_SETS = [
    "derivatives_all",
    "funding_only",
    "funding_plus_cftc_oi",
    "funding_plus_activity",
    "funding_plus_activity_plus_cftc_oi",
]

BUNDLE_PATH = Path("/kaggle/working/stage4_n16_1_derivatives_context_features_bundle.zip")


def first_existing(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
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
        "scripts/build_stage4_derivatives_context_features.py": "N16 derivatives leverage context feature builder",
        "data_inventory/crypto_derivatives/bitmex_xbtusd/processed/bitmex_xbtusd_funding_daily_2018_2024.csv": "",
        "data_inventory/crypto_derivatives/bitmex_xbtusd/processed/bitmex_xbtusd_derivatives_activity_daily_2018_2024.csv": "",
        "data_inventory/crypto_derivatives/cftc_cme_bitcoin_cot/processed/cftc_cme_bitcoin_main_plus_micro_cot_daily_release_lag3_ffill_2018_2024.csv": "",
    }
    problems = []
    for rel_path, marker in required.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            problems.append(f"{rel_path} missing")
            continue
        if marker and marker not in path.read_text(encoding="utf-8"):
            problems.append(f"{rel_path} missing marker {marker!r}")
    if problems:
        raise RuntimeError("Stage 4 code snapshot is stale/incomplete: " + "; ".join(problems))


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


def zip_outputs():
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    include_roots = [
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "outputs/stage4/context",
        PROJECT_ROOT / "data_inventory/crypto_derivatives",
    ]
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file() and ("n16" in str(path) or "crypto_derivatives" in str(path)):
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
# 2. Build N16 context artifacts
# ============================================================
for run_seed in RUN_SEEDS:
    for feature_set in FEATURE_SETS:
        cmd = [
            sys.executable, "-u",
            "scripts/build_stage4_derivatives_context_features.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(run_seed),
            "--feature-set", feature_set,
            "--feature-set-name", f"n16_{feature_set}",
            "--daily-asof-lag-days", str(DAILY_ASOF_LAG_DAYS),
        ]
        if int(run_seed) == 42:
            cmd.append("--write-report-copy")
        run(cmd)

zip_outputs()

tables = sorted((PROJECT_ROOT / "reports/tables").glob("stage4_derivatives_context_i60_ohlc_ma_vb_r20_n16_*manifest.json"))
rows = []
for path in tables:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "file": path.name,
        "feature_set": payload.get("feature_set"),
        "context_dim": payload.get("context_dim"),
        "split_counts": payload.get("split_counts"),
        "feature_group_counts": payload.get("feature_group_counts"),
    })

display(Markdown("# N16-1 Derivatives Context Feature Artifacts"))
display(pd.DataFrame(rows))

print("\nDONE", flush=True)
print("Bundle:", BUNDLE_PATH, flush=True)
```
