# Kaggle One-Cell: 4-N14-B1 Conditional Merge Table

Use this cell after the Stage 2 and Stage 4 prediction/context artifacts are
available in `/kaggle/working`, or after attaching result bundles that already
contain them. If Kaggle reset `/kaggle/working`, attach the Stage 2 N15-A
four-image output bundle as a Kaggle dataset; this cell will auto-detect or
extract it.

```python
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# ============================================================
# User settings
# ============================================================
CODE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n14b_snapshot.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_latest.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n16_snapshot.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n15c_snapshot.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning_n13_6_snapshot.zip"),
]

# If Kaggle was reset, attach a bundle containing outputs/stage4 and/or
# stage2 outputs, then add its path here. Leave empty if outputs already exist
# in /kaggle/working from the previous N16 run.
RESULT_BUNDLE_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage4/stage4_n16_5_derivatives_interpretability_local_bundle"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_n16_5_derivatives_interpretability_local_bundle.zip"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle"),
    Path("/kaggle/input/datasets/moskow/stage4/stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle.zip"),
    Path(
        "/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/"
        "stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle.zip"
    ),
]

# Stage 2 N15-A output bundle. This must contain outputs/stage2/predictions.
# It is not just model code and not only checkpoints: N14-B1 needs the baseline
# Stage 2 test_predictions.csv files for the same image specification.
STAGE2_RESULT_BUNDLE_CANDIDATES = [
    Path(
        "/kaggle/input/datasets/moskow/stage22/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"
    ),
    Path(
        "/kaggle/input/datasets/moskow/stage22/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"
    ),
    Path(
        "/kaggle/input/datasets/moskow/stage22/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"
    ),
    Path(
        "/kaggle/input/stage2-i60-r20-four-image-full-seed-checkpoints-for-stage4-n15/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"
    ),
    Path(
        "/kaggle/input/stage2-i60-r20-four-image-full-seed-checkpoints-for-stage4-n15/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"
    ),
    Path(
        "/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"
    ),
    Path(
        "/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning/"
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15.zip"
    ),
]

PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_BUNDLE_UNZIP_ROOT = Path("/kaggle/working/n14b1_stage2_result_bundle_unzipped")
STAGE2_OUTPUT_ROOT_CANDIDATES = [
    Path("/kaggle/working/stage2_btc_extension/outputs/stage2"),
    Path("/kaggle/working/stage4_film_conditioning/stage2_btc_extension/outputs/stage2"),
    Path("/kaggle/working/stage4_film_conditioning/outputs/stage2"),
]
STAGE4_OUTPUT_ROOT = PROJECT_ROOT / "outputs/stage4"

RUN_SEEDS = [42, 43, 44, 45, 46]
SPLIT = "test"

# Default N14-B1 candidate: N16 positive same-image case.
STAGE2_EXPERIMENT = "stage2_i60_ohlc_vb_r20"
STAGE4_EXPERIMENT = (
    "stage4_film_full_bounded_last_block_i60_ohlc_vb_r20_c60_"
    "n16d_funding_plus_cftc_oi_pretrained_frozen_s0p02"
)
CONTEXT_NAME = "stage4_derivatives_context_i60_ohlc_vb_r20_n16d_funding_plus_cftc_oi"
ANALYSIS_NAME = "n16_ohlc_vb_funding_plus_cftc_oi"
OUTPUT_PREFIX = "stage4_n14b1_n16_derivatives_conditional_merge"

REQUIRED_STAGE4_FILES = [
    "scripts/build_stage4_n14b_conditional_merge_table.py",
]


def first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_stage4_code_candidates():
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    found = []
    for path in input_root.rglob("*"):
        if path.name == "stage4_film_conditioning":
            found.append(path)
        elif path.is_file() and path.suffix.lower() == ".zip" and path.name.startswith("stage4_film_conditioning"):
            found.append(path)
    return found


def stage4_code_candidate_has_required_files(candidate: Path):
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(candidate) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile:
            return False
        return all(any(name.endswith(rel) for name in names) for rel in REQUIRED_STAGE4_FILES)

    if candidate.is_dir():
        if all((candidate / rel).exists() for rel in REQUIRED_STAGE4_FILES):
            return True
        nested = candidate / "stage4_film_conditioning"
        if all((nested / rel).exists() for rel in REQUIRED_STAGE4_FILES):
            return True
    return False


def resolve_stage4_code_input():
    candidates = [path for path in CODE_INPUT_CANDIDATES if path.exists()]
    for discovered in discover_stage4_code_candidates():
        if discovered not in candidates:
            candidates.append(discovered)

    for candidate in candidates:
        if not stage4_code_candidate_has_required_files(candidate):
            print("Skip stale Stage 4 code candidate:", candidate)
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            print("Stage 4 code zip found:", candidate)
            return candidate
        if candidate.is_dir():
            if (candidate / "scripts").exists() and (candidate / "src").exists():
                print("Stage 4 code folder found:", candidate)
                return candidate
            nested = candidate / "stage4_film_conditioning"
            if (nested / "scripts").exists() and (nested / "src").exists():
                print("Stage 4 nested code folder found:", candidate)
                return candidate

    return None


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
    raise FileNotFoundError(src)


def assert_stage4_code_ready():
    missing = [rel for rel in REQUIRED_STAGE4_FILES if not (PROJECT_ROOT / rel).exists()]
    if missing:
        raise RuntimeError(
            "Stage 4 code snapshot is stale/incomplete for N14-B1. "
            "Upload the latest stage4_film_conditioning snapshot. Missing: "
            + ", ".join(missing)
        )


def overlay_bundle(src: Path, dst: Path):
    if not src.exists():
        return False
    tmp = None
    source_root = src
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = Path("/kaggle/working/n14b1_result_bundle_unzipped")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as archive:
            archive.extractall(tmp)
        source_root = tmp
    candidates = [source_root]
    nested = list(source_root.rglob("stage4_film_conditioning")) if source_root.exists() else []
    candidates.extend(nested)
    for candidate in candidates:
        if (candidate / "outputs").exists() or (candidate / "reports").exists():
            shutil.copytree(candidate, dst, dirs_exist_ok=True)
            if tmp and tmp.exists():
                shutil.rmtree(tmp)
            return True
    if tmp and tmp.exists():
        shutil.rmtree(tmp)
    return False


def discover_stage4_result_bundle_candidates():
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    found = []
    markers = [
        "stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle",
        "stage4_n16_5_derivatives_interpretability_local_bundle",
    ]
    for path in input_root.rglob("*"):
        if any(marker in path.name for marker in markers):
            found.append(path)
    return found


def resolve_stage4_result_bundle():
    candidates = [path for path in RESULT_BUNDLE_CANDIDATES if path.exists()]
    for discovered in discover_stage4_result_bundle_candidates():
        if discovered not in candidates:
            candidates.append(discovered)
    for candidate in candidates:
        tmp = None
        source_root = candidate
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            tmp = Path("/kaggle/working/n14b1_result_probe_unzipped")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(tmp)
            source_root = tmp
        pred = next(
            source_root.rglob(
                "outputs/stage4/predictions/"
                f"{STAGE4_EXPERIMENT}/seed_42/{SPLIT}_predictions.csv"
            ),
            None,
        )
        ctx = next(
            source_root.rglob(
                "outputs/stage4/context/"
                f"{CONTEXT_NAME}/seed_42/context_features.csv"
            ),
            None,
        )
        if tmp and tmp.exists():
            shutil.rmtree(tmp)
        if pred is not None and ctx is not None:
            print("Stage 4 result bundle found:", candidate)
            return candidate
        print("Skip Stage 4 result bundle without N14-B1 artifacts:", candidate)
    return None


def is_stage2_output_root(path: Path):
    if not path.exists():
        return False
    for seed in RUN_SEEDS:
        pred = (
            path
            / "predictions"
            / STAGE2_EXPERIMENT
            / f"seed_{seed}"
            / f"{SPLIT}_predictions.csv"
        )
        if not pred.exists():
            return False
    return True


def find_stage2_output_root_inside(root: Path):
    if not root.exists():
        return None

    direct_candidates = [
        root,
        root / "outputs/stage2",
        root / "stage2",
        root
        / "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"
        / "outputs/stage2",
    ]
    for candidate in direct_candidates:
        if is_stage2_output_root(candidate):
            return candidate

    first_seed = f"seed_{RUN_SEEDS[0]}"
    for pred in root.rglob(f"{SPLIT}_predictions.csv"):
        if (
            pred.parent.name == first_seed
            and pred.parent.parent.name == STAGE2_EXPERIMENT
            and pred.parent.parent.parent.name == "predictions"
        ):
            candidate = pred.parent.parent.parent.parent
            if is_stage2_output_root(candidate):
                return candidate
    return None


def discover_stage2_result_bundle_candidates():
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    marker = "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15"
    found = []
    for path in input_root.rglob("*"):
        if marker in path.name:
            found.append(path)
    return found


def resolve_stage2_output_root():
    for path in STAGE2_OUTPUT_ROOT_CANDIDATES:
        output_root = find_stage2_output_root_inside(path)
        if output_root is not None:
            print("Stage 2 output root found in /kaggle/working:", output_root)
            return output_root

    candidates = [path for path in STAGE2_RESULT_BUNDLE_CANDIDATES if path.exists()]
    for discovered in discover_stage2_result_bundle_candidates():
        if discovered not in candidates:
            candidates.append(discovered)

    for candidate in candidates:
        source_root = candidate
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            if STAGE2_BUNDLE_UNZIP_ROOT.exists():
                shutil.rmtree(STAGE2_BUNDLE_UNZIP_ROOT)
            STAGE2_BUNDLE_UNZIP_ROOT.mkdir(parents=True, exist_ok=True)
            print("Extracting Stage 2 result bundle:", candidate)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(STAGE2_BUNDLE_UNZIP_ROOT)
            source_root = STAGE2_BUNDLE_UNZIP_ROOT

        output_root = find_stage2_output_root_inside(source_root)
        if output_root is not None:
            print("Stage 2 output root found:", output_root)
            return output_root

    return None


def run(cmd, cwd=PROJECT_ROOT):
    cmd = [str(item) for item in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=True)


code_input = resolve_stage4_code_input()
if code_input is None:
    raise FileNotFoundError(
        "Could not find Stage 4 code input. Attach the stage4 dataset that contains "
        "stage4_film_conditioning/ or stage4_film_conditioning_n16_snapshot.zip."
    )
copy_or_extract_input(code_input, PROJECT_ROOT, expected_child="stage4_film_conditioning")
print("Stage 4 code ready:", PROJECT_ROOT)
assert_stage4_code_ready()

bundle_input = resolve_stage4_result_bundle()
if bundle_input is not None:
    ok = overlay_bundle(bundle_input, PROJECT_ROOT)
    print("Result bundle overlay:", ok, bundle_input)
else:
    print(
        "No Stage 4 N16-4 result bundle attached; using existing /kaggle/working outputs. "
        "If Kaggle was reset, attach stage4_n16_4_ohlc_vb_derivatives_repeat_result_bundle.zip "
        "or rerun N16-4 first."
    )

stage2_output_root = resolve_stage2_output_root()
if stage2_output_root is None:
    raise FileNotFoundError(
        "Could not find Stage 2 predictions for "
        f"{STAGE2_EXPERIMENT}. Attach the Stage 2 N15-A output bundle "
        "stage2_i60_r20_four_image_full_seed_checkpoints_for_stage4_n15 "
        "or rebuild the Stage 2 predictions."
    )

stage4_pred = (
    STAGE4_OUTPUT_ROOT / "predictions" / STAGE4_EXPERIMENT / "seed_42" / f"{SPLIT}_predictions.csv"
)
stage4_ctx = (
    STAGE4_OUTPUT_ROOT / "context" / CONTEXT_NAME / "seed_42" / "context_features.csv"
)
if not stage4_pred.exists() or not stage4_ctx.exists():
    raise FileNotFoundError(
        "Could not find Stage 4 prediction/context artifacts. "
        f"Prediction exists={stage4_pred.exists()} path={stage4_pred}; "
        f"context exists={stage4_ctx.exists()} path={stage4_ctx}"
    )

run([
    sys.executable, "-u",
    "scripts/build_stage4_n14b_conditional_merge_table.py",
    "--config", "configs/env_kaggle.yaml",
    "--stage2-experiment", STAGE2_EXPERIMENT,
    "--stage4-experiment", STAGE4_EXPERIMENT,
    "--context-name", CONTEXT_NAME,
    "--analysis-name", ANALYSIS_NAME,
    "--stage2-output-root", stage2_output_root,
    "--stage4-output-root", STAGE4_OUTPUT_ROOT,
    "--run-seeds", *map(str, RUN_SEEDS),
    "--split", SPLIT,
    "--output-prefix", OUTPUT_PREFIX,
])

print("\nDONE")
print("Main output:", PROJECT_ROOT / "reports/tables" / f"{OUTPUT_PREFIX}_merged_decisions.csv")
print("Report:", PROJECT_ROOT / "reports/tables" / f"{OUTPUT_PREFIX}_report.md")
```
