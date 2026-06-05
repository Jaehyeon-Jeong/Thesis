# Kaggle Stage 4 N10 Stage2-vs-N10 Correction Analysis One Cell

Copy this cell into Kaggle **after the N10 selected news interpretability run**.

This cell does not train a new model. It:

- ensures Stage 2 selected-baseline test predictions exist;
- ensures N10 selected news-FiLM test predictions exist;
- compares predictions sample-by-sample;
- exports `Stage2 wrong -> N10 correct` and `Stage2 correct -> N10 wrong`
  tables for targeted Grad-CAM/gamma-beta/news interpretation.

```python
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
from IPython.display import display, Markdown

# ============================================================
# User settings
# ============================================================
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
STAGE2_PROJECT_ROOT = Path("/kaggle/working/stage2_btc_extension")

# Use this only if Stage 2 checkpoints are not already under
# /kaggle/working/stage2_btc_extension/outputs/stage2/checkpoints.
# Leave empty to auto-detect under /kaggle/working and /kaggle/input.
STAGE2_CHECKPOINT_BUNDLE = ""

IMAGE_WINDOW = 60
IMAGE_SPEC = "ohlc_ma_vb"
RETURN_HORIZON = 20
CONTEXT_WINDOW = 60
CONTEXT_METHOD = "film_full_bounded_last_block"
STAGE4_EXPERIMENT_SUFFIX = "news_tfidf_svd32_w7_20_60_pretrained_frozen_s0p02"
RUN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SPLIT = "test"
TOP_K = 30

OUTPUT_PREFIX = "stage4_n10_stage2_vs_n10_correction_analysis"
BUNDLE_PATH = Path(f"/kaggle/working/{OUTPUT_PREFIX}_bundle.zip")


def run(cmd, cwd=PROJECT_ROOT, capture=False, check=True):
    """Run one command."""

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


def stage2_experiment_name():
    return f"stage2_i{IMAGE_WINDOW}_{IMAGE_SPEC}_r{RETURN_HORIZON}"


def stage4_experiment_name():
    return (
        f"stage4_{CONTEXT_METHOD}_i{IMAGE_WINDOW}_{IMAGE_SPEC}_"
        f"r{RETURN_HORIZON}_c{CONTEXT_WINDOW}_{STAGE4_EXPERIMENT_SUFFIX}"
    )


def assert_required_files():
    required_stage4 = [
        "scripts/analyze_stage4_stage2_context_corrections.py",
        "scripts/evaluate_stage4_predictions.py",
    ]
    missing_stage4 = [path for path in required_stage4 if not (PROJECT_ROOT / path).exists()]
    if missing_stage4:
        raise RuntimeError("Stage 4 snapshot is missing: " + ", ".join(missing_stage4))

    required_stage2 = [
        "scripts/evaluate_stage2_predictions.py",
        "src/stage2_btc/models/stock_cnn.py",
    ]
    missing_stage2 = [path for path in required_stage2 if not (STAGE2_PROJECT_ROOT / path).exists()]
    if missing_stage2:
        raise RuntimeError("Stage 2 snapshot is missing: " + ", ".join(missing_stage2))


def resolve_checkpoint_candidate(candidate: Path) -> Path:
    """Return a directory containing Stage 2 outputs/checkpoints."""

    exp = stage2_experiment_name()
    candidate = candidate.expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        extract_root = Path("/kaggle/working/stage2_checkpoint_bundle_extracted") / candidate.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(extract_root)
        return resolve_checkpoint_candidate(extract_root)

    if not candidate.is_dir():
        raise FileNotFoundError(candidate)

    direct = candidate / "outputs/stage2/checkpoints" / exp / "seed_42/best.pt"
    if direct.exists():
        return candidate

    nested = next(candidate.rglob(f"outputs/stage2/checkpoints/{exp}/seed_42/best.pt"), None)
    if nested is not None:
        return nested.parents[5]

    checkpoint_root = candidate / "checkpoints" / exp / "seed_42/best.pt"
    if checkpoint_root.exists():
        return candidate

    raise FileNotFoundError(f"No Stage 2 checkpoint found under {candidate}")


def find_stage2_checkpoint_bundle() -> Path:
    if STAGE2_CHECKPOINT_BUNDLE:
        return resolve_checkpoint_candidate(Path(STAGE2_CHECKPOINT_BUNDLE))

    exp = stage2_experiment_name()
    existing = STAGE2_PROJECT_ROOT / "outputs/stage2/checkpoints" / exp / "seed_42/best.pt"
    if existing.exists():
        return STAGE2_PROJECT_ROOT

    for root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if candidate == PROJECT_ROOT or PROJECT_ROOT in candidate.parents:
                continue
            if candidate.is_file() and candidate.suffix.lower() != ".zip":
                continue
            try:
                resolved = resolve_checkpoint_candidate(candidate)
            except Exception:
                continue
            print("Stage 2 checkpoint bundle:", resolved, flush=True)
            return resolved
    raise FileNotFoundError("Could not find Stage 2 checkpoint bundle.")


def ensure_stage2_checkpoints():
    """Copy selected Stage 2 checkpoints into the Stage 2 project if needed."""

    exp = stage2_experiment_name()
    missing = [
        seed
        for seed in RUN_SEEDS
        if not (
            STAGE2_PROJECT_ROOT
            / "outputs/stage2/checkpoints"
            / exp
            / f"seed_{seed}/best.pt"
        ).exists()
    ]
    if not missing:
        print("Stage 2 checkpoints already available.", flush=True)
        return

    bundle_root = find_stage2_checkpoint_bundle()
    for seed in missing:
        src = next(bundle_root.rglob(f"checkpoints/{exp}/seed_{seed}/best.pt"), None)
        if src is None:
            raise FileNotFoundError(f"Missing Stage 2 best.pt for seed {seed} under {bundle_root}")
        src_dir = src.parent
        dst_dir = STAGE2_PROJECT_ROOT / "outputs/stage2/checkpoints" / exp / f"seed_{seed}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in src_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, dst_dir / file.name)
        print(f"Copied Stage 2 checkpoint seed {seed}: {dst_dir}", flush=True)


def ensure_stage2_predictions():
    """Generate Stage 2 selected-baseline predictions if missing."""

    exp = stage2_experiment_name()
    for seed in RUN_SEEDS:
        prediction = (
            STAGE2_PROJECT_ROOT
            / "outputs/stage2/predictions"
            / exp
            / f"seed_{seed}"
            / f"{EVAL_SPLIT}_predictions.csv"
        )
        if prediction.exists() and prediction.stat().st_size > 0:
            print(f"Stage 2 prediction exists seed {seed}: {prediction}", flush=True)
            continue
        run([
            sys.executable, "-u",
            "scripts/evaluate_stage2_predictions.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--run-seed", str(seed),
            "--split", EVAL_SPLIT,
        ], cwd=STAGE2_PROJECT_ROOT)


def ensure_stage4_predictions():
    """Generate N10 selected-model predictions if checkpoint exists but CSV is missing."""

    exp = stage4_experiment_name()
    for seed in RUN_SEEDS:
        prediction = (
            PROJECT_ROOT
            / "outputs/stage4/predictions"
            / exp
            / f"seed_{seed}"
            / f"{EVAL_SPLIT}_predictions.csv"
        )
        if prediction.exists() and prediction.stat().st_size > 0:
            print(f"N10 prediction exists seed {seed}: {prediction}", flush=True)
            continue

        checkpoint = (
            PROJECT_ROOT
            / "outputs/stage4/checkpoints"
            / exp
            / f"seed_{seed}/best.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"N10 checkpoint missing for seed {seed}. Run the N10 selected "
                f"interpretability cell first. Expected: {checkpoint}"
            )
        run([
            sys.executable, "-u",
            "scripts/evaluate_stage4_predictions.py",
            "--config", "configs/env_kaggle.yaml",
            "--image-window", str(IMAGE_WINDOW),
            "--image-spec", IMAGE_SPEC,
            "--return-horizon", str(RETURN_HORIZON),
            "--context-method", CONTEXT_METHOD,
            "--run-seed", str(seed),
            "--split", EVAL_SPLIT,
        ])


def zip_analysis_outputs():
    """Create a compact downloadable bundle of the correction-analysis outputs."""

    tables_root = PROJECT_ROOT / "reports/tables"
    targets = sorted(tables_root.glob(f"{OUTPUT_PREFIX}*"))
    if not targets:
        raise FileNotFoundError(f"No analysis outputs found for prefix {OUTPUT_PREFIX}")
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest = {"files": []}
        for path in targets:
            arcname = Path("reports/tables") / path.name
            archive.write(path, arcname)
            manifest["files"].append(str(arcname))
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print(f"Analysis bundle: {BUNDLE_PATH}", flush=True)


assert_required_files()
ensure_stage2_checkpoints()
ensure_stage2_predictions()
ensure_stage4_predictions()

run([
    sys.executable, "-u",
    "scripts/analyze_stage4_stage2_context_corrections.py",
    "--config", "configs/env_kaggle.yaml",
    "--image-window", str(IMAGE_WINDOW),
    "--image-spec", IMAGE_SPEC,
    "--return-horizon", str(RETURN_HORIZON),
    "--context-window", str(CONTEXT_WINDOW),
    "--context-method", CONTEXT_METHOD,
    "--stage4-experiment-suffix", STAGE4_EXPERIMENT_SUFFIX,
    "--stage2-output-root", str(STAGE2_PROJECT_ROOT / "outputs/stage2"),
    "--stage4-output-root", str(PROJECT_ROOT / "outputs/stage4"),
    "--run-seeds", *map(str, RUN_SEEDS),
    "--split", EVAL_SPLIT,
    "--top-k", str(TOP_K),
    "--output-prefix", OUTPUT_PREFIX,
])

tables_root = PROJECT_ROOT / "reports/tables"
seed_summary = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_seed_summary.csv")
transition_summary = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_transition_summary.csv")
top_corrections = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_top_corrections.csv")
top_regressions = pd.read_csv(tables_root / f"{OUTPUT_PREFIX}_top_regressions.csv")

display(Markdown("# Stage 2 vs N10 Seed Summary"))
display(seed_summary)

display(Markdown("# Transition Summary"))
display(transition_summary)

display(Markdown("# Top Stage2 Wrong -> N10 Correct"))
display(top_corrections.head(20))

display(Markdown("# Top Stage2 Correct -> N10 Wrong"))
display(top_regressions.head(20))

report_path = tables_root / f"{OUTPUT_PREFIX}_report.md"
display(Markdown(report_path.read_text(encoding="utf-8")))

zip_analysis_outputs()
print("DONE", flush=True)
print("Download:", BUNDLE_PATH, flush=True)
```
