# Kaggle Stage 5 5-9A FinBERT Sentiment One Cell

Copy the Python cell below into Kaggle after attaching:

- latest `stage5_llm_news_embedding` folder/zip, or a result bundle containing
  `stage5_embedding_input_items.csv`
- internet-enabled Kaggle session, or a Kaggle dataset containing a local
  FinBERT Hugging Face model directory

This cell runs only `5-9A`: headline-level FinBERT sentiment extraction. It does
not train Stage4/Stage5 FiLM models.

```python
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile

# ============================================================
# User settings
# ============================================================
STAGE5_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding"),
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding_stage5_snapshot/stage5_llm_news_embedding"),
    Path("/kaggle/input/datasets/moskow/stage5/stage5_llm_news_embedding_stage5_snapshot.zip"),
    Path("/kaggle/input/stage5-llm-news-embedding/stage5_llm_news_embedding"),
    Path("/kaggle/input/stage5-llm-news-embedding-stage5-snapshot/stage5_llm_news_embedding"),
]

# If you attach a local Hugging Face model dataset, set this to the model folder.
# Otherwise keep "ProsusAI/finbert" and Kaggle will download it from HF.
FINBERT_MODEL = "ProsusAI/finbert"

PROJECT_ROOT = Path("/kaggle/working/stage5_llm_news_embedding")
RUN_ID = "finbert_prosusai_headline_v1"
BATCH_SIZE = 128
MAX_LENGTH = 128
LIMIT = 0  # set to 256 for smoke test, 0 for full run

BUNDLE_PATH = Path(f"/kaggle/working/stage5_9a_{RUN_ID}_result_bundle.zip")


def looks_like_stage5_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "scripts/build_stage5_finbert_sentiment.py").exists()
        and (path / "checklist.md").exists()
    )


def find_stage5_input() -> Path:
    for candidate in STAGE5_INPUT_CANDIDATES:
        candidate = Path(candidate)
        if not candidate.exists():
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return candidate
        if looks_like_stage5_root(candidate):
            return candidate
        nested = candidate / "stage5_llm_news_embedding"
        if looks_like_stage5_root(nested):
            return nested

    for script_path in Path("/kaggle/input").rglob("build_stage5_finbert_sentiment.py"):
        root = script_path.parents[1]
        if looks_like_stage5_root(root):
            return root

    inspected = [str(path) for path in list(Path("/kaggle/input").rglob("*stage5*"))[:60]]
    raise FileNotFoundError(
        "Could not find Stage 5 code input. First stage5-like paths: "
        + json.dumps(inspected, indent=2)
    )


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


def ensure_python_packages():
    missing = []
    for package in ["torch", "transformers"]:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    if missing:
        print("[setup] installing missing packages:", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def find_input_items() -> Path:
    candidates = [
        PROJECT_ROOT / "outputs/stage5/embedding_inputs/stage5_embedding_input_items.csv",
        PROJECT_ROOT / "reports/tables/stage5_embedding_input_items.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in Path("/kaggle/input").rglob("stage5_embedding_input_items.csv"):
        return path

    for path in Path("/kaggle/working").rglob("stage5_embedding_input_items.csv"):
        return path

    raise FileNotFoundError(
        "Could not find stage5_embedding_input_items.csv. Attach the Stage5 "
        "input/context bundle or a Stage5 folder containing outputs/stage5/embedding_inputs."
    )


def run(cmd, cwd=PROJECT_ROOT):
    cmd = [str(item) for item in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), text=True, check=True)


def zip_outputs():
    include_roots = [
        PROJECT_ROOT / "outputs/stage5/finbert_sentiment",
        PROJECT_ROOT / "reports/tables",
        PROJECT_ROOT / "data_inventory",
    ]
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        zf.write(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    print("Bundle:", BUNDLE_PATH, round(BUNDLE_PATH.stat().st_size / 1024 / 1024, 3), "MB")


stage5_input = find_stage5_input()
print("Stage5 input:", stage5_input)
copy_or_extract_input(stage5_input, PROJECT_ROOT, expected_child="stage5_llm_news_embedding")
ensure_python_packages()

input_items = find_input_items()
print("Input items:", input_items)

cmd = [
    sys.executable, "-u",
    "scripts/build_stage5_finbert_sentiment.py",
    "--run-id", RUN_ID,
    "--model-name", FINBERT_MODEL,
    "--input-items", str(input_items),
    "--batch-size", str(BATCH_SIZE),
    "--max-length", str(MAX_LENGTH),
]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]

run(cmd)
zip_outputs()

print("\nDONE")
print("Report:", PROJECT_ROOT / "reports/tables" / f"{RUN_ID}_report.md")
print("Summary:", PROJECT_ROOT / "reports/tables" / f"{RUN_ID}_summary.csv")
print("Bundle:", BUNDLE_PATH)
```
