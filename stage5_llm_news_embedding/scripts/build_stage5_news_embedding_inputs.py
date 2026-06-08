#!/usr/bin/env python3
"""Build deterministic Stage 5 news-item embedding inputs.

This script does not call an embedding API. It creates the stable news-item
table that later embedding jobs will hash, cache, and embed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT.parent
STAGE4_TABLES = THESIS_ROOT / "stage4_film_conditioning" / "reports" / "tables"
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"
OUTPUT_ROOT = ROOT / "outputs" / "stage5" / "embedding_inputs"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_news_source() -> Path:
    source_audit = read_json(STAGE4_TABLES / "stage4_news_source_audit.json")
    candidates: list[Path] = []
    for key in ["source_file", "resolved_source_file", "file"]:
        value = source_audit.get("dataset", {}).get(key)
        if value:
            candidates.append(Path(value))
    for key in ["source_file", "resolved_source_file", "file"]:
        value = source_audit.get(key)
        if value:
            candidates.append(Path(value))

    for path in candidates:
        if path.exists():
            return path

    fallback = list(THESIS_ROOT.rglob("BTC_match_title.csv"))
    if fallback:
        return fallback[0]

    tried = ", ".join(str(path) for path in candidates) or "no candidates"
    raise FileNotFoundError(f"Could not resolve Bitcoin news source. Tried: {tried}")


def first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def standardize_news(raw: pd.DataFrame) -> pd.DataFrame:
    news_time_col = first_existing_column(raw, ["news_time", "date_time", "datetime"])
    unix_col = first_existing_column(raw, ["time_unix", "timestamp"])
    title_col = first_existing_column(raw, ["title_text", "title", "headline"])
    url_col = first_existing_column(raw, ["url_text", "url", "link"])
    source_col = first_existing_column(raw, ["source_text", "source", "publisher"])
    article_col = first_existing_column(raw, ["article_text_string", "article_text", "text"])

    if title_col is None:
        raise ValueError("News source does not contain a title/headline column.")

    frame = pd.DataFrame()
    if news_time_col is not None:
        frame["news_time"] = pd.to_datetime(
            raw[news_time_col], errors="coerce", utc=True
        ).dt.tz_convert(None)
    elif unix_col is not None:
        frame["news_time"] = pd.to_datetime(
            raw[unix_col], errors="coerce", unit="s", utc=True
        ).dt.tz_convert(None)
    else:
        raise ValueError("News source does not contain a parseable time column.")

    if "news_date" in raw.columns:
        frame["news_date"] = pd.to_datetime(raw["news_date"], errors="coerce").dt.normalize()
    else:
        frame["news_date"] = frame["news_time"].dt.normalize()

    title_text = raw[title_col].fillna("").astype("string")
    frame["title_text"] = title_text
    frame["headline_text"] = title_text.map(clean_text)
    frame["normalized_title"] = title_text.map(normalize_title)

    if url_col is not None:
        frame["url_text"] = raw[url_col].map(clean_text)
    else:
        frame["url_text"] = ""

    if source_col is not None:
        frame["source_text"] = raw[source_col].map(clean_text)
    else:
        frame["source_text"] = ""

    if article_col is not None:
        article_text = raw[article_col].map(clean_text)
        frame["article_length_chars"] = article_text.str.len().fillna(0).astype(int)
        frame["has_article_text"] = article_text.str.len().fillna(0).gt(0)
    else:
        frame["article_length_chars"] = 0
        frame["has_article_text"] = False

    frame["title_length_chars"] = frame["headline_text"].str.len().fillna(0).astype(int)
    frame["has_title"] = frame["headline_text"].str.len().fillna(0).gt(0)
    return frame


def dedupe_news(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    initial_rows = len(frame)
    filtered = frame.copy()
    filtered["headline_text"] = filtered["title_text"].map(clean_text)
    filtered = filtered.loc[filtered["headline_text"].str.strip().ne("")].copy()
    after_title_filter = len(filtered)

    filtered = filtered.sort_values(
        ["news_time", "url_text", "normalized_title"]
    ).reset_index(drop=True)

    non_empty_url = filtered["url_text"].ne("")
    duplicate_url = non_empty_url & filtered["url_text"].duplicated(keep="first")
    after_url = filtered.loc[~duplicate_url].copy()

    non_empty_title = after_url["normalized_title"].ne("")
    duplicate_title = non_empty_title & after_url["normalized_title"].duplicated(keep="first")
    deduped = after_url.loc[~duplicate_title].copy().reset_index(drop=True)

    audit = {
        "initial_rows": int(initial_rows),
        "after_title_filter_rows": int(after_title_filter),
        "dropped_empty_title_rows": int(initial_rows - after_title_filter),
        "dropped_duplicate_url_rows": int(duplicate_url.sum()),
        "dropped_duplicate_title_rows": int(duplicate_title.sum()),
        "deduped_rows": int(len(deduped)),
    }
    return deduped, audit


def build_embedding_inputs(deduped: pd.DataFrame, policy: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_range = policy["stage5_sample_range"]
    windows = policy["stage5_decision"]["windows"]
    max_window = max(int(window) for window in windows)
    sample_min = pd.Timestamp(sample_range["date_min"])
    sample_max = pd.Timestamp(sample_range["date_max"])
    usable_start = (sample_min - pd.Timedelta(days=max_window)).date()
    usable_end = (sample_max - pd.Timedelta(days=1)).date()

    news = deduped.copy()
    news["news_date_ts"] = pd.to_datetime(news["news_date"], errors="coerce")
    usable = news.loc[
        (news["news_date_ts"].dt.date >= usable_start)
        & (news["news_date_ts"].dt.date <= usable_end)
    ].copy()

    usable["embedding_input_text"] = usable["headline_text"]
    usable["embedding_input_hash"] = usable["embedding_input_text"].map(sha256_text)
    usable["headline_hash"] = usable["headline_text"].map(sha256_text)
    usable["news_item_id"] = usable["embedding_input_hash"].str.slice(0, 16).map(
        lambda value: f"stage5_news_{value}"
    )
    usable["embedding_input_char_count"] = usable["embedding_input_text"].str.len().astype(int)
    usable["embedding_input_word_count"] = (
        usable["embedding_input_text"].str.split().map(len).fillna(0).astype(int)
    )
    usable["embedding_template"] = "headline_only_v1"
    usable["embedding_unit"] = "news_item"
    usable["news_date"] = pd.to_datetime(usable["news_date"]).dt.strftime("%Y-%m-%d")
    usable["news_time"] = pd.to_datetime(usable["news_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    columns = [
        "news_item_id",
        "news_time",
        "news_date",
        "source_text",
        "url_text",
        "headline_text",
        "normalized_title",
        "embedding_template",
        "embedding_unit",
        "embedding_input_text",
        "embedding_input_hash",
        "headline_hash",
        "embedding_input_char_count",
        "embedding_input_word_count",
        "title_length_chars",
        "article_length_chars",
        "has_article_text",
    ]
    usable = usable[columns].sort_values(["news_date", "news_time", "news_item_id"]).reset_index(
        drop=True
    )

    input_policy = {
        "embedding_template": "headline_only_v1",
        "embedding_input_text": "headline_text only",
        "excluded_from_embedding_text": ["news_date", "source_text", "url_text"],
        "exclusion_reason": (
            "Date/source/url are retained as metadata but excluded from embedding text "
            "to reduce time/source leakage and keep the representation content-based."
        ),
        "usable_news_date_min": str(usable_start),
        "usable_news_date_max": str(usable_end),
        "stage_sample_date_min": sample_range["date_min"],
        "stage_sample_date_max": sample_range["date_max"],
        "windows": windows,
        "strict_lag": policy["leakage_policy"],
    }
    return usable, input_policy


def write_report(manifest: dict[str, Any], summary: pd.DataFrame) -> None:
    metric = dict(zip(summary["metric"], summary["value"]))
    report = f"""# 5-3 Embedding Input Construction

Status: completed.

## Decision

Stage 5 will embed each deduplicated news headline as one item. The embedding
text is headline-only:

```text
<headline>
```

Date, source, and URL are kept as metadata but are not included in the embedding
input. This keeps the first embedding experiment content-based and avoids
encoding calendar/source artifacts into the vector.

## Counts

- Raw news rows: `{metric["raw_rows"]}`
- Deduplicated news rows: `{metric["deduped_rows"]}`
- Stage 5 usable news rows: `{metric["usable_rows"]}`
- Usable news date range: `{metric["usable_news_date_min"]}` to `{metric["usable_news_date_max"]}`
- Unique usable news dates: `{metric["usable_unique_news_dates"]}`
- Mean input chars: `{metric["mean_input_chars"]}`
- Max input chars: `{metric["max_input_chars"]}`
- Duplicate embedding hashes after dedupe: `{metric["duplicate_embedding_hashes"]}`

## Outputs

- Full input table: `{manifest["outputs"]["embedding_input_items"]}`
- Sample preview: `{manifest["outputs"]["embedding_input_examples"]}`
- Summary table: `{manifest["outputs"]["embedding_input_summary"]}`
- Manifest: `{manifest["outputs"]["manifest"]}`

## Next Step

Proceed to `5-4/5-5`: define cache policy and call the embedding backend.
"""
    (REPORT_TABLES / "stage5_embedding_input_construction_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source_path = resolve_news_source()
    policy = read_json(REPORT_TABLES / "stage5_news_leakage_policy.json")

    raw = pd.read_csv(source_path)
    standardized = standardize_news(raw)
    deduped, dedupe_audit = dedupe_news(standardized)
    inputs, input_policy = build_embedding_inputs(deduped, policy)

    input_path = OUTPUT_ROOT / "stage5_embedding_input_items.csv"
    sample_path = REPORT_TABLES / "stage5_embedding_input_examples.csv"
    summary_path = REPORT_TABLES / "stage5_embedding_input_summary.csv"
    manifest_path = DATA_INVENTORY / "stage5_embedding_input_manifest.json"

    inputs.to_csv(input_path, index=False)
    inputs.head(30).to_csv(sample_path, index=False)

    summary_rows = [
        ("raw_rows", len(raw)),
        ("deduped_rows", dedupe_audit["deduped_rows"]),
        ("usable_rows", len(inputs)),
        ("usable_news_date_min", inputs["news_date"].min()),
        ("usable_news_date_max", inputs["news_date"].max()),
        ("usable_unique_news_dates", inputs["news_date"].nunique()),
        ("mean_input_chars", round(float(inputs["embedding_input_char_count"].mean()), 3)),
        ("max_input_chars", int(inputs["embedding_input_char_count"].max())),
        (
            "duplicate_embedding_hashes",
            int(inputs["embedding_input_hash"].duplicated(keep=False).sum()),
        ),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary.to_csv(summary_path, index=False)

    manifest = {
        "status": "ok",
        "stage": "5-3",
        "source_file": str(source_path),
        "dedupe_policy": {
            "sort_keys": ["news_time", "url_text", "normalized_title"],
            "drop_duplicate_non_empty_url": True,
            "drop_duplicate_normalized_title": True,
            "audit": dedupe_audit,
        },
        "input_policy": input_policy,
        "embedding_backend_target": {
            "backend": "openai",
            "model": "text-embedding-3-small",
            "api_call_performed": False,
        },
        "outputs": {
            "embedding_input_items": str(input_path),
            "embedding_input_examples": str(sample_path),
            "embedding_input_summary": str(summary_path),
            "manifest": str(manifest_path),
            "report": str(REPORT_TABLES / "stage5_embedding_input_construction_report.md"),
        },
    }
    write_json(manifest_path, manifest)
    write_json(REPORT_TABLES / "stage5_embedding_input_manifest.json", manifest)
    write_report(manifest, summary)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
