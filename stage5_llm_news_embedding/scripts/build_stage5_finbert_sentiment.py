#!/usr/bin/env python3
"""Build FinBERT headline sentiment labels for Stage 5 news items.

This is Stage 5 5-9A. It converts each deduplicated headline into a
financial-tone label and probabilities. It does not train a market model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"
DEFAULT_INPUT_MANIFEST = DATA_INVENTORY / "stage5_embedding_input_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="finbert_prosusai_headline_v1")
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--input-items", default="", help="Optional direct input CSV override.")
    parser.add_argument("--text-column", default="embedding_input_text")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--label-order",
        default="positive,negative,neutral",
        help=(
            "Fallback class order when model id2label is generic, e.g. "
            "'positive,negative,neutral'."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if final output exists.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_input_items(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if args.input_items:
        candidates.append(Path(args.input_items))

    manifest_path = Path(args.input_manifest)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        output_path = manifest.get("outputs", {}).get("embedding_input_items")
        if output_path:
            candidates.append(Path(output_path))

    candidates.extend(
        [
            ROOT / "outputs/stage5/embedding_inputs/stage5_embedding_input_items.csv",
            ROOT / "reports/tables/stage5_embedding_input_items.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Stage 5 news input table is missing. Run 5-3 or attach a bundle with "
        f"stage5_embedding_input_items.csv. Tried: {tried}"
    )


def pick_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_label(value: object) -> str:
    return str(value or "").strip().lower().replace("label_", "")


def build_label_mapping(model, fallback_order_text: str) -> dict[str, int]:
    id2label = getattr(model.config, "id2label", {}) or {}
    label_by_id = {int(idx): normalize_label(label) for idx, label in id2label.items()}

    mapping: dict[str, int] = {}
    for idx, label in label_by_id.items():
        for wanted in ["positive", "negative", "neutral"]:
            if wanted in label:
                mapping[wanted] = idx

    if set(mapping) == {"positive", "negative", "neutral"}:
        return mapping

    fallback = [normalize_label(item) for item in fallback_order_text.split(",")]
    if len(fallback) != int(model.config.num_labels):
        raise ValueError(
            "Could not infer FinBERT label mapping from model config and fallback "
            f"label count does not match model labels. id2label={id2label}, "
            f"fallback={fallback}"
        )
    mapping = {label: idx for idx, label in enumerate(fallback)}
    missing = {"positive", "negative", "neutral"} - set(mapping)
    if missing:
        raise ValueError(f"Fallback label order is missing labels: {sorted(missing)}")
    return mapping


def run_finbert(
    *,
    frame: pd.DataFrame,
    text_column: str,
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str,
    label_order: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if text_column not in frame.columns:
        raise ValueError(f"Text column {text_column!r} not found in input table.")

    device = pick_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    mapping = build_label_mapping(model, label_order)
    texts = frame[text_column].fillna("").astype(str).tolist()
    probabilities: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        print(f"[finbert] batch={start:06d}_{end:06d} items={len(batch_texts)}", flush=True)
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probabilities.append(probs.astype(np.float32))

    prob = np.vstack(probabilities)
    positive = prob[:, mapping["positive"]]
    negative = prob[:, mapping["negative"]]
    neutral = prob[:, mapping["neutral"]]
    labels = np.asarray(["positive", "negative", "neutral"])
    ordered = np.column_stack([positive, negative, neutral])
    label_idx = ordered.argmax(axis=1)

    result = frame.copy()
    result["finbert_positive_prob"] = positive
    result["finbert_negative_prob"] = negative
    result["finbert_neutral_prob"] = neutral
    result["finbert_label"] = labels[label_idx]
    result["finbert_confidence"] = ordered.max(axis=1)
    result["finbert_sentiment_score"] = positive - negative
    result["finbert_model"] = model_name
    result["finbert_text_column"] = text_column

    meta = {
        "model_name": model_name,
        "device": str(device),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "model_num_labels": int(model.config.num_labels),
        "model_id2label": {
            str(key): str(value) for key, value in (getattr(model.config, "id2label", {}) or {}).items()
        },
        "label_mapping": {key: int(value) for key, value in mapping.items()},
    }
    return result, meta


def write_report(manifest: dict[str, Any], summary: pd.DataFrame) -> None:
    metrics = dict(zip(summary["metric"], summary["value"]))
    report = f"""# 5-9A FinBERT Headline Sentiment Extraction

Status: `{manifest["status"]}`.

## Setup

- Model: `{manifest["model"]["model_name"]}`
- Input rows: `{metrics["num_rows"]}`
- News date range: `{metrics["news_date_min"]}` to `{metrics["news_date_max"]}`
- Text column: `{manifest["text_column"]}`
- Device: `{manifest["model"]["device"]}`

## Label Distribution

- Positive rows: `{metrics["positive_rows"]}`
- Negative rows: `{metrics["negative_rows"]}`
- Neutral rows: `{metrics["neutral_rows"]}`
- Mean sentiment score: `{metrics["mean_sentiment_score"]}`
- Mean confidence: `{metrics["mean_confidence"]}`

## Interpretation

This artifact is a headline-level financial text tone table. It is not the
official Crypto Fear & Greed Index and it is not a direct BTC return forecast.
The next step is to aggregate these item-level labels into strict `t-1`
7/20/60-day news sentiment windows.

## Outputs

- Sentiment table: `{manifest["outputs"]["finbert_sentiment_items"]}`
- Summary: `{manifest["outputs"]["summary"]}`
- Manifest: `{manifest["outputs"]["manifest"]}`
"""
    Path(manifest["outputs"]["report"]).write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)

    output_root = ROOT / "outputs" / "stage5" / "finbert_sentiment" / args.run_id
    item_output = output_root / "stage5_finbert_sentiment_items.csv"
    summary_output = REPORT_TABLES / f"{args.run_id}_summary.csv"
    manifest_output = DATA_INVENTORY / f"{args.run_id}_manifest.json"
    report_output = REPORT_TABLES / f"{args.run_id}_report.md"

    if item_output.exists() and manifest_output.exists() and not args.force:
        print(json.dumps(read_json(manifest_output), indent=2, ensure_ascii=False))
        return

    input_path = resolve_input_items(args)
    frame = pd.read_csv(input_path)
    if args.limit and args.limit > 0:
        frame = frame.head(args.limit).copy()

    result, model_meta = run_finbert(
        frame=frame,
        text_column=args.text_column,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_name=args.device,
        label_order=args.label_order,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(item_output, index=False)

    label_counts = result["finbert_label"].value_counts().to_dict()
    summary_rows = [
        ("num_rows", int(len(result))),
        ("news_date_min", str(result["news_date"].min()) if "news_date" in result.columns else ""),
        ("news_date_max", str(result["news_date"].max()) if "news_date" in result.columns else ""),
        ("positive_rows", int(label_counts.get("positive", 0))),
        ("negative_rows", int(label_counts.get("negative", 0))),
        ("neutral_rows", int(label_counts.get("neutral", 0))),
        ("mean_positive_prob", round(float(result["finbert_positive_prob"].mean()), 6)),
        ("mean_negative_prob", round(float(result["finbert_negative_prob"].mean()), 6)),
        ("mean_neutral_prob", round(float(result["finbert_neutral_prob"].mean()), 6)),
        ("mean_sentiment_score", round(float(result["finbert_sentiment_score"].mean()), 6)),
        ("std_sentiment_score", round(float(result["finbert_sentiment_score"].std(ddof=1)), 6)),
        ("mean_confidence", round(float(result["finbert_confidence"].mean()), 6)),
        ("low_confidence_lt_0p50_rows", int(result["finbert_confidence"].lt(0.50).sum())),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary.to_csv(summary_output, index=False)

    manifest = {
        "status": "ok",
        "stage": "5-9A",
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_items": str(input_path),
        "text_column": args.text_column,
        "limit": int(args.limit),
        "model": model_meta,
        "outputs": {
            "finbert_sentiment_items": str(item_output),
            "summary": str(summary_output),
            "manifest": str(manifest_output),
            "report": str(report_output),
        },
    }
    write_json(manifest_output, manifest)
    write_json(REPORT_TABLES / f"{args.run_id}_manifest.json", manifest)
    write_report(manifest, summary)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
