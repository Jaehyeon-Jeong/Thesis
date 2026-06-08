#!/usr/bin/env python3
"""Build headline-level embedding artifacts for Stage 5.

The script supports resumable OpenAI embedding generation. It writes each
successful batch to disk before building the final matrix, so interrupted runs
can resume without paying for already cached items again.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"
DEFAULT_INPUT_MANIFEST = DATA_INVENTORY / "stage5_embedding_input_manifest.json"
DEFAULT_BACKEND_PLAN = REPORT_TABLES / "stage5_embedding_backend_plan.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="openai_text_embedding_3_small_primary")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--backend-plan", default=str(DEFAULT_BACKEND_PLAN))
    parser.add_argument("--limit", type=int, default=0, help="Optional item limit for smoke runs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override backend-plan batch size.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore final manifest and rebuild batches.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input_items(input_manifest: dict[str, Any]) -> Path:
    output_path = Path(input_manifest["outputs"]["embedding_input_items"])
    if output_path.exists():
        return output_path
    fallback = ROOT / "outputs" / "stage5" / "embedding_inputs" / "stage5_embedding_input_items.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Stage 5 embedding input table is missing. Run 5-3 first. "
        f"Tried: {output_path}, {fallback}"
    )


def load_backend_config(backend_plan_path: Path, run_id: str) -> dict[str, Any]:
    if not backend_plan_path.exists():
        raise FileNotFoundError(f"Backend plan missing: {backend_plan_path}")
    plan = pd.read_csv(backend_plan_path)
    matches = plan.loc[plan["run_id"].eq(run_id)]
    if matches.empty:
        raise ValueError(f"Unknown run-id {run_id!r}. Available: {sorted(plan['run_id'].tolist())}")
    row = matches.iloc[0].to_dict()
    if row["provider"] != "openai":
        raise ValueError(
            f"This script currently implements OpenAI embeddings only; got provider={row['provider']!r}."
        )
    return row


def final_manifest_path(config: dict[str, Any]) -> Path:
    return ROOT / str(config["cache_manifest"])


def output_root(config: dict[str, Any]) -> Path:
    return ROOT / str(config["artifact_root"])


def batch_id(start: int, end: int) -> str:
    return f"{start:06d}_{end:06d}"


def batch_paths(root: Path, start: int, end: int) -> tuple[Path, Path, Path]:
    bid = batch_id(start, end)
    batches = root / "batches"
    return (
        batches / f"embedding_vectors_{bid}.npy",
        batches / f"embedding_items_{bid}.csv",
        batches / f"embedding_response_{bid}.json",
    )


def call_openai_embeddings(
    *,
    api_key: str,
    model: str,
    texts: list[str],
    dimensions: int | None,
    timeout: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    if dimensions is not None:
        payload["dimensions"] = int(dimensions)

    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    data = sorted(body["data"], key=lambda item: int(item["index"]))
    vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
    usage = body.get("usage", {})
    return vectors, {"usage": usage, "response_model": body.get("model", model)}


def run_with_retries(
    *,
    api_key: str,
    config: dict[str, Any],
    texts: list[str],
    dimensions: int | None,
    max_retries: int,
    retry_base_seconds: float,
    timeout: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_openai_embeddings(
                api_key=api_key,
                model=str(config["model"]),
                texts=texts,
                dimensions=dimensions,
                timeout=timeout,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = repr(exc)
            if attempt >= max_retries:
                break
            sleep_seconds = retry_base_seconds * (2**attempt)
            print(f"[retry] attempt={attempt + 1} error={last_error} sleep={sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"OpenAI embedding request failed after retries: {last_error}")


def write_batch(
    *,
    root: Path,
    items: pd.DataFrame,
    vectors: np.ndarray,
    response_meta: dict[str, Any],
    start: int,
    end: int,
    config: dict[str, Any],
    dimensions: int,
) -> None:
    vector_path, item_path, response_path = batch_paths(root, start, end)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, vectors.astype(np.float32))

    batch_items = items.copy()
    batch_items["embedding_provider"] = str(config["provider"])
    batch_items["embedding_model"] = str(config["model"])
    batch_items["requested_dimensions"] = int(dimensions)
    batch_items["embedding_dim"] = int(vectors.shape[1])
    batch_items["embedding_status"] = "ok"
    batch_items["embedding_batch_id"] = batch_id(start, end)
    batch_items["embedding_batch_path"] = str(vector_path)
    batch_items.to_csv(item_path, index=False)

    write_json(
        response_path,
        {
            "batch_id": batch_id(start, end),
            "start": int(start),
            "end": int(end),
            "num_items": int(len(items)),
            "vector_shape": list(vectors.shape),
            "provider": str(config["provider"]),
            "model": str(config["model"]),
            "requested_dimensions": int(dimensions),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **response_meta,
        },
    )


def collect_batches(root: Path, expected_rows: int) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]]]:
    batch_item_paths = sorted((root / "batches").glob("embedding_items_*.csv"))
    if not batch_item_paths:
        raise FileNotFoundError(f"No embedding batch item files found under {root / 'batches'}")

    item_frames: list[pd.DataFrame] = []
    vector_arrays: list[np.ndarray] = []
    responses: list[dict[str, Any]] = []
    for item_path in batch_item_paths:
        suffix = item_path.stem.replace("embedding_items_", "")
        vector_path = item_path.parent / f"embedding_vectors_{suffix}.npy"
        response_path = item_path.parent / f"embedding_response_{suffix}.json"
        if not vector_path.exists():
            raise FileNotFoundError(f"Missing vector batch for {item_path}: {vector_path}")
        frame = pd.read_csv(item_path)
        vectors = np.load(vector_path)
        if len(frame) != len(vectors):
            raise ValueError(f"Batch row mismatch for {suffix}: rows={len(frame)} vectors={len(vectors)}")
        item_frames.append(frame)
        vector_arrays.append(vectors.astype(np.float32))
        if response_path.exists():
            responses.append(read_json(response_path))

    items = pd.concat(item_frames, ignore_index=True)
    vectors = np.vstack(vector_arrays).astype(np.float32)
    if len(items) != expected_rows or vectors.shape[0] != expected_rows:
        raise ValueError(
            f"Final row mismatch: expected={expected_rows} items={len(items)} vectors={vectors.shape[0]}"
        )
    items["embedding_vector_index"] = np.arange(len(items), dtype=int)
    return vectors, items, responses


def write_report(manifest: dict[str, Any], report_path: Path) -> None:
    report = f"""# 5-5 Headline-Level Embedding Table

Status: {manifest["status"]}.

## Backend

- Provider: `{manifest["provider"]}`
- Model: `{manifest["model"]}`
- Requested dimensions: `{manifest["requested_dimensions"]}`
- Input template: `{manifest["input_template"]}`

## Rows

- Items requested: `{manifest["item_count"]:,}`
- Items completed: `{manifest["completed_count"]:,}`
- Vector shape: `{manifest.get("embedding_shape")}`

## Outputs

- Embedding matrix: `{manifest.get("embedding_array_path")}`
- Item manifest: `{manifest.get("item_manifest_path")}`
- Cache manifest: `{manifest.get("cache_manifest_path")}`
- Failure log: `{manifest.get("failure_log_path")}`

## Notes

Embedding matrices are generated artifacts and should not be committed to
GitHub. Keep them in local/Kaggle output bundles.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_manifest_path = Path(args.input_manifest)
    backend_plan_path = Path(args.backend_plan)
    input_manifest = read_json(input_manifest_path)
    config = load_backend_config(backend_plan_path, args.run_id)

    items_path = resolve_input_items(input_manifest)
    items = pd.read_csv(items_path)
    if args.limit and args.limit > 0:
        items = items.head(int(args.limit)).copy()
    items = items.reset_index(drop=True)
    items["input_row_index"] = np.arange(len(items), dtype=int)

    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    final_manifest = final_manifest_path(config)
    report_path = REPORT_TABLES / f"stage5_{args.run_id}_embedding_report.md"
    failure_path = root / "embedding_failures.csv"
    vector_output_path = root / "embedding_vectors.npy"
    item_output_path = root / "embedding_item_manifest.csv"

    requested_dimensions = int(config["requested_dimensions"])
    batch_size = int(args.batch_size or config["batch_size"])

    if final_manifest.exists() and not args.force:
        existing = read_json(final_manifest)
        if existing.get("status") == "ok" and int(existing.get("completed_count", -1)) == len(items):
            print(json.dumps(existing, indent=2, ensure_ascii=False))
            return

    dry_run_manifest = {
        "status": "dry_run",
        "stage": "5-5",
        "run_id": args.run_id,
        "provider": str(config["provider"]),
        "model": str(config["model"]),
        "requested_dimensions": requested_dimensions,
        "input_template": input_manifest["input_policy"]["embedding_template"],
        "item_count": int(len(items)),
        "batch_size": batch_size,
        "planned_batches": int(np.ceil(len(items) / batch_size)),
        "input_items_path": str(items_path),
        "artifact_root": str(root),
        "cache_manifest_path": str(final_manifest),
    }
    if args.dry_run:
        write_json(DATA_INVENTORY / f"{args.run_id}_dry_run_manifest.json", dry_run_manifest)
        print(json.dumps(dry_run_manifest, indent=2, ensure_ascii=False))
        return

    api_key_env = str(config["api_key_env"])
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        write_json(DATA_INVENTORY / f"{args.run_id}_missing_api_key_manifest.json", dry_run_manifest)
        raise RuntimeError(
            f"Missing {api_key_env}. Set it in the environment before running 5-5."
        )

    failures: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        vector_path, item_path, _ = batch_paths(root, start, end)
        if vector_path.exists() and item_path.exists() and not args.force:
            print(f"[skip] existing batch {batch_id(start, end)}")
            continue

        batch = items.iloc[start:end].copy()
        texts = batch["embedding_input_text"].fillna("").astype(str).tolist()
        print(f"[embed] batch={batch_id(start, end)} items={len(batch)}")
        try:
            vectors, response_meta = run_with_retries(
                api_key=api_key,
                config=config,
                texts=texts,
                dimensions=requested_dimensions,
                max_retries=int(args.max_retries),
                retry_base_seconds=float(args.retry_base_seconds),
                timeout=float(args.request_timeout),
            )
            write_batch(
                root=root,
                items=batch,
                vectors=vectors,
                response_meta=response_meta,
                start=start,
                end=end,
                config=config,
                dimensions=requested_dimensions,
            )
        except Exception as exc:
            for _, row in batch.iterrows():
                failures.append(
                    {
                        "input_row_index": int(row["input_row_index"]),
                        "news_item_id": row["news_item_id"],
                        "embedding_input_hash": row["embedding_input_hash"],
                        "error": repr(exc),
                    }
                )
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            raise

    vectors, item_manifest, responses = collect_batches(root, expected_rows=len(items))
    np.save(vector_output_path, vectors.astype(np.float32))
    item_manifest.to_csv(item_output_path, index=False)

    total_prompt_tokens = 0
    total_tokens = 0
    for response in responses:
        usage = response.get("usage", {})
        total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)

    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    else:
        pd.DataFrame(columns=["input_row_index", "news_item_id", "embedding_input_hash", "error"]).to_csv(
            failure_path, index=False
        )

    manifest = {
        "status": "ok",
        "stage": "5-5",
        "run_id": args.run_id,
        "provider": str(config["provider"]),
        "model": str(config["model"]),
        "requested_dimensions": requested_dimensions,
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(items_path),
        "input_template": input_manifest["input_policy"]["embedding_template"],
        "input_hash_column": "embedding_input_hash",
        "item_count": int(len(items)),
        "completed_count": int(len(item_manifest)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_root": str(root),
        "embedding_array_path": str(vector_output_path),
        "embedding_shape": list(vectors.shape),
        "item_manifest_path": str(item_output_path),
        "failure_log_path": str(failure_path),
        "cache_manifest_path": str(final_manifest),
        "batch_count": int(len(responses)),
        "api_usage": {
            "prompt_tokens": int(total_prompt_tokens),
            "total_tokens": int(total_tokens),
        },
    }
    write_json(final_manifest, manifest)
    write_json(REPORT_TABLES / f"{args.run_id}_cache_manifest.json", manifest)
    write_report(manifest, report_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
