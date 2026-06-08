#!/usr/bin/env python3
"""Write the Stage 5 embedding backend and cache policy report.

This is a planning/manifest step. It does not call OpenAI, Voyage, or any
local embedding model.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
DATA_INVENTORY = ROOT / "data_inventory"
CONFIG_PATH = ROOT / "configs" / "stage5_embedding_backends.yaml"
INPUT_MANIFEST_PATH = DATA_INVENTORY / "stage5_embedding_input_manifest.json"


BACKEND_ROWS = [
    {
        "run_id": "openai_text_embedding_3_small_primary",
        "execution_order": 1,
        "provider": "openai",
        "model": "text-embedding-3-small",
        "role": "primary",
        "api_key_env": "OPENAI_API_KEY",
        "default_dimensions": 1536,
        "requested_dimensions": 1536,
        "batch_size": 128,
        "status": "run_first",
        "artifact_root": "outputs/stage5/embeddings/openai_text_embedding_3_small",
        "cache_manifest": "data_inventory/openai_text_embedding_3_small_cache_manifest.json",
        "current_price_per_1m_tokens_usd": 0.02,
        "price_source": "OpenAI pricing page, checked 2026-06-08",
        "reason": "First-pass balance of quality, cost, and sufficient dimensionality before train-only SVD.",
    },
    {
        "run_id": "voyage_4_anthropic_side_comparison",
        "execution_order": 2,
        "provider": "voyageai",
        "model": "voyage-4",
        "role": "anthropic_side_comparison",
        "api_key_env": "VOYAGE_API_KEY",
        "default_dimensions": 1024,
        "requested_dimensions": 1024,
        "batch_size": 128,
        "status": "run_after_openai_small",
        "artifact_root": "outputs/stage5/embeddings/voyage_4",
        "cache_manifest": "data_inventory/voyage_4_cache_manifest.json",
        "current_price_per_1m_tokens_usd": None,
        "price_source": "Verify on Voyage AI before API execution.",
        "reason": "Anthropic docs state Claude has no native embeddings and recommend Voyage AI as an embedding provider.",
    },
    {
        "run_id": "openai_text_embedding_3_large_optional",
        "execution_order": 3,
        "provider": "openai",
        "model": "text-embedding-3-large",
        "role": "optional_stronger_openai_check",
        "api_key_env": "OPENAI_API_KEY",
        "default_dimensions": 3072,
        "requested_dimensions": 3072,
        "batch_size": 128,
        "status": "optional",
        "artifact_root": "outputs/stage5/embeddings/openai_text_embedding_3_large",
        "cache_manifest": "data_inventory/openai_text_embedding_3_large_cache_manifest.json",
        "current_price_per_1m_tokens_usd": 0.13,
        "price_source": "OpenAI pricing page, checked 2026-06-08",
        "reason": "OpenAI upper-bound comparison if small is promising or inconclusive.",
    },
    {
        "run_id": "voyage_4_large_optional",
        "execution_order": 4,
        "provider": "voyageai",
        "model": "voyage-4-large",
        "role": "optional_stronger_anthropic_side_check",
        "api_key_env": "VOYAGE_API_KEY",
        "default_dimensions": 1024,
        "requested_dimensions": 1024,
        "batch_size": 128,
        "status": "optional",
        "artifact_root": "outputs/stage5/embeddings/voyage_4_large",
        "cache_manifest": "data_inventory/voyage_4_large_cache_manifest.json",
        "current_price_per_1m_tokens_usd": None,
        "price_source": "Verify on Voyage AI before API execution.",
        "reason": "Stronger Voyage comparison if voyage-4 is promising or inconclusive.",
    },
]


LOCAL_FALLBACK_ROWS = [
    {
        "run_id": "bge_small_en_v1_5",
        "provider": "local_huggingface",
        "model": "BAAI/bge-small-en-v1.5",
        "role": "reproducible_local_fallback",
        "default_dimensions": 384,
        "reason": "No paid API dependency; useful if OpenAI/Voyage access is blocked.",
    },
    {
        "run_id": "all_minilm_l6_v2",
        "provider": "local_huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "role": "lightweight_local_fallback",
        "default_dimensions": 384,
        "reason": "Very light sentence-transformer baseline, not the main thesis model.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input_items(manifest: dict[str, Any]) -> Path:
    output_path = Path(manifest["outputs"]["embedding_input_items"])
    if output_path.exists():
        return output_path
    relative = ROOT / "outputs" / "stage5" / "embedding_inputs" / "stage5_embedding_input_items.csv"
    if relative.exists():
        return relative
    raise FileNotFoundError(
        "Stage 5 embedding input table is missing. Run 5-3 first. "
        f"Tried: {output_path}, {relative}"
    )


def estimate_tokens_from_chars(total_chars: int) -> int:
    # This is a conservative planning estimate only. The API job will record
    # actual token usage returned by the provider when available.
    return int(math.ceil(total_chars / 4.0))


def build_report(
    input_stats: dict[str, Any],
    backend_df: pd.DataFrame,
    cache_policy: dict[str, Any],
) -> str:
    primary = backend_df.loc[backend_df["run_id"] == "openai_text_embedding_3_small_primary"].iloc[0]
    voyage = backend_df.loc[backend_df["run_id"] == "voyage_4_anthropic_side_comparison"].iloc[0]
    large = backend_df.loc[backend_df["run_id"] == "openai_text_embedding_3_large_optional"].iloc[0]
    return f"""# 5-4 Embedding Backend And Cache Policy

Status: completed.

## Locked Execution Order

1. `{primary["model"]}` via OpenAI.
2. `{voyage["model"]}` via Voyage AI as the Anthropic-side comparison.
3. `{large["model"]}` as optional OpenAI upper-bound comparison.
4. `voyage-4-large` as optional stronger Voyage comparison.

Claude itself is not used for embeddings because Anthropic documentation states
that Anthropic does not offer a native embedding model. Voyage AI is used as
the Claude/Anthropic-side embedding comparison.

## Input Scale

- Embedding items: `{input_stats["item_count"]:,}`
- Input template: `{input_stats["input_template"]}`
- Total input characters: `{input_stats["total_chars"]:,}`
- Rough token estimate: `{input_stats["rough_token_estimate"]:,}`
- Max input characters: `{input_stats["max_chars"]}`

## Cache Policy

- API keys are read only from environment variables.
- API keys are never written to configs, CSVs, JSON manifests, logs, or reports.
- Every embedding row is keyed by `input_hash + provider + model + requested_dimensions`.
- Reruns must skip already cached matching rows.
- Failed items must be written to a failure log and retried separately.
- GitHub tracks only small manifests/reports; embedding matrices remain in local/Kaggle artifacts.

## Required Cache Manifest Fields

```text
{chr(10).join(cache_policy["required_manifest_fields"])}
```

## Next Step

Proceed to `5-5`: generate the headline-level embedding table for
`text-embedding-3-small`.
"""


def main() -> None:
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY.mkdir(parents=True, exist_ok=True)

    input_manifest = read_json(INPUT_MANIFEST_PATH)
    input_items_path = resolve_input_items(input_manifest)
    input_items = pd.read_csv(input_items_path)

    total_chars = int(input_items["embedding_input_char_count"].sum())
    input_stats = {
        "item_count": int(len(input_items)),
        "input_template": input_manifest["input_policy"]["embedding_template"],
        "input_items_path": str(input_items_path),
        "input_items_sha256": sha256_file(input_items_path),
        "total_chars": total_chars,
        "rough_token_estimate": estimate_tokens_from_chars(total_chars),
        "mean_chars": round(float(input_items["embedding_input_char_count"].mean()), 3),
        "max_chars": int(input_items["embedding_input_char_count"].max()),
        "duplicate_embedding_hashes": int(
            input_items["embedding_input_hash"].duplicated(keep=False).sum()
        ),
    }

    backend_df = pd.DataFrame(BACKEND_ROWS)
    for column in ["current_price_per_1m_tokens_usd"]:
        backend_df[column] = pd.to_numeric(backend_df[column], errors="coerce")
    backend_df["rough_estimated_input_cost_usd"] = backend_df[
        "current_price_per_1m_tokens_usd"
    ].map(
        lambda price: None
        if pd.isna(price)
        else round(input_stats["rough_token_estimate"] / 1_000_000 * float(price), 6)
    )

    local_df = pd.DataFrame(LOCAL_FALLBACK_ROWS)
    cache_policy = {
        "never_store_api_keys": True,
        "api_key_storage": "environment_variables_only",
        "cache_key": ["embedding_input_hash", "provider", "model", "requested_dimensions"],
        "rerun_rule": "skip existing matching input_hash/provider/model/dim rows",
        "failure_rule": "write failed item ids and retry only failures",
        "github_policy": "commit small manifests only; never commit embedding matrices",
        "required_manifest_fields": [
            "provider",
            "model",
            "requested_dimensions",
            "input_manifest_path",
            "input_manifest_sha256",
            "input_template",
            "input_hash_column",
            "item_count",
            "created_utc",
            "embedding_array_path",
            "item_manifest_path",
            "failure_log_path",
        ],
    }

    outputs = {
        "backend_plan": str(REPORT_TABLES / "stage5_embedding_backend_plan.csv"),
        "local_fallbacks": str(REPORT_TABLES / "stage5_embedding_local_fallbacks.csv"),
        "cache_policy": str(DATA_INVENTORY / "stage5_embedding_cache_policy.json"),
        "cache_policy_report_copy": str(REPORT_TABLES / "stage5_embedding_cache_policy.json"),
        "report": str(REPORT_TABLES / "stage5_embedding_backend_cache_policy_report.md"),
    }
    manifest = {
        "status": "ok",
        "stage": "5-4",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(CONFIG_PATH),
        "input_stats": input_stats,
        "primary_backend": "openai_text_embedding_3_small_primary",
        "anthropic_side_comparison": "voyage_4_anthropic_side_comparison",
        "optional_backends": [
            "openai_text_embedding_3_large_optional",
            "voyage_4_large_optional",
        ],
        "cache_policy": cache_policy,
        "sources": {
            "openai_embeddings_guide": "https://platform.openai.com/docs/guides/embeddings",
            "openai_pricing": "https://platform.openai.com/docs/pricing",
            "anthropic_embeddings": "https://platform.claude.com/docs/en/docs/build-with-claude/embeddings",
            "voyage_docs": "https://docs.voyageai.com/",
        },
        "outputs": outputs,
    }

    backend_df.to_csv(REPORT_TABLES / "stage5_embedding_backend_plan.csv", index=False)
    local_df.to_csv(REPORT_TABLES / "stage5_embedding_local_fallbacks.csv", index=False)
    write_json(DATA_INVENTORY / "stage5_embedding_cache_policy.json", manifest)
    write_json(REPORT_TABLES / "stage5_embedding_cache_policy.json", manifest)
    (REPORT_TABLES / "stage5_embedding_backend_cache_policy_report.md").write_text(
        build_report(input_stats, backend_df, cache_policy), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
