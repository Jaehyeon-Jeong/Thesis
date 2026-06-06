"""Build a compact selected-combo Stage 4 prebuilt context artifact.

This is the 4-N13-5B handoff step. It does not refit feature preprocessing.
Instead, it reads already-built train-normalized source contexts and copies a
small set of N13-5A-selected normalized columns into one compact context table.

Marker: N13-5B selected-combo context feature builder
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


OUTPUT_PREFIX = "stage4_selected_combo_context"
DEFAULT_FEATURE_SET_NAME = "n13_5a_combo6"
METADATA_COLUMNS = [
    "sample_index",
    "split",
    "Date",
    "image_end_date",
    "label_end_date",
    "start_index",
    "end_index",
    "label_end_index",
    "image_window",
    "return_horizon",
    "entry_close",
    "exit_close",
    "future_return",
    "label",
]


@dataclass(frozen=True)
class SelectedFeature:
    source_key: str
    source_feature: str
    target_feature: str
    rationale: str


BASE_SELECTED_FEATURES: tuple[SelectedFeature, ...] = (
    SelectedFeature("news", "news_svd_60d_09", "combo_news_svd_60d_09", "Top train-only N13-5A feature."),
    SelectedFeature("news", "news_svd_20d_18", "combo_news_svd_20d_18", "High train-only signal with lower redundancy than the top duplicate dimensions."),
    SelectedFeature("structured", "fg_mean_60", "combo_fg_mean_60", "Clean compact F&G regime level."),
    SelectedFeature("structured", "fg_delta_60", "combo_fg_delta_60", "F&G regime change over the matched 60-day window."),
    SelectedFeature("fsi", "ofr_fsi_std_60", "combo_ofr_fsi_std_60", "Best OFR FSI train-only feature in N13-5A."),
    SelectedFeature("roro", "riskoff_dollar_return_20", "combo_riskoff_dollar_return_20", "Best RORO/public macro train-only feature in N13-5A."),
)
OPTIONAL_TECHNICAL_FEATURE = SelectedFeature(
    "structured",
    "bb_bandwidth_60",
    "combo_bb_bandwidth_60",
    "Optional chart-derived volatility/width add-on; kept separate because technical context was mostly redundant.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_local.yaml")
    parser.add_argument("--image-window", type=int, default=60)
    parser.add_argument("--image-spec", default="ohlc_ma_vb")
    parser.add_argument("--return-horizon", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--feature-set-name", default=DEFAULT_FEATURE_SET_NAME)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--include-technical-bb-bandwidth", action="store_true")
    parser.add_argument("--structured-context-name", default="")
    parser.add_argument("--news-context-name", default="")
    parser.add_argument("--fsi-context-name", default="")
    parser.add_argument("--roro-context-name", default="")
    parser.add_argument("--structured-context-csv", type=Path, default=None)
    parser.add_argument("--news-context-csv", type=Path, default=None)
    parser.add_argument("--fsi-context-csv", type=Path, default=None)
    parser.add_argument("--roro-context-csv", type=Path, default=None)
    parser.add_argument("--write-report-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(Path(args.config))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = _paths_from_config(config)
    paths["context_root"].mkdir(parents=True, exist_ok=True)
    paths["tables_root"].mkdir(parents=True, exist_ok=True)

    context_names = _default_context_names(args)
    source_paths = {
        "structured": _resolve_source_context_csv(
            explicit=args.structured_context_csv,
            context_root=paths["context_root"],
            context_name=context_names["structured"],
            run_seed=int(args.run_seed),
            label="structured F&G/technical context",
        ),
        "news": _resolve_source_context_csv(
            explicit=args.news_context_csv,
            context_root=paths["context_root"],
            context_name=context_names["news"],
            run_seed=int(args.run_seed),
            label="news context",
        ),
        "fsi": _resolve_source_context_csv(
            explicit=args.fsi_context_csv,
            context_root=paths["context_root"],
            context_name=context_names["fsi"],
            run_seed=int(args.run_seed),
            label="FSI context",
        ),
        "roro": _resolve_source_context_csv(
            explicit=args.roro_context_csv,
            context_root=paths["context_root"],
            context_name=context_names["roro"],
            run_seed=int(args.run_seed),
            label="RORO context",
        ),
    }
    source_tables = {key: _read_source_table(path) for key, path in source_paths.items()}
    selected_features = list(BASE_SELECTED_FEATURES)
    if args.include_technical_bb_bandwidth:
        selected_features.append(OPTIONAL_TECHNICAL_FEATURE)

    output = _build_selected_context_table(source_tables, selected_features)
    context_name = make_selected_combo_context_name(
        output_prefix=str(args.output_prefix),
        image_window=int(args.image_window),
        image_spec=str(args.image_spec),
        return_horizon=int(args.return_horizon),
        feature_set_name=str(args.feature_set_name),
    )
    output_dir = paths["context_root"] / context_name / f"seed_{int(args.run_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    context_csv = output_dir / "context_features.csv"
    scaler_json = output_dir / "context_scaler.json"
    audit_json = output_dir / "context_feature_audit.json"
    summary_csv = output_dir / "context_feature_summary.csv"
    manifest_json = output_dir / "selected_combo_context_manifest.json"

    output.to_csv(context_csv, index=False)
    feature_order = [feature.target_feature for feature in selected_features]
    normalized = [f"{feature}_normalized" for feature in feature_order]
    scaler = _build_scaler_payload(selected_features, normalized)
    audit = _build_audit(
        context_name=context_name,
        source_paths=source_paths,
        selected_features=selected_features,
        output=output,
        feature_set_name=str(args.feature_set_name),
    )
    summary = _build_feature_summary(output, feature_order)

    scaler_json.write_text(json.dumps(_jsonable(scaler), indent=2), encoding="utf-8")
    audit_json.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    summary.to_csv(summary_csv, index=False)
    manifest = {
        "status": "ok",
        "stage": "4-N13-5B",
        "context_name": context_name,
        "feature_set_name": str(args.feature_set_name),
        "context_dim": len(feature_order),
        "feature_order": feature_order,
        "normalized_feature_columns": normalized,
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "outputs": {
            "context_features": str(context_csv),
            "context_scaler": str(scaler_json),
            "context_feature_audit": str(audit_json),
            "context_feature_summary": str(summary_csv),
        },
    }
    manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    report_files: dict[str, str] = {}
    if args.write_report_copy:
        report_prefix = f"{context_name}_seed{int(args.run_seed)}"
        report_audit = paths["tables_root"] / f"{report_prefix}_selected_combo_context_feature_audit.json"
        report_summary = paths["tables_root"] / f"{report_prefix}_selected_combo_context_feature_summary.csv"
        report_manifest = paths["tables_root"] / f"{report_prefix}_selected_combo_context_manifest.json"
        report_audit.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
        summary.to_csv(report_summary, index=False)
        report_manifest.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        report_files = {
            "report_audit": str(report_audit),
            "report_summary": str(report_summary),
            "report_manifest": str(report_manifest),
        }

    print(
        json.dumps(
            _jsonable(
                {
                    "status": "ok",
                    "stage": "4-N13-5B",
                    "context_name": context_name,
                    "context_dim": len(feature_order),
                    "feature_order": feature_order,
                    "split_counts": output["split"].astype(str).value_counts().sort_index().to_dict(),
                    "written": {
                        "context_features": str(context_csv),
                        "context_scaler": str(scaler_json),
                        "context_feature_audit": str(audit_json),
                        "context_feature_summary": str(summary_csv),
                        "selected_combo_context_manifest": str(manifest_json),
                        **report_files,
                    },
                }
            ),
            indent=2,
        )
    )


def _resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser()
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[1] / path


def _paths_from_config(config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths", {})
    return {
        "context_root": Path(str(paths["context_root"])).expanduser(),
        "tables_root": Path(str(paths["tables_root"])).expanduser(),
    }


def _default_context_names(args: argparse.Namespace) -> dict[str, str]:
    image_window = int(args.image_window)
    image_spec = str(args.image_spec)
    return_horizon = int(args.return_horizon)
    return {
        "structured": str(args.structured_context_name)
        or f"stage4_context_i{image_window}_{image_spec}_r{return_horizon}_c{image_window}",
        "news": str(args.news_context_name)
        or f"stage4_news_context_i{image_window}_{image_spec}_r{return_horizon}_tfidf_svd_w7_20_60",
        "fsi": str(args.fsi_context_name)
        or f"stage4_fsi_context_i{image_window}_{image_spec}_r{return_horizon}_ofr_fsi_lag1_w20_60",
        "roro": str(args.roro_context_name)
        or f"stage4_roro_context_i{image_window}_{image_spec}_r{return_horizon}_public_roro_pca_lag1_w20_60",
    }


def _resolve_source_context_csv(
    *,
    explicit: Path | None,
    context_root: Path,
    context_name: str,
    run_seed: int,
    label: str,
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        [
            context_root / context_name / f"seed_{int(run_seed)}" / "context_features.csv",
            context_root / context_name / f"seed_{int(run_seed)}" / "context_features.parquet",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {label}. Tried: " + ", ".join(str(path) for path in candidates)
    )


def _read_source_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build_selected_context_table(
    source_tables: dict[str, pd.DataFrame],
    selected_features: list[SelectedFeature],
) -> pd.DataFrame:
    base = source_tables["structured"].copy()
    metadata = [column for column in METADATA_COLUMNS if column in base.columns]
    output = base[metadata].copy()
    if "image_end_date" not in output.columns and "Date" in output.columns:
        output["image_end_date"] = output["Date"]

    lookup: dict[str, pd.DataFrame] = {}
    for source_key, table in source_tables.items():
        required = {"sample_index", "split"}
        missing = required.difference(table.columns)
        if missing:
            raise KeyError(f"{source_key} context missing required columns: {sorted(missing)}")
        slim = table.copy()
        slim["sample_index"] = pd.to_numeric(slim["sample_index"], errors="raise").astype(int)
        lookup[source_key] = slim

    output["sample_index"] = pd.to_numeric(output["sample_index"], errors="raise").astype(int)
    for feature in selected_features:
        source = lookup[feature.source_key]
        raw_col = feature.source_feature
        norm_col = f"{feature.source_feature}_normalized"
        missing = [column for column in [raw_col, norm_col] if column not in source.columns]
        if missing:
            raise KeyError(
                f"{feature.source_key} context missing selected column(s) for "
                f"{feature.source_feature}: {missing}"
            )
        values = source[["sample_index", "split", raw_col, norm_col]].rename(
            columns={
                raw_col: feature.target_feature,
                norm_col: f"{feature.target_feature}_normalized",
            }
        )
        before_rows = len(output)
        output = output.merge(values, on=["sample_index", "split"], how="left", validate="one_to_one")
        if len(output) != before_rows:
            raise RuntimeError(f"Merge changed row count for {feature.target_feature}")
    normalized_columns = [f"{feature.target_feature}_normalized" for feature in selected_features]
    missing_rates = output[normalized_columns].isna().mean()
    bad = missing_rates[missing_rates.gt(0)]
    if not bad.empty:
        raise ValueError("Selected context has missing normalized values: " + bad.to_dict().__repr__())
    return output.sort_values(["split", "Date", "sample_index"]).reset_index(drop=True)


def _build_scaler_payload(
    selected_features: list[SelectedFeature],
    normalized_columns: list[str],
) -> dict[str, Any]:
    feature_order = [feature.target_feature for feature in selected_features]
    return {
        "feature_order": feature_order,
        "normalized_feature_columns": normalized_columns,
        "fit_on": "source_context_train_scalers",
        "passthrough_normalized_context": True,
        "source_feature_map": [
            {
                "target_feature": feature.target_feature,
                "target_normalized": f"{feature.target_feature}_normalized",
                "source_key": feature.source_key,
                "source_feature": feature.source_feature,
                "source_normalized": f"{feature.source_feature}_normalized",
                "rationale": feature.rationale,
            }
            for feature in selected_features
        ],
    }


def _build_audit(
    *,
    context_name: str,
    source_paths: dict[str, Path],
    selected_features: list[SelectedFeature],
    output: pd.DataFrame,
    feature_set_name: str,
) -> dict[str, Any]:
    normalized_columns = [f"{feature.target_feature}_normalized" for feature in selected_features]
    missing_by_split: dict[str, dict[str, float]] = {}
    for split_name, frame in output.groupby(output["split"].astype(str), sort=True):
        missing_by_split[split_name] = {
            column: float(pd.to_numeric(frame[column], errors="coerce").isna().mean())
            for column in normalized_columns
        }
    return {
        "status": "ok",
        "stage": "4-N13-5B",
        "context_name": context_name,
        "feature_set_name": feature_set_name,
        "context_dim": int(len(selected_features)),
        "feature_order": [feature.target_feature for feature in selected_features],
        "normalized_feature_columns": normalized_columns,
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "selected_features": [
            {
                "source_key": feature.source_key,
                "source_feature": feature.source_feature,
                "target_feature": feature.target_feature,
                "rationale": feature.rationale,
            }
            for feature in selected_features
        ],
        "num_rows": int(len(output)),
        "split_counts": output["split"].astype(str).value_counts().sort_index().to_dict(),
        "missing_rate_by_split": missing_by_split,
        "warnings": [],
    }


def _build_feature_summary(output: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in output.groupby(output["split"].astype(str), sort=True):
        for feature in feature_order:
            values = pd.to_numeric(frame[feature], errors="coerce")
            normalized = pd.to_numeric(frame[f"{feature}_normalized"], errors="coerce")
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "num_rows": int(len(frame)),
                    "raw_missing_rate": float(values.isna().mean()),
                    "raw_mean": float(values.mean()),
                    "raw_std": float(values.std(ddof=1)),
                    "raw_min": float(values.min()),
                    "raw_max": float(values.max()),
                    "normalized_mean": float(normalized.mean()),
                    "normalized_std": float(normalized.std(ddof=1)),
                    "normalized_min": float(normalized.min()),
                    "normalized_max": float(normalized.max()),
                }
            )
    return pd.DataFrame(rows)


def make_selected_combo_context_name(
    *,
    output_prefix: str,
    image_window: int,
    image_spec: str,
    return_horizon: int,
    feature_set_name: str,
) -> str:
    safe = str(feature_set_name).strip().replace(" ", "_").replace("/", "_")
    return f"{output_prefix}_i{int(image_window)}_{image_spec}_r{int(return_horizon)}_{safe}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
