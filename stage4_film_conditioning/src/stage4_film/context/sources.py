"""Stage 4 context source loading and audit helpers.

역할:
    Stage 4 primary context에서 외부 데이터가 필요한 것은 F&G뿐이다. 이 파일은
    local/Kaggle config에서 F&G CSV를 찾고, leakage-safe as-of merge가 가능한지
    빠르게 감사한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from stage4_film.config import get_context_config, get_config_section
from stage4_film.paths import Stage4Paths


def find_fear_greed_source(config: Mapping[str, Any], paths: Stage4Paths) -> Path:
    """config/path 기준으로 Fear & Greed CSV를 찾는다.

    우선순위:
        1. `paths.fear_greed_file`
        2. `context.fear_greed.local_file`
        3. `context.fear_greed.kaggle_file`
        4. `paths.data_root` 아래에서 `source_filename` 재귀 검색
    """

    context_config = get_context_config(config)
    fg_config = context_config.get("fear_greed", {})
    if not isinstance(fg_config, Mapping):
        raise TypeError("Config context.fear_greed must be a mapping.")

    candidates: list[Path] = []
    if paths.fear_greed_file is not None:
        candidates.append(paths.fear_greed_file)
    for key in ("local_file", "kaggle_file"):
        value = str(fg_config.get(key, "")).strip()
        if value:
            candidates.append(Path(value).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    source_filename = str(fg_config.get("source_filename", "fear_greed_index.csv"))
    matches = sorted(paths.data_root.rglob(source_filename))
    if matches:
        return matches[0]

    searched = ", ".join(str(item) for item in candidates) or str(paths.data_root)
    raise FileNotFoundError(
        f"Fear & Greed CSV not found. searched={searched}, filename={source_filename}"
    )


def load_fear_greed_index(
    source_file: str | Path,
    *,
    date_column: str = "date",
    value_column: str = "value",
    classification_column: str = "classification",
) -> pd.DataFrame:
    """F&G CSV를 표준 column으로 읽는다.

    출력 column:
        `fg_date`, `fg_value`, `fg_classification`, `fg_source_row`.
    """

    path = Path(source_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Fear & Greed source file does not exist: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Fear & Greed source file is empty: {path}")

    missing = [column for column in (date_column, value_column) if column not in raw.columns]
    if missing:
        raise KeyError(f"Fear & Greed CSV missing required column(s): {', '.join(missing)}")

    frame = pd.DataFrame(
        {
            "fg_date": pd.to_datetime(raw[date_column], errors="coerce").dt.normalize(),
            "fg_value": pd.to_numeric(raw[value_column], errors="coerce"),
            "fg_source_row": raw.index.astype(int),
        }
    )
    if classification_column in raw.columns:
        frame["fg_classification"] = raw[classification_column].astype("string")
    else:
        frame["fg_classification"] = pd.Series([pd.NA] * len(raw), dtype="string")

    before = len(frame)
    frame = frame.dropna(subset=["fg_date", "fg_value"]).copy()
    dropped = before - len(frame)
    if dropped:
        frame.attrs["dropped_missing_rows"] = int(dropped)

    frame = frame.sort_values("fg_date")
    frame = frame.drop_duplicates(subset="fg_date", keep="last")
    frame = frame.reset_index(drop=True)
    return frame


def audit_context_sources(
    *,
    ohlcv: pd.DataFrame,
    samples_by_split: Mapping[str, pd.DataFrame],
    fear_greed: pd.DataFrame,
    context_window: int,
    btc_source_file: str | Path,
    fear_greed_source_file: str | Path,
) -> dict[str, Any]:
    """BTC/F&G/context split coverage audit dictionary를 만든다."""

    all_samples = pd.concat(
        [frame.assign(split=name) for name, frame in samples_by_split.items()],
        ignore_index=True,
    )
    sample_dates = pd.to_datetime(all_samples["Date"]).dt.normalize()
    fg_dates = pd.DatetimeIndex(pd.to_datetime(fear_greed["fg_date"]).dt.normalize())
    fg_min = fg_dates.min()
    fg_max = fg_dates.max()
    full_fg_range = pd.date_range(fg_min, fg_max, freq="D")
    missing_fg_dates = full_fg_range.difference(fg_dates)

    fg_for_asof = fear_greed.loc[:, ["fg_date", "fg_value"]].sort_values("fg_date")
    sample_dates_frame = pd.DataFrame({"Date": sample_dates.sort_values().unique()})
    asof = pd.merge_asof(
        sample_dates_frame.sort_values("Date"),
        fg_for_asof,
        left_on="Date",
        right_on="fg_date",
        direction="backward",
    )
    asof["fg_age_days"] = (asof["Date"] - asof["fg_date"]).dt.days

    split_coverage: dict[str, Any] = {}
    for split_name, frame in samples_by_split.items():
        dates = pd.to_datetime(frame["Date"]).dt.normalize()
        split_dates_frame = pd.DataFrame({"Date": dates.sort_values().unique()})
        split_asof = pd.merge_asof(
            split_dates_frame.sort_values("Date"),
            fg_for_asof,
            left_on="Date",
            right_on="fg_date",
            direction="backward",
        )
        split_asof["fg_age_days"] = (split_asof["Date"] - split_asof["fg_date"]).dt.days
        missing_count = int(split_asof["fg_value"].isna().sum())
        exact_count = int((split_asof["Date"] == split_asof["fg_date"]).sum())
        split_coverage[split_name] = {
            "num_rows": int(len(frame)),
            "date_min": _date_or_none(dates.min()),
            "date_max": _date_or_none(dates.max()),
            "unique_dates": int(dates.nunique()),
            "fg_asof_missing_dates": missing_count,
            "fg_exact_date_matches": exact_count,
            "fg_max_age_days": _safe_int(split_asof["fg_age_days"].max()),
            "fg_mean_age_days": _safe_float(split_asof["fg_age_days"].mean()),
        }

    return {
        "btc": {
            "source_file": str(btc_source_file),
            "num_rows": int(len(ohlcv)),
            "date_min": _date_or_none(pd.to_datetime(ohlcv["Date"]).min()),
            "date_max": _date_or_none(pd.to_datetime(ohlcv["Date"]).max()),
        },
        "fear_greed": {
            "source_file": str(fear_greed_source_file),
            "num_rows": int(len(fear_greed)),
            "date_min": _date_or_none(fg_min),
            "date_max": _date_or_none(fg_max),
            "duplicate_dates_after_cleaning": int(fear_greed["fg_date"].duplicated().sum()),
            "missing_days_inside_range": int(len(missing_fg_dates)),
            "missing_dates_first_20": [
                item.date().isoformat() for item in missing_fg_dates[:20]
            ],
            "value_min": _safe_float(fear_greed["fg_value"].min()),
            "value_max": _safe_float(fear_greed["fg_value"].max()),
            "classification_counts": {
                str(key): int(value)
                for key, value in fear_greed["fg_classification"]
                .value_counts(dropna=False)
                .items()
            },
        },
        "context_policy": {
            "context_window": int(context_window),
            "asof_rule": "latest Fear & Greed date <= image end date t",
            "rolling_rule": "trailing calendar windows only; no future backfill",
        },
        "sample_coverage": {
            "num_samples": int(len(all_samples)),
            "date_min": _date_or_none(sample_dates.min()),
            "date_max": _date_or_none(sample_dates.max()),
            "fg_asof_missing_dates": int(asof["fg_value"].isna().sum()),
            "fg_max_age_days": _safe_int(asof["fg_age_days"].max()),
            "fg_mean_age_days": _safe_float(asof["fg_age_days"].mean()),
            "by_split": split_coverage,
        },
    }


def _date_or_none(value: Any) -> str | None:
    """Timestamp/date 값을 ISO date string으로 변환한다."""

    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _safe_float(value: Any) -> float | None:
    """NaN이면 None, 아니면 float."""

    if pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    """NaN이면 None, 아니면 int."""

    if pd.isna(value):
        return None
    return int(value)
