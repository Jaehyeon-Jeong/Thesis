"""BTC OHLCV loading and validation.

역할:
    Stage 2의 raw input은 Kaggle BTC OHLCV CSV다. 이 파일은 CSV 경로를 찾고,
    column 이름을 Stage 2 표준 이름으로 맞춘 뒤, 날짜순 daily OHLCV DataFrame을
    만든다.

출력 DataFrame shape:
    `(num_days, 7)` 이상. 기본 column은
    `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `source_row`다.

Leakage 관련:
    이 module은 label을 만들지 않는다. 현재/과거/미래를 나누는 일은
    `label_split.py`에서 한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from stage2_btc.config import get_config_section
from stage2_btc.paths import Stage2Paths

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("Date", "Open", "High", "Low", "Close", "Volume")


def find_btc_ohlcv_source(config: Mapping[str, Any], paths: Stage2Paths) -> Path:
    """config/path 기준으로 BTC daily CSV를 찾는다.

    우선순위:
        1. `paths.source_file`
        2. `data.source_file`
        3. `paths.data_root` 아래에서 `data.source_filename` 재귀 검색

    Kaggle에서는 dataset attach 위치가 사용자의 dataset 이름에 따라 달라질 수 있어
    3번 auto-detect가 필요하다.
    """

    data_config = get_config_section(config, "data")

    candidates: list[Path] = []
    if paths.source_file is not None:
        candidates.append(paths.source_file)

    data_source = str(data_config.get("source_file", "")).strip()
    if data_source:
        candidates.append(Path(data_source).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    source_filename = str(data_config.get("source_filename", "btc_1d_data_2018_to_2025.csv"))
    matches = sorted(paths.data_root.rglob(source_filename))
    if matches:
        return matches[0]

    searched = ", ".join(str(item) for item in candidates) or str(paths.data_root)
    raise FileNotFoundError(
        f"BTC OHLCV CSV not found. searched={searched}, filename={source_filename}"
    )


def load_btc_ohlcv(source_file: str | Path) -> pd.DataFrame:
    """BTC OHLCV CSV를 읽고 Stage 2 표준 daily DataFrame으로 변환한다.

    입력:
        Kaggle/local BTC daily CSV path.

    출력:
        날짜순 정렬된 DataFrame. `Date`는 timezone 없는 pandas datetime이다.
    """

    path = Path(source_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"BTC OHLCV source file does not exist: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"BTC OHLCV source file is empty: {path}")

    frame = _canonicalize_columns(raw)
    frame = frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    frame["Date"] = _parse_datetime_series(frame["Date"])

    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before_drop = len(frame)
    frame = frame.dropna(subset=list(REQUIRED_OHLCV_COLUMNS)).copy()
    dropped = before_drop - len(frame)
    if dropped:
        frame.attrs["dropped_missing_rows"] = int(dropped)

    frame = frame.sort_values("Date").drop_duplicates(subset="Date", keep="first")
    frame = frame.reset_index(drop=True)
    frame["source_row"] = frame.index.astype(int)

    invalid = frame[
        (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1))
        | (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1))
        | (frame["Volume"] < 0)
    ]
    if not invalid.empty:
        raise ValueError(f"Invalid OHLCV rows found after cleaning: {len(invalid)}")

    return frame


def _canonicalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """다양한 Kaggle column 표기를 Stage 2 표준 이름으로 맞춘다."""

    rename_map: dict[str, str] = {}
    date_column_found = False
    for column in raw.columns:
        lower = str(column).strip().lower()
        if lower == "open":
            rename_map[column] = "Open"
        elif lower == "high":
            rename_map[column] = "High"
        elif lower == "low":
            rename_map[column] = "Low"
        elif lower == "close":
            rename_map[column] = "Close"
        elif lower == "volume":
            rename_map[column] = "Volume"
        elif not date_column_found and ("open time" in lower or lower in {"date", "time"}):
            rename_map[column] = "Date"
            date_column_found = True

    frame = raw.rename(columns=rename_map)
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"BTC OHLCV missing required column(s): {', '.join(missing)}")
    return frame


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    """UTC 문자열/숫자 timestamp를 timezone 없는 datetime으로 변환한다."""

    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="ms", utc=True)
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.tz_convert(None)
