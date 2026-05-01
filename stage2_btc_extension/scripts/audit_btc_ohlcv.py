"""Stage 2 BTC OHLCV 데이터 audit script.

목적:
    Kaggle에 attach한 BTC OHLCV CSV를 열어서 Stage 2 image pipeline에 쓸 수
    있는지 먼저 확인한다.

확인 항목:
    - 정확한 CSV path와 raw column
    - OHLCV canonical column 매핑
    - timestamp parsing 가능 여부
    - row 수, 날짜 범위, daily frequency 여부
    - 결측/중복/비정상 OHLCV row
    - I5/I20/I60, R5/R20/R60 조합별 만들 수 있는 sample 수와 positive rate

주의:
    이 script는 raw data를 GitHub에 저장하지 않는다. 작은 JSON/Markdown audit
    report와 head sample CSV만 저장한다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREFERRED_FILENAME = "btc_1d_data_2018_to_2025.csv"
CANONICAL_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
WINDOWS = (5, 20, 60)
HORIZONS = (5, 20, 60)


def main() -> None:
    """CLI entry point.

    이 함수는 command-line argument를 읽고, BTC CSV를 찾은 뒤, audit report를
    output directory에 저장한다.
    """

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_file, candidates = resolve_source_file(
        data_root=Path(args.data_root).expanduser(),
        source_file=Path(args.source_file).expanduser() if args.source_file else None,
        preferred_filename=args.preferred_filename,
    )

    if source_file is None:
        result = {
            "status": "not_found",
            "message": "No BTC OHLCV CSV was found in the requested data root.",
            "data_root": str(Path(args.data_root).expanduser()),
            "preferred_filename": args.preferred_filename,
            "candidate_files": [str(path) for path in candidates],
        }
        write_outputs(result=result, cleaned_head=None, output_dir=output_dir, prefix=args.prefix)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    raw_frame = pd.read_csv(source_file)
    canonical_frame, column_map, parse_warnings = build_canonical_ohlcv_frame(raw_frame)
    result = audit_canonical_frame(
        raw_frame=raw_frame,
        canonical_frame=canonical_frame,
        source_file=source_file,
        candidate_files=candidates,
        column_map=column_map,
        parse_warnings=parse_warnings,
    )
    cleaned_head = canonical_frame.loc[:, list(CANONICAL_COLUMNS)].head(args.head_rows)
    write_outputs(result=result, cleaned_head=cleaned_head, output_dir=output_dir, prefix=args.prefix)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))


def parse_args() -> argparse.Namespace:
    """argument를 정의한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/kaggle/input",
        help="CSV를 찾을 root folder. Kaggle 기본값은 /kaggle/input.",
    )
    parser.add_argument(
        "--source-file",
        default="",
        help="정확한 CSV path를 알고 있으면 직접 지정한다.",
    )
    parser.add_argument(
        "--preferred-filename",
        default=PREFERRED_FILENAME,
        help="자동 탐색에서 가장 먼저 찾을 파일명.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/data_audit",
        help="audit JSON/Markdown/head CSV를 저장할 folder.",
    )
    parser.add_argument(
        "--prefix",
        default="btc_ohlcv",
        help="저장 파일 prefix.",
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=10,
        help="head sample CSV에 저장할 row 수.",
    )
    return parser.parse_args()


def resolve_source_file(
    data_root: Path,
    source_file: Path | None,
    preferred_filename: str,
) -> tuple[Path | None, list[Path]]:
    """BTC CSV 후보를 찾고 audit 대상 파일을 고른다.

    우선순위:
        1. `--source-file`이 존재하면 그것을 사용한다.
        2. `preferred_filename`과 정확히 같은 파일을 `data_root` 아래에서 찾는다.
        3. 이름에 btc/bitcoin과 1d/daily가 들어간 CSV를 점수화해서 고른다.

    반환:
        선택된 source file 또는 None, 그리고 사람이 확인할 candidate list.
    """

    if source_file is not None and source_file.exists():
        return source_file, [source_file]

    if not data_root.exists():
        return None, []

    all_csvs = sorted(data_root.rglob("*.csv"))
    exact_matches = [path for path in all_csvs if path.name == preferred_filename]
    if exact_matches:
        return exact_matches[0], exact_matches

    scored: list[tuple[int, Path]] = []
    for path in all_csvs:
        name = path.name.lower()
        score = 0
        if "btc" in name or "bitcoin" in name:
            score += 10
        if "1d" in name or "daily" in name or "day" in name:
            score += 5
        if "ohlcv" in name:
            score += 3
        if score > 0:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    candidates = [path for _score, path in scored]
    return (candidates[0] if candidates else None), candidates


def build_canonical_ohlcv_frame(
    raw_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str | None], list[str]]:
    """raw CSV column을 Stage 2에서 쓸 canonical OHLCV column으로 정리한다.

    출력 DataFrame은 최소한 `Date`, `Open`, `High`, `Low`, `Close`, `Volume`을
    가진다. 원본 column은 보존하지 않고 audit에 필요한 값만 canonical form으로
    만든다.
    """

    warnings: list[str] = []
    column_map = infer_column_map(raw_frame.columns)

    missing = [canonical for canonical, source in column_map.items() if source is None]
    if missing:
        warnings.append(f"Missing canonical column mapping: {', '.join(missing)}")

    canonical = pd.DataFrame(index=raw_frame.index)
    date_source = column_map.get("Date")
    if date_source is not None:
        canonical["Date"] = parse_datetime_series(raw_frame[date_source])
    else:
        canonical["Date"] = pd.NaT

    for canonical_name in ("Open", "High", "Low", "Close", "Volume"):
        source_name = column_map.get(canonical_name)
        if source_name is None:
            canonical[canonical_name] = np.nan
        else:
            canonical[canonical_name] = pd.to_numeric(raw_frame[source_name], errors="coerce")

    return canonical, column_map, warnings


def infer_column_map(columns: pd.Index) -> dict[str, str | None]:
    """raw column name을 canonical column name에 매핑한다."""

    normalized = {normalize_column_name(column): str(column) for column in columns}

    # Binance daily CSV는 보통 `Open time`이 시작 시점이고, `Close time`이 종료 시점이다.
    # Image window 종료 시점은 candle close 기준으로 해석할 수 있지만, row alignment는
    # 하루 candle의 시작 날짜와 종료 날짜가 같은 calendar day이므로 `Open time`을
    # 기본 Date로 둔다.
    date_candidates = (
        "open time",
        "opentime",
        "date",
        "datetime",
        "timestamp",
        "time",
    )
    column_map: dict[str, str | None] = {
        "Date": first_existing(normalized, date_candidates),
        "Open": first_existing(normalized, ("open",)),
        "High": first_existing(normalized, ("high",)),
        "Low": first_existing(normalized, ("low",)),
        "Close": first_existing(normalized, ("close",)),
        "Volume": first_existing(normalized, ("volume", "base asset volume")),
    }
    return column_map


def normalize_column_name(column: Any) -> str:
    """column name 비교를 쉽게 하도록 공백과 대소문자를 정리한다."""

    return " ".join(str(column).strip().lower().replace("_", " ").split())


def first_existing(normalized_columns: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    """후보 이름 중 raw DataFrame에 존재하는 첫 column을 반환한다."""

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """문자열 또는 unix timestamp column을 tz-naive pandas datetime으로 바꾼다."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_not_null = numeric.dropna()
    if len(numeric_not_null) == len(series.dropna()) and not numeric_not_null.empty:
        median_abs = float(numeric_not_null.abs().median())
        if median_abs > 1e17:
            unit = "ns"
        elif median_abs > 1e14:
            unit = "us"
        elif median_abs > 1e11:
            unit = "ms"
        else:
            unit = "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        cleaned = series.astype(str).str.replace(" UTC", "", regex=False)
        parsed = pd.to_datetime(cleaned, utc=True, errors="coerce")

    # Kaggle/Stage code에서는 timezone-naive datetime을 쓰는 편이 비교와 split에 안전하다.
    return parsed.dt.tz_convert(None)


def audit_canonical_frame(
    raw_frame: pd.DataFrame,
    canonical_frame: pd.DataFrame,
    source_file: Path,
    candidate_files: list[Path],
    column_map: dict[str, str | None],
    parse_warnings: list[str],
) -> dict[str, Any]:
    """canonical OHLCV DataFrame에 대한 audit dictionary를 만든다."""

    sorted_frame = canonical_frame.sort_values("Date").reset_index(drop=True)
    valid_ohlcv_mask = build_valid_ohlcv_mask(sorted_frame)
    cleaned_frame = sorted_frame[valid_ohlcv_mask].drop_duplicates(subset=["Date"], keep="first")
    cleaned_frame = cleaned_frame.sort_values("Date").reset_index(drop=True)

    date_series = cleaned_frame["Date"].dropna()
    date_diffs = date_series.sort_values().diff().dropna()

    result = {
        "status": "ok",
        "source_file": str(source_file),
        "candidate_files": [str(path) for path in candidate_files],
        "raw_shape": [int(raw_frame.shape[0]), int(raw_frame.shape[1])],
        "raw_columns": [str(column) for column in raw_frame.columns],
        "canonical_column_map": column_map,
        "parse_warnings": parse_warnings,
        "missing_values": {
            column: int(canonical_frame[column].isna().sum()) for column in CANONICAL_COLUMNS
        },
        "duplicate_dates": int(canonical_frame["Date"].duplicated().sum()),
        "invalid_ohlcv_rows": int((~valid_ohlcv_mask).sum()),
        "clean_row_count": int(len(cleaned_frame)),
        "date_min": to_iso_or_none(date_series.min() if not date_series.empty else None),
        "date_max": to_iso_or_none(date_series.max() if not date_series.empty else None),
        "frequency": summarize_frequency(date_diffs),
        "calendar_gap_check": summarize_calendar_gaps(date_series),
        "value_checks": summarize_value_checks(cleaned_frame),
        "window_horizon_availability": summarize_window_horizon_availability(cleaned_frame),
    }
    return result


def build_valid_ohlcv_mask(frame: pd.DataFrame) -> pd.Series:
    """Stage 2 image 생성에 사용할 수 있는 OHLCV row인지 판단한다."""

    mask = frame.loc[:, list(CANONICAL_COLUMNS)].notna().all(axis=1)
    mask &= frame["Open"].gt(0)
    mask &= frame["High"].gt(0)
    mask &= frame["Low"].gt(0)
    mask &= frame["Close"].gt(0)
    mask &= frame["Volume"].ge(0)
    mask &= frame["High"].ge(frame[["Open", "Close", "Low"]].max(axis=1))
    mask &= frame["Low"].le(frame[["Open", "Close", "High"]].min(axis=1))
    return mask


def summarize_frequency(date_diffs: pd.Series) -> dict[str, Any]:
    """날짜 차이를 요약해서 daily data인지 확인한다."""

    if date_diffs.empty:
        return {
            "median_delta": None,
            "is_mostly_daily": False,
            "top_delta_counts": {},
        }

    seconds = date_diffs.dt.total_seconds().astype(int)
    counts = Counter(seconds.tolist())
    top_counts = {
        seconds_to_readable(delta_seconds): int(count)
        for delta_seconds, count in counts.most_common(5)
    }
    daily_count = counts.get(86_400, 0)
    total = int(seconds.size)
    return {
        "median_delta": seconds_to_readable(int(seconds.median())),
        "daily_delta_share": float(daily_count / total),
        "is_mostly_daily": bool(total > 0 and daily_count / total >= 0.95),
        "top_delta_counts": top_counts,
    }


def summarize_calendar_gaps(date_series: pd.Series) -> dict[str, Any]:
    """BTC는 7일 거래되므로 calendar day gap이 있는지 확인한다."""

    if date_series.empty:
        return {
            "expected_calendar_days": 0,
            "unique_dates": 0,
            "missing_calendar_days": None,
        }

    dates = pd.to_datetime(date_series.dt.date)
    unique_dates = pd.Series(dates.unique()).sort_values()
    expected_days = int((unique_dates.max() - unique_dates.min()).days + 1)
    missing_days = int(expected_days - len(unique_dates))
    return {
        "expected_calendar_days": expected_days,
        "unique_dates": int(len(unique_dates)),
        "missing_calendar_days": missing_days,
    }


def summarize_value_checks(frame: pd.DataFrame) -> dict[str, Any]:
    """OHLCV 값의 기본 범위를 요약한다."""

    if frame.empty:
        return {}
    return {
        "open_min": float(frame["Open"].min()),
        "open_max": float(frame["Open"].max()),
        "close_min": float(frame["Close"].min()),
        "close_max": float(frame["Close"].max()),
        "volume_min": float(frame["Volume"].min()),
        "volume_max": float(frame["Volume"].max()),
        "zero_volume_rows": int(frame["Volume"].eq(0).sum()),
    }


def summarize_window_horizon_availability(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """I-window/R-horizon 조합별 sample 수와 positive rate를 계산한다.

    index `t`를 image 종료일로 두면, image는 `[t-window+1, t]`만 사용하고
    label은 `Close[t+horizon] / Close[t] - 1`로 만든다.
    """

    rows: list[dict[str, Any]] = []
    close = frame["Close"].astype(float).reset_index(drop=True)
    dates = frame["Date"].reset_index(drop=True)
    n_rows = int(len(frame))
    for window in WINDOWS:
        for horizon in HORIZONS:
            sample_count = max(0, n_rows - window - horizon + 1)
            if sample_count <= 0:
                rows.append(
                    {
                        "image_window": f"I{window}",
                        "return_horizon": f"R{horizon}",
                        "sample_count": 0,
                        "positive_rate": None,
                        "first_image_end_date": None,
                        "last_image_end_date": None,
                    }
                )
                continue

            end_start = window - 1
            end_stop_exclusive = n_rows - horizon
            entry_close = close.iloc[end_start:end_stop_exclusive].to_numpy()
            exit_close = close.shift(-horizon).iloc[end_start:end_stop_exclusive].to_numpy()
            future_return = exit_close / entry_close - 1.0
            labels = future_return > 0
            rows.append(
                {
                    "image_window": f"I{window}",
                    "return_horizon": f"R{horizon}",
                    "sample_count": int(sample_count),
                    "positive_rate": float(labels.mean()),
                    "first_image_end_date": to_iso_or_none(dates.iloc[end_start]),
                    "last_image_end_date": to_iso_or_none(dates.iloc[end_stop_exclusive - 1]),
                }
            )
    return rows


def write_outputs(
    result: dict[str, Any],
    cleaned_head: pd.DataFrame | None,
    output_dir: Path,
    prefix: str,
) -> None:
    """JSON, Markdown, head CSV를 저장한다."""

    json_path = output_dir / f"{prefix}_audit.json"
    markdown_path = output_dir / f"{prefix}_audit.md"
    head_path = output_dir / f"{prefix}_head.csv"

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(result), encoding="utf-8")
    if cleaned_head is not None:
        cleaned_head.to_csv(head_path, index=False)


def render_markdown_report(result: dict[str, Any]) -> str:
    """audit result를 사람이 읽기 쉬운 Markdown으로 바꾼다."""

    lines = [
        "# BTC OHLCV Data Audit",
        "",
        "## English",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Source file: `{result.get('source_file')}`",
        f"- Raw shape: `{result.get('raw_shape')}`",
        f"- Date range: `{result.get('date_min')}` to `{result.get('date_max')}`",
        f"- Clean rows: `{result.get('clean_row_count')}`",
        f"- Duplicate dates: `{result.get('duplicate_dates')}`",
        f"- Invalid OHLCV rows: `{result.get('invalid_ohlcv_rows')}`",
        "",
        "Canonical column map:",
        "",
    ]
    for key, value in dict(result.get("canonical_column_map") or {}).items():
        lines.append(f"- `{key}` <- `{value}`")

    lines.extend(
        [
            "",
            "Frequency:",
            "",
            f"- Median delta: `{(result.get('frequency') or {}).get('median_delta')}`",
            f"- Mostly daily: `{(result.get('frequency') or {}).get('is_mostly_daily')}`",
            f"- Daily delta share: `{(result.get('frequency') or {}).get('daily_delta_share')}`",
            "",
            "Window/horizon availability:",
            "",
            "| Image | Horizon | Samples | Positive rate | First end date | Last end date |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in result.get("window_horizon_availability") or []:
        positive_rate = row.get("positive_rate")
        positive_text = "" if positive_rate is None else f"{positive_rate:.4f}"
        lines.append(
            f"| {row.get('image_window')} | {row.get('return_horizon')} | "
            f"{row.get('sample_count')} | {positive_text} | "
            f"{row.get('first_image_end_date')} | {row.get('last_image_end_date')} |"
        )

    lines.extend(
        [
            "",
            "## 한국어",
            "",
            f"- 상태: `{result.get('status')}`",
            f"- 원본 파일: `{result.get('source_file')}`",
            f"- raw shape: `{result.get('raw_shape')}`",
            f"- 날짜 범위: `{result.get('date_min')}`부터 `{result.get('date_max')}`까지",
            f"- cleaning 후 row 수: `{result.get('clean_row_count')}`",
            f"- duplicate date 수: `{result.get('duplicate_dates')}`",
            f"- 비정상 OHLCV row 수: `{result.get('invalid_ohlcv_rows')}`",
            "",
            "canonical column mapping:",
            "",
        ]
    )
    for key, value in dict(result.get("canonical_column_map") or {}).items():
        lines.append(f"- `{key}` <- `{value}`")
    lines.extend(
        [
            "",
            "frequency 확인:",
            "",
            f"- median delta: `{(result.get('frequency') or {}).get('median_delta')}`",
            f"- 대부분 daily인지: `{(result.get('frequency') or {}).get('is_mostly_daily')}`",
            f"- daily delta 비율: `{(result.get('frequency') or {}).get('daily_delta_share')}`",
            "",
            "window/horizon별 사용 가능 sample:",
            "",
            "| Image | Horizon | Samples | Positive rate | 첫 image 종료일 | 마지막 image 종료일 |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in result.get("window_horizon_availability") or []:
        positive_rate = row.get("positive_rate")
        positive_text = "" if positive_rate is None else f"{positive_rate:.4f}"
        lines.append(
            f"| {row.get('image_window')} | {row.get('return_horizon')} | "
            f"{row.get('sample_count')} | {positive_text} | "
            f"{row.get('first_image_end_date')} | {row.get('last_image_end_date')} |"
        )
    lines.append("")
    return "\n".join(lines)


def seconds_to_readable(delta_seconds: int) -> str:
    """초 단위 시간 차이를 읽기 쉬운 문자열로 바꾼다."""

    if delta_seconds % 86_400 == 0:
        return f"{delta_seconds // 86_400} days"
    if delta_seconds % 3_600 == 0:
        return f"{delta_seconds // 3_600} hours"
    if delta_seconds % 60 == 0:
        return f"{delta_seconds // 60} minutes"
    return f"{delta_seconds} seconds"


def to_iso_or_none(value: Any) -> str | None:
    """Timestamp-like 값을 ISO string으로 바꾼다."""

    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def json_default(value: Any) -> Any:
    """NumPy/pandas 값을 JSON으로 저장할 수 있게 변환한다."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    main()
