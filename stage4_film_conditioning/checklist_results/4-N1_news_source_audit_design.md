# 4-N1. News Source Audit

## English

Status: complete.

Purpose:
- Verify that the Stage 4 news source covers the selected BTC sample period.
- Choose the exact file for the first headline-only, non-LLM news-context track.
- Confirm that strict `t-1` alignment is feasible before vector construction.

Command:

```bash
python -u scripts/audit_stage4_news_source.py \
  --config configs/env_local.yaml \
  --dataset-name edaschau/bitcoin_news \
  --dataset-filename BTC_match_title.csv \
  --split train \
  --output-prefix stage4_news
```

Candidate dataset:
- Hugging Face dataset: `edaschau/bitcoin_news`
- Selected file for first model: `BTC_match_title.csv`
- Reason: it contains headline-level BTC matches and covers the full selected
  Stage 4 sample period. `BTC_yahoo.csv` was checked but ends on `2024-01-24`,
  so it does not cover the full Stage 4 test range.

Audit result:

| Item | Value |
|:---|:---|
| Selected news file | `BTC_match_title.csv` |
| News rows | `30,626` |
| News date range | `2011-06-22` to `2025-06-03` |
| Stage 4 sample range | `2018-04-29` to `2024-12-11` |
| Stage 4 sample count | `2,399` |
| Split counts | train `671`, validation `287`, test `1,441` |
| Title coverage | `100.00%` |
| Article-text coverage | `100.00%` |
| Number of sources | `219` |
| Duplicate URL rate | `0.00%` |
| Duplicate normalized-title rate | `4.63%` |

Coverage under strict `t-1`:

| Split | Samples | `t-1` coverage | 7-day coverage | Median `t-1` news count | Median 7-day news count |
|:---|---:|---:|---:|---:|---:|
| train | 671 | 96.57% | 100.00% | 6 | 44 |
| validation | 287 | 97.21% | 100.00% | 7 | 45 |
| test | 1,441 | 95.56% | 100.00% | 12 | 81 |
| all | 2,399 | 96.04% | 100.00% | 9 | 67 |

Decision:
- Proceed to `4-N2`.
- Use `BTC_match_title.csv` as the first news source.
- Keep the first policy strict: for a chart image ending at date `t`, use only
  news with calendar date `<= t-1`.
- Same-day news remains excluded until BTC close cutoff and news timestamp
  cutoff are explicitly defended.
- The first vector track is updated in 4-N2 to 7/20/60-day headline windows:
  `news_svd_7d + news_svd_20d + news_svd_60d + news_count_7d/20d/60d`.

Output files:
- `reports/tables/stage4_news_source_audit.json`
- `reports/tables/stage4_news_daily_coverage.csv`
- `reports/tables/stage4_news_source_distribution.csv`
- `reports/tables/stage4_news_duplicate_audit.csv`
- `reports/tables/stage4_news_sample_coverage_by_split.csv`

## 한국어

상태: 완료.

목적:
- Stage 4 news source가 selected BTC sample period를 덮는지 확인합니다.
- 첫 headline-only, non-LLM news-context track에 사용할 정확한 파일을 정합니다.
- Vector를 만들기 전에 strict `t-1` alignment가 가능한지 확인합니다.

실행 command:

```bash
python -u scripts/audit_stage4_news_source.py \
  --config configs/env_local.yaml \
  --dataset-name edaschau/bitcoin_news \
  --dataset-filename BTC_match_title.csv \
  --split train \
  --output-prefix stage4_news
```

후보 dataset:
- Hugging Face dataset: `edaschau/bitcoin_news`
- 첫 모델에 사용할 파일: `BTC_match_title.csv`
- 이유: headline-level BTC match를 담고 있고 selected Stage 4 sample period 전체를
  덮습니다. `BTC_yahoo.csv`도 확인했지만 `2024-01-24`에서 끝나므로 Stage 4 test
  range 전체를 덮지 못합니다.

Audit 결과:

| 항목 | 값 |
|:---|:---|
| 선택한 news file | `BTC_match_title.csv` |
| News rows | `30,626` |
| News date range | `2011-06-22` to `2025-06-03` |
| Stage 4 sample range | `2018-04-29` to `2024-12-11` |
| Stage 4 sample count | `2,399` |
| Split counts | train `671`, validation `287`, test `1,441` |
| Title coverage | `100.00%` |
| Article-text coverage | `100.00%` |
| Source 수 | `219` |
| Duplicate URL rate | `0.00%` |
| Duplicate normalized-title rate | `4.63%` |

Strict `t-1` 기준 coverage:

| Split | Samples | `t-1` coverage | 7-day coverage | Median `t-1` news count | Median 7-day news count |
|:---|---:|---:|---:|---:|---:|
| train | 671 | 96.57% | 100.00% | 6 | 44 |
| validation | 287 | 97.21% | 100.00% | 7 | 45 |
| test | 1,441 | 95.56% | 100.00% | 12 | 81 |
| all | 2,399 | 96.04% | 100.00% | 9 | 67 |

판단:
- `4-N2`로 진행합니다.
- 첫 news source는 `BTC_match_title.csv`로 고정합니다.
- 첫 정책은 strict `t-1`입니다. Chart image end date가 `t`이면 calendar date
  `<= t-1`인 뉴스만 사용합니다.
- BTC close cutoff와 news timestamp cutoff를 명확히 방어하기 전까지 same-day
  news는 제외합니다.
- 첫 vector track은 4-N2에서 7/20/60-day headline window로 업데이트했습니다:
  `news_svd_7d + news_svd_20d + news_svd_60d + news_count_7d/20d/60d`.

Output files:
- `reports/tables/stage4_news_source_audit.json`
- `reports/tables/stage4_news_daily_coverage.csv`
- `reports/tables/stage4_news_source_distribution.csv`
- `reports/tables/stage4_news_duplicate_audit.csv`
- `reports/tables/stage4_news_sample_coverage_by_split.csv`
