# 4-N2. News Publication-Time Alignment

## English

Status: complete.

Purpose:
- Lock a defensible no-future-leakage alignment rule before building news
  vectors.
- Verify sample-level news counts under strict `t-1`, 7-day, 20-day, and
  60-day windows.
- Define exactly which daily headline documents can be used to fit the text
  vectorizer.

Command:

```bash
python -u scripts/audit_stage4_news_alignment.py \
  --config configs/env_local.yaml \
  --dataset-name edaschau/bitcoin_news \
  --dataset-filename BTC_match_title.csv \
  --split train \
  --output-prefix stage4_news_alignment
```

Locked alignment policy:

```text
For a BTC chart image ending at calendar date t:
    allowed news date <= t - 1
    same-calendar-day news is excluded
    future news is excluded
```

Reason:
- The BTC daily candle close cutoff and news publication-time cutoff are not
  defended yet.
- Same-day headlines may be published after the model's decision point.
- Strict `t-1` is conservative, but it is easy to defend in the thesis.

Split-level result:

| Split | Samples | Strict `t-1` coverage | 7d/20d/60d coverage | Median 7d | Median 20d | Median 60d | Same-day excluded |
|:---|---:|---:|---:|---:|---:|---:|---:|
| train | 671 | 96.57% | 100.00% | 44 | 123 | 364 | 649 |
| validation | 287 | 97.21% | 100.00% | 45 | 130 | 380 | 278 |
| test | 1,441 | 95.56% | 100.00% | 81 | 236 | 717 | 1,377 |
| all | 2,399 | 96.04% | 100.00% | 67 | 198 | 605 | 2,304 |

Text vectorizer fit rule:
- Fit preprocessing, TF-IDF, IDF weights, and SVD only on train strict-`t-1`
  7/20/60-day headline-window documents.
- This gives `671` train samples x `3` windows = `2,013` train window documents.
- Validation/test headline-window documents are transform-only.
- Same-day and future news never enter the vectorizer fit step.

Implementation note:
- News `date_time` is parsed as UTC.
- It is then converted to a timezone-naive calendar date for strict daily
  alignment.
- This avoids accidentally mixing timezone-aware news timestamps with
  timezone-naive BTC daily dates.

Decision:
- Proceed to `4-N3`.
- Build headline-only daily aggregation from `BTC_match_title.csv`.
- First sample-window fields should be:
  - `headline_text_7d`, `headline_text_20d`, `headline_text_60d`;
  - `news_count_7d`, `news_count_20d`, `news_count_60d`;
  - optional source-count features per window.
- Rationale: 7 days captures short news shocks, 20 days matches the R20 forecast
  horizon, and 60 days matches the I60 chart window/regime.

Output files:
- `reports/tables/stage4_news_alignment_policy.json`
- `reports/tables/stage4_news_alignment_by_split.csv`
- `reports/tables/stage4_news_alignment_sample_counts.csv`
- `reports/tables/stage4_news_alignment_missing_t_minus_1_dates.csv`
- `reports/tables/stage4_news_alignment_examples.csv`

## 한국어

상태: 완료.

목적:
- News vector를 만들기 전에 future leakage가 없는 정렬 규칙을 고정합니다.
- Strict `t-1`, 7-day, 20-day, 60-day window에서 sample별 news count를
  확인합니다.
- Text vectorizer를 어떤 daily headline document에 fit할지 정확히 정의합니다.

실행 command:

```bash
python -u scripts/audit_stage4_news_alignment.py \
  --config configs/env_local.yaml \
  --dataset-name edaschau/bitcoin_news \
  --dataset-filename BTC_match_title.csv \
  --split train \
  --output-prefix stage4_news_alignment
```

고정한 alignment policy:

```text
BTC chart image end date가 calendar date t이면:
    allowed news date <= t - 1
    same-calendar-day news는 제외
    future news는 제외
```

이유:
- BTC daily candle close cutoff와 news publication-time cutoff를 아직 명확히
  방어하지 않았습니다.
- Same-day headline은 model decision point 이후에 나온 뉴스일 수 있습니다.
- Strict `t-1`은 보수적이지만 논문에서 방어하기 쉽습니다.

Split-level 결과:

| Split | Samples | Strict `t-1` coverage | 7d/20d/60d coverage | Median 7d | Median 20d | Median 60d | Same-day excluded |
|:---|---:|---:|---:|---:|---:|---:|---:|
| train | 671 | 96.57% | 100.00% | 44 | 123 | 364 | 649 |
| validation | 287 | 97.21% | 100.00% | 45 | 130 | 380 | 278 |
| test | 1,441 | 95.56% | 100.00% | 81 | 236 | 717 | 1,377 |
| all | 2,399 | 96.04% | 100.00% | 67 | 198 | 605 | 2,304 |

Text vectorizer fit rule:
- Text preprocessing, TF-IDF, IDF weight, SVD는 train strict-`t-1` 7/20/60-day
  headline-window document에만 fit합니다.
- 즉, `671` train samples x `3` windows = `2,013` train window documents입니다.
- Validation/test headline-window document는 transform-only입니다.
- Same-day와 future news는 vectorizer fit 단계에도 들어가지 않습니다.

구현 note:
- News `date_time`은 UTC로 parse합니다.
- 이후 strict daily alignment를 위해 timezone-naive calendar date로 변환합니다.
- 이렇게 하면 timezone-aware news timestamp와 timezone-naive BTC daily date를
  섞다가 생기는 오류를 피할 수 있습니다.

판단:
- `4-N3`로 진행합니다.
- `BTC_match_title.csv`에서 headline-only daily aggregation을 만듭니다.
- 첫 sample-window field:
  - `headline_text_7d`, `headline_text_20d`, `headline_text_60d`;
  - `news_count_7d`, `news_count_20d`, `news_count_60d`;
  - window별 optional source-count feature.
- 이유: 7일은 단기 뉴스 충격, 20일은 R20 예측 horizon context, 60일은 I60
  chart window/regime과 맞습니다.

Output files:
- `reports/tables/stage4_news_alignment_policy.json`
- `reports/tables/stage4_news_alignment_by_split.csv`
- `reports/tables/stage4_news_alignment_sample_counts.csv`
- `reports/tables/stage4_news_alignment_missing_t_minus_1_dates.csv`
- `reports/tables/stage4_news_alignment_examples.csv`
