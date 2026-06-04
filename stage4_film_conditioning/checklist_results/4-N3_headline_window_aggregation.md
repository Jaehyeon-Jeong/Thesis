# 4-N3. Headline-Window Aggregation

## English

Status: complete.

Purpose:
- Convert the selected BTC news source into leakage-safe headline-window
  documents for each Stage 4 BTC sample.
- Keep the first news track headline-only and non-LLM.
- Prepare full text windows for 4-N4 train-only TF-IDF/SVD.

Command:

```bash
python -u scripts/build_stage4_news_headline_windows.py \
  --config configs/env_local.yaml \
  --dataset-name edaschau/bitcoin_news \
  --dataset-filename BTC_match_title.csv \
  --split train \
  --window-days 7 20 60 \
  --output-prefix stage4_news_headline_windows
```

Locked policy:

```text
For image end date t and window W:
    use headlines from t-W through t-1 inclusive
    exclude same-day t headlines
    exclude future headlines
```

Why 7/20/60:
- `7d`: short news shock.
- `20d`: matches the R20 forecast horizon.
- `60d`: matches the I60 chart window and broader market-regime context.

Deduplication:

| Item | Value |
|:---|---:|
| Raw headline rows | 30,626 |
| Rows after empty-title filter | 30,626 |
| Dropped duplicate URL rows | 0 |
| Dropped duplicate normalized-title rows | 1,418 |
| Deduped rows | 29,208 |

Split-level headline-window summary:

| Split | Samples | 7d coverage | Median 7d count | 20d coverage | Median 20d count | 60d coverage | Median 60d count |
|:---|---:|---:|---:|---:|---:|---:|---:|
| train | 671 | 100.00% | 43 | 100.00% | 120 | 100.00% | 347 |
| validation | 287 | 100.00% | 44 | 100.00% | 124 | 100.00% | 372 |
| test | 1,441 | 100.00% | 79 | 100.00% | 228 | 100.00% | 689 |
| all | 2,399 | 100.00% | 64 | 100.00% | 192 | 100.00% | 591 |

Generated full tables:
- `outputs/stage4/news/stage4_news_headline_windows_i60_r20/daily_headlines.parquet`
- `outputs/stage4/news/stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet`

Report tables:
- `reports/tables/stage4_news_headline_windows_manifest.json`
- `reports/tables/stage4_news_headline_windows_summary.csv`
- `reports/tables/stage4_news_headline_windows_examples.csv`
- `reports/tables/stage4_news_headline_windows_daily_coverage.csv`
- `reports/tables/stage4_news_headline_windows_dedupe_audit.csv`

Decision:
- Proceed to `4-N4`.
- Use the sample-window parquet as the input for train-only TF-IDF/SVD.
- First vectorization target:

```text
headline_text_7d  -> news_svd_7d
headline_text_20d -> news_svd_20d
headline_text_60d -> news_svd_60d
```

## 한국어

상태: 완료.

목적:
- 선택한 BTC news source를 Stage 4 BTC sample별 leakage-safe headline-window
  document로 변환합니다.
- 첫 news track은 headline-only, non-LLM으로 유지합니다.
- 4-N4 train-only TF-IDF/SVD가 사용할 full text window를 준비합니다.

실행 command:

```bash
python -u scripts/build_stage4_news_headline_windows.py \
  --config configs/env_local.yaml \
  --dataset-name edaschau/bitcoin_news \
  --dataset-filename BTC_match_title.csv \
  --split train \
  --window-days 7 20 60 \
  --output-prefix stage4_news_headline_windows
```

고정 policy:

```text
Image end date가 t이고 window가 W이면:
    t-W부터 t-1까지의 headline만 사용
    same-day t headline 제외
    future headline 제외
```

7/20/60을 쓰는 이유:
- `7d`: 단기 뉴스 충격.
- `20d`: R20 예측 horizon과 맞는 중기 context.
- `60d`: I60 chart window와 broader market-regime context.

Deduplication:

| 항목 | 값 |
|:---|---:|
| Raw headline rows | 30,626 |
| Empty-title filter 이후 rows | 30,626 |
| 제거한 duplicate URL rows | 0 |
| 제거한 duplicate normalized-title rows | 1,418 |
| Deduped rows | 29,208 |

Split-level headline-window summary:

| Split | Samples | 7d coverage | Median 7d count | 20d coverage | Median 20d count | 60d coverage | Median 60d count |
|:---|---:|---:|---:|---:|---:|---:|---:|
| train | 671 | 100.00% | 43 | 100.00% | 120 | 100.00% | 347 |
| validation | 287 | 100.00% | 44 | 100.00% | 124 | 100.00% | 372 |
| test | 1,441 | 100.00% | 79 | 100.00% | 228 | 100.00% | 689 |
| all | 2,399 | 100.00% | 64 | 100.00% | 192 | 100.00% | 591 |

생성한 full table:
- `outputs/stage4/news/stage4_news_headline_windows_i60_r20/daily_headlines.parquet`
- `outputs/stage4/news/stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet`

Report table:
- `reports/tables/stage4_news_headline_windows_manifest.json`
- `reports/tables/stage4_news_headline_windows_summary.csv`
- `reports/tables/stage4_news_headline_windows_examples.csv`
- `reports/tables/stage4_news_headline_windows_daily_coverage.csv`
- `reports/tables/stage4_news_headline_windows_dedupe_audit.csv`

판단:
- `4-N4`로 진행합니다.
- Sample-window parquet을 train-only TF-IDF/SVD의 input으로 사용합니다.
- 첫 vectorization target:

```text
headline_text_7d  -> news_svd_7d
headline_text_20d -> news_svd_20d
headline_text_60d -> news_svd_60d
```
