# 4-3 News Dataset Audit and Feasibility Decision

## English

Status: complete for planning.

Checked on: 2026-05-25.

External source:
- Hugging Face dataset: `edaschau/bitcoin_news`
- Dataset page: `https://huggingface.co/datasets/edaschau/bitcoin_news`
- Dataset server metadata endpoint:
  `https://datasets-server.huggingface.co/info?dataset=edaschau/bitcoin_news`

## Metadata Observed

Hugging Face public metadata reports:
- Dataset subset/config: `default`
- Split: `train`
- Rows: `210,832`
- Format: CSV, auto-converted to Parquet in the viewer
- Language: English
- Date range shown in the viewer:
  `2011-06-22 10:56:00` to `2025-06-04 14:01:05`
- Main columns:
  - `time_unix`
  - `date_time`
  - `text_matches`
  - `title_matches`
  - `title`
  - `url`
  - `source`
  - `source_url`
  - `article_text`
- Source diversity shown in the viewer:
  - `source`: 364 values
  - `source_url`: 361 values
- Dataset server metadata includes repository revision-like paths under
  `b436ba3f8af941a3bc125b8cb1d7c5297fd01e13`.

The dataset overlaps the Stage 2 BTC test period:

```text
Stage 2 test: 2021-01-01 to 2024-12-31
News data:    2011-06-22 to 2025-06-04
```

Therefore the dataset is feasible as a BTC news-context source.

## Main Risk: Time Leakage

The main issue is not row count. The main issue is timestamp alignment.

For a chart image ending at date `t`, the target is future return after `t`.
Any news used as context must be known at or before the model decision time.
If we use same-calendar-day news without a precise cutoff, we may accidentally
include news published after the daily close. That would leak future
information into the model.

Decision:

```text
First news experiment uses strict t-1 policy.

For chart end date t:
    allowed_news_time <= end of calendar date t-1
```

This is slightly conservative, but it makes the first news experiment easier to
defend. Same-day news can only be tested later if the BTC candle timestamp and
news timestamp cutoff are explicitly defined.

## First News Version

Use headline-only first.

Do not use full `article_text` in the first news-context run because:
- article text is long and noisy;
- article length distribution is very wide;
- duplicate/syndicated articles are likely;
- full text increases compute and storage;
- LLM summarization introduces cache/version/prompt reproducibility issues.

Recommended first news document:

```text
daily_news_doc[d] =
    concatenated title values for date d
    after de-duplicating exact title/url duplicates
    with a fixed top-k or max-character limit
```

Recommended first context vector for the selected `I60/R20` baseline:

```text
news_context[t] =
    train-fit TF-IDF/SVD vector from titles up to t-1
    plus simple coverage features:
        news_count_1d[t-1]
        news_count_7d[t-7:t-1]
        news_count_60d[t-60:t-1]
```

If the first implementation needs to be even simpler, start with
`daily_news_doc[t-1]` only, then add trailing 7-day and 60-day summaries as an
ablation.

## Encoder Decision

Use non-LLM first:
- Fit text preprocessing, vocabulary, TF-IDF, and SVD on train-period news only.
- Transform validation/test with the train-fitted encoder.
- Save encoder config, vocabulary/hash, SVD dimension, and dataset revision.

Defer:
- article summaries;
- sentence-transformer embeddings;
- LLM summaries;
- LLM embeddings.

These are still thesis-relevant, but only after the no-leakage headline track is
stable and reproducible.

## Fusion Decision

News should use the same Stage 4 fusion comparison after the numeric context
track is stable:

| Track | News version |
| --- | --- |
| `4-N-A` | CNN + news-context concat |
| `4-N-B` | CNN + news-context gating |
| `4-N-C` | CNN + news-context FiLM gamma-only |
| `4-N-D` | CNN + news-context FiLM full gamma/beta |

This keeps the thesis logic clean:
- numeric context tests whether structured market state helps;
- news context tests whether textual market state helps;
- concat/gating/FiLM tests how the context should be attached.

## Required Code-Level Audit Later

Before a news model is trained, run a code-level audit that writes:
- row count and date range after parsing;
- rows per day over train/validation/test;
- missing-news days by split;
- duplicate URL/title rate;
- empty title/article counts;
- source distribution;
- article/title length distribution;
- exact no-leakage alignment examples for several chart dates;
- train-only encoder fit verification.

This belongs to the later `4-N` implementation track.

## Decision

Use `edaschau/bitcoin_news` as the planned second-phase news source, but do not
make it the first Stage 4 main run.

Stage 4 immediate priority remains:

```text
I60/R20/ohlc_ma_vb image
    + structured numeric context
    + concat / gating / gamma-only FiLM / full FiLM
```

After that is stable, add:

```text
headline-only BTC news context
    + strict t-1 alignment
    + train-fit non-LLM encoder
    + same four fusion heads
```

## 한국어

상태: 계획 단계 완료.

확인일: 2026-05-25.

외부 source:
- Hugging Face dataset: `edaschau/bitcoin_news`
- Dataset page: `https://huggingface.co/datasets/edaschau/bitcoin_news`
- Dataset server metadata endpoint:
  `https://datasets-server.huggingface.co/info?dataset=edaschau/bitcoin_news`

## 확인한 metadata

Hugging Face 공개 metadata 기준:
- Dataset subset/config: `default`
- Split: `train`
- Rows: `210,832`
- Format: CSV, viewer에서는 Parquet으로 자동 변환
- Language: English
- Viewer에 표시된 date range:
  `2011-06-22 10:56:00` to `2025-06-04 14:01:05`
- 주요 column:
  - `time_unix`
  - `date_time`
  - `text_matches`
  - `title_matches`
  - `title`
  - `url`
  - `source`
  - `source_url`
  - `article_text`
- Viewer에 표시된 source 다양성:
  - `source`: 364 values
  - `source_url`: 361 values
- Dataset server metadata에는
  `b436ba3f8af941a3bc125b8cb1d7c5297fd01e13` revision-like path가 포함됩니다.

이 dataset은 Stage 2 BTC test 기간과 겹칩니다.

```text
Stage 2 test: 2021-01-01 to 2024-12-31
News data:    2011-06-22 to 2025-06-04
```

따라서 BTC news-context source로 사용할 수 있습니다.

## 핵심 위험: time leakage

문제는 row 수가 아니라 timestamp alignment입니다.

chart image end date가 `t`이면 label은 `t` 이후의 미래 return입니다. Context로 쓰는
뉴스는 모델 의사결정 시점에 이미 알려진 뉴스여야 합니다. 같은 calendar date의 뉴스를
그냥 쓰면 daily close 이후에 나온 뉴스가 섞일 수 있고, 그러면 미래 정보 누수가 됩니다.

결정:

```text
첫 news experiment는 strict t-1 policy를 사용한다.

chart end date가 t일 때:
    allowed_news_time <= t-1 calendar date의 끝
```

이 방식은 약간 보수적이지만 첫 news experiment를 방어하기 쉽습니다. Same-day news는
BTC candle timestamp와 news cutoff를 명확히 정의한 뒤 나중에만 테스트합니다.

## 첫 news version

처음에는 headline-only를 사용합니다.

첫 news-context run에서 `article_text` 전체를 쓰지 않는 이유:
- article text가 길고 noise가 큽니다.
- article length 편차가 큽니다.
- duplicate/syndicated article 가능성이 큽니다.
- full text는 compute/storage 부담을 키웁니다.
- LLM summary는 cache/version/prompt 재현성 문제가 생깁니다.

추천 첫 news document:

```text
daily_news_doc[d] =
    date d의 title을 concat
    exact title/url duplicate 제거
    fixed top-k 또는 max-character limit 적용
```

선택된 `I60/R20` baseline 기준 추천 첫 context vector:

```text
news_context[t] =
    t-1까지의 title에서 만든 train-fit TF-IDF/SVD vector
    plus coverage features:
        news_count_1d[t-1]
        news_count_7d[t-7:t-1]
        news_count_60d[t-60:t-1]
```

구현을 더 단순하게 시작해야 하면 `daily_news_doc[t-1]`만 먼저 쓰고, 이후 7일/60일
summary를 ablation으로 추가합니다.

## Encoder 결정

먼저 non-LLM을 사용합니다.
- Text preprocessing, vocabulary, TF-IDF, SVD는 train-period news에만 fit합니다.
- Validation/test는 train-fitted encoder로 transform합니다.
- Encoder config, vocabulary/hash, SVD dimension, dataset revision을 저장합니다.

뒤로 미루는 것:
- article summaries;
- sentence-transformer embeddings;
- LLM summaries;
- LLM embeddings.

이 방향은 논문에 여전히 중요하지만, no-leakage headline track이 안정화된 뒤 진행합니다.

## Fusion 결정

뉴스도 numeric context track이 안정화된 뒤 같은 Stage 4 fusion 비교를 사용합니다.

| Track | News version |
| --- | --- |
| `4-N-A` | CNN + news-context concat |
| `4-N-B` | CNN + news-context gating |
| `4-N-C` | CNN + news-context FiLM gamma-only |
| `4-N-D` | CNN + news-context FiLM full gamma/beta |

이렇게 해야 논리 구조가 깔끔합니다.
- numeric context는 structured market state가 도움이 되는지 확인합니다.
- news context는 textual market state가 도움이 되는지 확인합니다.
- concat/gating/FiLM은 context를 어떻게 붙이는 것이 좋은지 확인합니다.

## 이후 필요한 code-level audit

뉴스 모델을 학습하기 전에는 다음을 저장하는 code-level audit이 필요합니다.
- parse 이후 row count와 date range;
- train/validation/test 기간별 rows per day;
- split별 missing-news days;
- duplicate URL/title rate;
- empty title/article counts;
- source distribution;
- article/title length distribution;
- 여러 chart date에 대한 no-leakage alignment 예시;
- train-only encoder fit 검증.

이 부분은 이후 `4-N` implementation track에서 진행합니다.

## 결정

`edaschau/bitcoin_news`는 planned second-phase news source로 사용합니다. 하지만 첫
Stage 4 main run으로 두지는 않습니다.

Stage 4의 즉시 우선순위는 그대로 다음입니다.

```text
I60/R20/ohlc_ma_vb image
    + structured numeric context
    + concat / gating / gamma-only FiLM / full FiLM
```

그 다음에 다음을 추가합니다.

```text
headline-only BTC news context
    + strict t-1 alignment
    + train-fit non-LLM encoder
    + same four fusion heads
```
