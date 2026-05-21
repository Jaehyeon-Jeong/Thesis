# Stage 4 News Context Feasibility Plan

## English

News context is useful for the thesis, but it should be added after the numeric
context fusion comparison is stable. This prevents the Stage 4 question from
mixing two hard problems at once: context-source quality and fusion-method
quality.

## Candidate Dataset

Candidate:
- Hugging Face: `edaschau/bitcoin_news`
- URL: `https://huggingface.co/datasets/edaschau/bitcoin_news`

Current public metadata checked on 2026-05-21:
- Split: `train`
- Rows: about `210,832`
- Date range in viewer: 2011-06-22 to 2025-06-04
- Format: CSV/Parquet
- Text language: English
- Important columns:
  - `time_unix`
  - `date_time`
  - `title`
  - `url`
  - `source`
  - `source_url`
  - `article_text`
  - `text_matches`
  - `title_matches`

The dataset appears usable for BTC news context because it overlaps the BTC
test period used in Stage 2 (`2021-01-01` to `2024-12-31`).

## Why News Is Not the First Main Experiment

News can help, but it has extra risks:
- publication time must be aligned carefully;
- late-day news may not be known before a chart close depending on the chosen
  timestamp convention;
- article text is large and noisy;
- many rows may mention crypto broadly rather than BTC-specific information;
- LLM summarization/embedding requires cache and version control.

For this reason, the first Stage 4 main experiment should use numeric context.
News can then be tested with the same four fusion methods:

```text
news context concat
news context gating
news context gamma-only FiLM
news context full FiLM
```

## Recommended News Pipeline

### 1. Audit

Check:
- row count;
- date range;
- rows per day;
- missing days in Stage 2 train/validation/test periods;
- duplicate URL/title rate;
- article length distribution;
- source distribution;
- title-only coverage versus article-text coverage.

### 2. Time Alignment

For each BTC image ending at date `t`, allowed news should be:

```text
news_time <= t close cutoff
```

Simpler first policy:
- use all news from calendar date `t-1` through date `t` before UTC daily close;
- if exact intraday cutoff is uncertain, use news up to `t-1` for a stricter
  no-leakage baseline.

The strict `t-1` policy is safer for the first news experiment.

### 3. Daily Aggregation Options

Start simple:
1. latest headline of the allowed day;
2. top-k headlines by recency;
3. concatenated headlines with max token limit;
4. daily bag-of-words/TF-IDF;
5. article summaries only after caching rules are stable.

### 4. Encoder Options

Non-LLM first:
- TF-IDF + SVD daily vector;
- trainable word embedding + GRU over concatenated headlines;
- fixed sentence embedding only if model version and cache are recorded.

LLM later:
- daily summary text;
- daily embedding vector;
- must record model name, prompt, revision/version, cache hash, and runtime.

### 5. Fusion Experiments

Once a daily news vector exists, use the same four heads:

| Track | News version |
|:---|:---|
| 4-N-A | CNN + news-context concat |
| 4-N-B | CNN + news-context gating |
| 4-N-C | CNN + news-context FiLM gamma-only |
| 4-N-D | CNN + news-context FiLM full |

## Recommendation

Do not remove news from Stage 4. Treat it as the second context source after the
structured numeric context. This keeps the immediate experiment feasible while
preserving the advisor-facing News/LLM direction.

## 한국어

뉴스 context는 논문에 유용할 수 있지만, numeric context fusion 비교가 안정화된 뒤에
넣는 것이 좋습니다. 그래야 Stage 4가 context-source quality와 fusion-method quality를
동시에 섞어서 어려워지는 것을 막을 수 있습니다.

## 후보 데이터셋

후보:
- Hugging Face: `edaschau/bitcoin_news`
- URL: `https://huggingface.co/datasets/edaschau/bitcoin_news`

2026-05-21 기준 공개 metadata 확인:
- Split: `train`
- Rows: 약 `210,832`
- Viewer 기준 date range: 2011-06-22 to 2025-06-04
- Format: CSV/Parquet
- Text language: English
- 주요 column:
  - `time_unix`
  - `date_time`
  - `title`
  - `url`
  - `source`
  - `source_url`
  - `article_text`
  - `text_matches`
  - `title_matches`

이 dataset은 Stage 2 BTC test period인 `2021-01-01` to `2024-12-31`과 겹치므로
BTC news context 후보로 사용할 수 있습니다.

## 뉴스가 첫 main experiment가 아닌 이유

뉴스는 도움이 될 수 있지만 추가 위험이 있습니다.
- publication time을 정확히 정렬해야 합니다.
- 하루 중 늦게 나온 뉴스는 chart close 시점 전에 알 수 있었는지 애매할 수 있습니다.
- article text가 크고 noise가 많습니다.
- crypto 전체 뉴스가 BTC-specific signal이 아닐 수 있습니다.
- LLM summary/embedding은 cache와 version control이 필요합니다.

그래서 첫 Stage 4 main experiment는 numeric context로 진행하고, 뉴스는 같은 네 가지
fusion 방식으로 나중에 테스트합니다.

```text
news context concat
news context gating
news context gamma-only FiLM
news context full FiLM
```

## 추천 뉴스 파이프라인

### 1. Audit

확인할 것:
- row count;
- date range;
- day별 기사 수;
- Stage 2 train/validation/test 기간의 missing day;
- duplicate URL/title 비율;
- article length distribution;
- source distribution;
- title-only coverage와 article-text coverage.

### 2. 시간 정렬

각 BTC image end date가 `t`라면 허용되는 news는 다음이어야 합니다.

```text
news_time <= t close cutoff
```

첫 실험용 단순 정책:
- calendar date `t-1`부터 `t`의 UTC daily close 전 뉴스만 사용;
- intraday cutoff가 애매하면 더 보수적으로 `t-1`까지만 사용.

첫 뉴스 실험은 strict `t-1` 정책이 더 안전합니다.

### 3. Daily aggregation 선택지

간단한 순서:
1. 해당 날짜의 latest headline;
2. recency 기준 top-k headlines;
3. max token limit 안에서 concatenated headlines;
4. daily bag-of-words/TF-IDF;
5. cache rule이 안정화된 뒤 article summaries.

### 4. Encoder 선택지

Non-LLM first:
- TF-IDF + SVD daily vector;
- concatenated headline에 trainable word embedding + GRU;
- model version/cache를 기록할 수 있을 때 fixed sentence embedding.

LLM later:
- daily summary text;
- daily embedding vector;
- model name, prompt, revision/version, cache hash, runtime을 반드시 기록.

### 5. Fusion experiments

일별 news vector가 만들어지면 같은 네 가지 head를 사용합니다.

| Track | News version |
|:---|:---|
| 4-N-A | CNN + news-context concat |
| 4-N-B | CNN + news-context gating |
| 4-N-C | CNN + news-context FiLM gamma-only |
| 4-N-D | CNN + news-context FiLM full |

## 추천

뉴스를 Stage 4에서 제거하지 않습니다. 다만 structured numeric context 이후의 두 번째
context source로 둡니다. 이렇게 하면 당장 실행 가능한 실험을 유지하면서도, 교수님이
말한 News/LLM 방향성을 보존할 수 있습니다.
