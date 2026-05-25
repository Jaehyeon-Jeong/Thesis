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

Current public metadata checked on 2026-05-25:
- Split: `train`
- Rows: `210,832`
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

Planning decision from 4-3:
- Use this dataset as a second-phase news-context source.
- First news experiment should be headline-only.
- Use strict `t-1` alignment by default.
- Fit text preprocessing, TF-IDF, and SVD on train-period news only.
- Defer article summaries, sentence-transformer embeddings, LLM summaries, and
  LLM embeddings until the no-leakage headline track is stable.

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

First policy:
- use news up to the end of calendar date `t-1`;
- do not use same-calendar-day news until BTC candle cutoff and news timestamp
  cutoff are explicitly defined.

The strict `t-1` policy is the default for the first news experiment.

### 3. Daily Aggregation Options

Start simple:
1. exact title/url duplicate removal;
2. concatenated headlines with fixed top-k or max-character limit;
3. daily bag-of-words/TF-IDF;
4. TF-IDF + SVD daily vector fit on train-period news only;
5. add trailing 7-day and 60-day news-count/embedding summaries as ablations;
6. article summaries only after caching rules are stable.

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

2026-05-25 기준 공개 metadata 확인:
- Split: `train`
- Rows: `210,832`
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

4-3 계획 결정:
- 이 dataset은 second-phase news-context source로 사용합니다.
- 첫 news experiment는 headline-only로 시작합니다.
- 기본 alignment는 strict `t-1`입니다.
- Text preprocessing, TF-IDF, SVD는 train-period news에만 fit합니다.
- Article summary, sentence-transformer embedding, LLM summary, LLM embedding은
  no-leakage headline track이 안정화된 뒤로 미룹니다.

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

첫 정책:
- calendar date `t-1`의 끝까지 나온 뉴스만 사용합니다.
- BTC candle cutoff와 news timestamp cutoff가 명확해지기 전까지 same-day news는
  사용하지 않습니다.

첫 뉴스 실험은 strict `t-1` 정책을 기본값으로 둡니다.

### 3. Daily aggregation 선택지

간단한 순서:
1. exact title/url duplicate 제거;
2. fixed top-k 또는 max-character limit 안에서 headline concat;
3. daily bag-of-words/TF-IDF;
4. train-period news에만 fit한 TF-IDF + SVD daily vector;
5. trailing 7-day와 60-day news-count/embedding summary를 ablation으로 추가;
6. cache rule이 안정화된 뒤 article summaries.

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
