# Stage 4 News Context Feasibility Plan

## English

News context is now the next Stage 4 track after the V9 numeric-context result.
V9 showed that F&G-only structured numeric FiLM did not robustly improve the
strong `I60/R20/ohlc_ma_vb` visual baseline. The next defensible context source
is richer external news information, but it must be added through a strict
no-leakage daily-vector pipeline.

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

4-N1 local source audit on 2026-06-04:
- First selected file: `BTC_match_title.csv`
- Rows: `30,626`
- Date range: `2011-06-22` to `2025-06-03`
- Selected Stage 4 sample range: `2018-04-29` to `2024-12-11`
- Strict `t-1` sample coverage: `96.04%`
- Trailing 7-day news coverage: `100.00%`
- Note: `BTC_yahoo.csv` was checked but ends on `2024-01-24`, so it is not the
  first news source for the full Stage 4 sample period.

4-N2 publication-time alignment audit on 2026-06-04:
- Locked policy: for a chart image ending at calendar date `t`, use only news
  with calendar date `<= t-1`.
- Same-calendar-day news is excluded until BTC close cutoff and news timestamp
  cutoff are explicitly defended.
- Strict `t-1` coverage: train `96.57%`, validation `97.21%`, test `95.56%`.
- Trailing 7/20/60-day coverage: `100.00%` for train, validation, and test.
- Text vectorizer fit rule: fit only on the train strict-`t-1` 7/20/60-day
  headline-window documents (`671` samples x `3` windows); validation/test
  documents are transform-only.

4-N3 headline-window aggregation on 2026-06-04:
- Raw headline rows: `30,626`.
- Deduped headline rows: `29,208`.
- Removed duplicate normalized-title rows: `1,418`.
- Built sample-level headline windows for `7d`, `20d`, and `60d`.
- Coverage is `100.00%` for all three windows across train, validation, and
  test.
- Full output:
  `outputs/stage4/news/stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet`.

4-N4 train-only TF-IDF/SVD vectorizer on 2026-06-04:
- TF-IDF vocabulary, IDF weights, and SVD were fit only on train split headline
  windows.
- Fit documents: `2,013` = `671` train samples x `3` windows.
- Vectorizer: 1-2 grams, English stop words, `min_df=2`, `max_df=0.95`,
  `max_features=10,000`.
- SVD: requested `32` components, actual `32` components.
- Explained variance ratio sum: `0.5856`.
- Full output:
  `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`.
- First output vector family:
  `news_svd_7d/20d/60d`, `news_count_7d/20d/60d`,
  `unique_source_count_7d/20d/60d`, and log-count variants.

## 4-N5 Final News Context Table

The sample-level news context builder now writes a model-ready context artifact:

- Script: `scripts/build_stage4_news_context_features.py`
- Context artifact:
  `outputs/stage4/context/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60/seed_42/`
- Rows: `2,399`
- Split counts: train `671`, validation `287`, test `1,441`
- Context dimension: `102`
  - `96` SVD features = `32` components x `7/20/60` windows
  - `6` log-count features = news-count and unique-source-count for each
    window
- Normalization: train median imputation, train quantile clipping, train
  z-score scaling.
- Missing warnings: none.

Planning decision after V9:
- Use this dataset as the next Stage 4 context source.
- First news experiment should be headline-only.
- Use strict `t-1` alignment by default.
- Fit text preprocessing, TF-IDF, and SVD on train-period news only.
- First news windows: 7-day short shock, 20-day forecast-horizon context, and
  60-day I60 chart-regime context.
- First vector family: `news_svd_7d + news_svd_20d + news_svd_60d` plus
  `news_count_7d/20d/60d`.
- Compare `CNN + news concat` before claiming that FiLM modulation helps.
- Then run `CNN + news bounded last-block FiLM`.
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

For this reason, news should start with the simplest leakage-safe text vector,
then use the same fusion logic:

```text
news context concat
news bounded last-block FiLM
news + F&G combined context, only if news-only is useful
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
5. build trailing 7-day, 20-day, and 60-day news-count/embedding summaries;
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

Once a daily news vector exists, use this order:

| Track | News version |
|:---|:---|
| 4-N-A | visual-only same news-aligned sample control |
| 4-N-B | CNN + news-context concat |
| 4-N-C | CNN + news bounded last-block FiLM |
| 4-N-D | news + F&G combined context, only if news-only is useful |

## Recommendation

Do not continue arbitrary gamma/beta scale tuning after V9. Treat F&G-only
numeric FiLM as a negative/unstable result for the main claim, then test whether
headline-level news gives richer external market-regime signal.

## 한국어

뉴스 context는 V9 numeric-context 결과 이후 Stage 4의 다음 track입니다. V9에서
F&G-only structured numeric FiLM이 강한 `I60/R20/ohlc_ma_vb` visual baseline을
robust하게 개선하지 못했기 때문입니다. 다음으로 방어 가능한 context source는 더
풍부한 외부 뉴스 정보이지만, strict no-leakage daily-vector pipeline으로 넣어야
합니다.

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

2026-06-04 4-N1 local source audit 결과:
- 첫 선택 파일: `BTC_match_title.csv`
- Rows: `30,626`
- Date range: `2011-06-22` to `2025-06-03`
- Selected Stage 4 sample range: `2018-04-29` to `2024-12-11`
- Strict `t-1` sample coverage: `96.04%`
- Trailing 7-day news coverage: `100.00%`
- 참고: `BTC_yahoo.csv`도 확인했지만 `2024-01-24`에서 끝나므로 full Stage 4
  sample period의 첫 news source로 쓰지 않습니다.

2026-06-04 4-N2 publication-time alignment audit 결과:
- 고정 policy: chart image end date가 calendar date `t`이면 calendar date
  `<= t-1`인 뉴스만 사용합니다.
- Same-calendar-day news는 BTC close cutoff와 news timestamp cutoff를 명확히
  방어하기 전까지 제외합니다.
- Strict `t-1` coverage: train `96.57%`, validation `97.21%`, test `95.56%`.
- Trailing 7/20/60-day coverage: train/validation/test 모두 `100.00%`.
- Text vectorizer fit rule: train strict-`t-1` 7/20/60-day headline-window
  document에만 fit합니다 (`671` samples x `3` windows). Validation/test
  document는 transform-only입니다.

2026-06-04 4-N3 headline-window aggregation 결과:
- Raw headline rows: `30,626`.
- Deduped headline rows: `29,208`.
- 제거한 duplicate normalized-title rows: `1,418`.
- Sample-level `7d`, `20d`, `60d` headline window를 만들었습니다.
- Train/validation/test에서 세 window 모두 coverage `100.00%`입니다.
- Full output:
  `outputs/stage4/news/stage4_news_headline_windows_i60_r20/sample_headline_windows.parquet`.

2026-06-04 4-N4 train-only TF-IDF/SVD vectorizer 결과:
- TF-IDF vocabulary, IDF weight, SVD는 train split headline window에만 fit했습니다.
- Fit document 수: `2,013` = train sample `671`개 x window `3`개.
- Vectorizer: 1-2 gram, English stop words, `min_df=2`, `max_df=0.95`,
  `max_features=10,000`.
- SVD: requested `32` components, actual `32` components.
- Explained variance ratio sum: `0.5856`.
- Full output:
  `outputs/stage4/news/stage4_news_tfidf_svd_i60_r20/news_tfidf_svd_features.parquet`.
- 첫 output vector family:
  `news_svd_7d/20d/60d`, `news_count_7d/20d/60d`,
  `unique_source_count_7d/20d/60d`, log-count variants입니다.

## 4-N5 최종 News Context Table

Sample-level news context builder가 모델 입력용 context artifact를 생성했습니다.

- Script: `scripts/build_stage4_news_context_features.py`
- Context artifact:
  `outputs/stage4/context/stage4_news_context_i60_ohlc_ma_vb_r20_tfidf_svd_w7_20_60/seed_42/`
- Row 수: `2,399`
- Split counts: train `671`, validation `287`, test `1,441`
- Context dimension: `102`
  - `96`개 SVD feature = `32` components x `7/20/60` windows
  - `6`개 log-count feature = 각 window의 news-count와 unique-source-count
- Normalization: train median imputation, train quantile clipping, train
  z-score scaling입니다.
- Missing warning은 없습니다.

V9 이후 계획 결정:
- 이 dataset은 다음 Stage 4 context source로 사용합니다.
- 첫 news experiment는 headline-only로 시작합니다.
- 기본 alignment는 strict `t-1`입니다.
- Text preprocessing, TF-IDF, SVD는 train-period news에만 fit합니다.
- 첫 news window는 7-day 단기 충격, 20-day 예측 horizon context, 60-day I60
  chart-regime context로 나눕니다.
- 첫 vector family는 `news_svd_7d + news_svd_20d + news_svd_60d`와
  `news_count_7d/20d/60d`입니다.
- FiLM modulation을 주장하기 전에 `CNN + news concat`을 먼저 비교합니다.
- 그 다음 `CNN + news bounded last-block FiLM`을 실행합니다.
- Article summary, sentence-transformer embedding, LLM summary, LLM embedding은
  no-leakage headline track이 안정화된 뒤로 미룹니다.

## 뉴스가 첫 main experiment가 아닌 이유

뉴스는 도움이 될 수 있지만 추가 위험이 있습니다.
- publication time을 정확히 정렬해야 합니다.
- 하루 중 늦게 나온 뉴스는 chart close 시점 전에 알 수 있었는지 애매할 수 있습니다.
- article text가 크고 noise가 많습니다.
- crypto 전체 뉴스가 BTC-specific signal이 아닐 수 있습니다.
- LLM summary/embedding은 cache와 version control이 필요합니다.

그래서 뉴스는 가장 단순한 leakage-safe text vector에서 시작하고, 같은 fusion logic으로
테스트합니다.

```text
news context concat
news bounded last-block FiLM
news-only가 유용할 때 news + F&G combined context
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
5. trailing 7-day, 20-day, 60-day news-count/embedding summary를 만듭니다;
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

일별 news vector가 만들어지면 아래 순서로 사용합니다.

| Track | News version |
|:---|:---|
| 4-N-A | 같은 news-aligned sample의 visual-only control |
| 4-N-B | CNN + news-context concat |
| 4-N-C | CNN + news bounded last-block FiLM |
| 4-N-D | news-only가 유용할 때 news + F&G combined context |

## 추천

V9 이후 임의적인 gamma/beta scale tuning은 계속하지 않습니다. F&G-only numeric
FiLM은 main claim 기준 negative/unstable result로 정리하고, headline-level news가 더
풍부한 external market-regime signal을 주는지 테스트합니다.
