# 4-N0. Numeric-Context Handoff and News Scope Lock

## English

Status: complete.

Decision:
- Structured numeric context experiments are complete through `4-V9`.
- F&G-only numeric FiLM produced some ranking signal in selected seeds, but did
  not robustly beat the Stage 2 `I60/R20/ohlc_ma_vb` visual baseline.
- Further arbitrary gamma/beta scale search is not the next priority.
- The next Stage 4 context source is external news.

Locked news scope:
- First version: headline-only, non-LLM.
- Alignment: strict `t-1`; for an image ending at date `t`, use news available
  up to calendar date `t-1`.
- First vector family after 4-N2: `news_svd_7d + news_svd_20d + news_svd_60d`
  plus `news_count_7d/20d/60d`.
- Fit text preprocessing, TF-IDF, and SVD on train-period news only.
- Keep the chart image fixed as `I60/R20/ohlc_ma_vb`.

Next experiment:
- `4-N1`: source audit for `edaschau/bitcoin_news`.
- `4-N2`: no-leakage publication-time alignment audit.

## 한국어

상태: 완료.

결정:
- Structured numeric context 실험은 `4-V9`까지 완료했습니다.
- F&G-only numeric FiLM은 일부 seed에서 ranking signal을 보였지만 Stage 2
  `I60/R20/ohlc_ma_vb` visual baseline을 안정적으로 넘지 못했습니다.
- 따라서 임의적인 gamma/beta scale search를 계속하는 것은 다음 우선순위가
  아닙니다.
- 다음 Stage 4 context source는 외부 뉴스입니다.

고정한 news scope:
- 첫 version: headline-only, non-LLM.
- 정렬: strict `t-1`; image end date가 `t`이면 calendar date `t-1`까지의
  뉴스만 사용합니다.
- 4-N2 이후 첫 vector family:
  `news_svd_7d + news_svd_20d + news_svd_60d` +
  `news_count_7d/20d/60d`.
- Text preprocessing, TF-IDF, SVD는 train-period news에만 fit합니다.
- Chart image는 `I60/R20/ohlc_ma_vb`로 고정합니다.

다음 실험:
- `4-N1`: `edaschau/bitcoin_news` source audit.
- `4-N2`: no-leakage publication-time alignment audit.
