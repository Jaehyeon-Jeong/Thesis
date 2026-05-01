# Stage 2 Source Map

## English

This file records sources that Stage 2 must check before implementation.

| Area | Source | Stage 2 use |
| --- | --- | --- |
| Stage 1 CNN core | `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py` | Reuse the confirmed Stock_CNN-style baseline core where applicable. |
| Stage 1 data/label/eval patterns | `stage1_reimage_reproduction/src/stage1_reimage/` | Reuse code style, output schema, and leakage guards. |
| Re-image paper summary | `자료조사/Re-image 요약.md` | Image construction, label rule, CNN/training settings, Grad-CAM reporting style. |
| Re-image paper PDF | `자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | Page-level source check before implementing exact image/CNN choices. |
| Grad-CAM summary | `자료조사/Grad-CAM요약.md` | BTC Grad-CAM implementation and interpretation. |
| Grad-CAM PDF | `자료조사/Grad-CAM.pdf` | Original Grad-CAM method check before implementation. |
| BTC OHLCV data | `https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024` | Primary Stage 2 dataset. Local copy audited in 2-2. |

Initial decisions:
- Stage 2 starts with BTC OHLCV only.
- Batch size default remains paper value `128`.
- Stage 2 does not use stock cross-sectional H-L decile portfolios.
- Stage 2 must generate BTC Grad-CAM for baseline runs.

Open checks:
- Exact BTC CSV filename and columns inside Kaggle input.
- Whether the dataset is already daily or needs resampling.
- Date range after cleaning.
- Trading-cost assumption for adjusted returns.
- Train/validation/test date ranges.

## 2-1 Re-check Result

Checked on: 2026-05-01

Checked sources:
- Root plan: `PLAN.md`
- Stage 2 checklist: `stage2_btc_extension/checklist.md`
- Re-image summary: `자료조사/Re-image 요약.md`
- Grad-CAM summary: `자료조사/Grad-CAM요약.md`
- Stage 1 model: `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`
- Stage 1 label/split/normalization: `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`
- Stage 1 prediction/metrics: `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`
- Stage 1 Grad-CAM: `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`

Reusable Stage 1 decisions:
- Binary label rule remains `future R-day return > 0 -> Up class`.
- Training defaults remain cross-entropy, Adam, learning rate `1e-5`, dropout `0.5`, Xavier-style initialization, early stopping by validation loss, and batch size `128` unless a later checklist item records a reason to change a value.
- Pixel normalization must be fit on training images only.
- Evaluation should preserve date, future return, label, logits, probabilities, predicted class, and correctness in prediction outputs.
- Grad-CAM must use the target pre-softmax logit and convolution-layer activation/gradient, then upsample heatmaps to the input image size.

Not directly reusable without adaptation:
- `StockCNNI20` is an I20-only implementation. It reshapes inputs to `(batch, 1, 64, 60)` and uses a fixed classifier input size of `46,080`.
- BTC `I20` can reuse the current Stage 1 CNN core directly after BTC image generation and normalization are implemented.
- BTC `I5` and `I60` must use Stage-1/Stock_CNN-style model variants or a model factory. They must not silently reuse the I20 model with the wrong input shape.
- Stage 1 label code is tied to public stock columns `Ret_5d`, `Ret_20d`, `Ret_60d`, `StockID`, `MarketCap`, `EWMA_vol`, and yearly `.dat/.feather` shards. BTC needs its own OHLCV row alignment and future-return construction.
- Stage 1 prediction schema is stock-specific. BTC prediction outputs should be BTC-specific, for example `Date`, `entry_close`, `exit_close`, `future_return`, `label`, logits, probabilities, predicted class, and correctness.
- Stage 1 Grad-CAM selection assumes stock metadata such as `StockID`, `year`, `shard_index`, and `local_row`. BTC Grad-CAM needs sample selection based on BTC dates and BTC generated image indices.

Stage 2 fixed constraints after 2-1:
- Stage 2 uses the BTC OHLCV dataset first. BTC news and LLM conditioning stay in Stage 4.
- Stage 2 remains a single-asset time-series setting. It does not use stock cross-sectional H-L decile portfolios.
- Stage 2 default batch size is `128`; the Stage 1 Kaggle fast diagnostic batch `1024` is not inherited.
- Stage 2 final comparison tables wait for Stage 1 full outputs, but Stage 2 data audit and implementation planning can continue now.

Open items passed to 2-2:
- Confirm the exact Kaggle input path and CSV filename.
- Confirm BTC OHLCV columns, timestamp format, frequency, duplicates, missing values, and date range.
- Confirm whether daily resampling is needed.
- Confirm volume column meaning and whether volume is usable for Re-image-style volume bars.
- Confirm the first feasible BTC split dates before implementation.

## 2-2 BTC OHLCV Data Audit Result

Checked on: 2026-05-01

Primary source:
- Kaggle dataset: `https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Local audited folder: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV`
- Audited file: `btc_1d_data_2018_to_2025.csv`

Available files in the local BTC folder:
- `btc_15m_data_2018_to_2025.csv`
- `btc_1d_data_2018_to_2025.csv`
- `btc_1h_data_2018_to_2025.csv`
- `btc_4h_data_2018_to_2025.csv`

Data audit summary:
- Raw shape: `2997 x 12`
- Date range: `2018-01-01` to `2026-03-16`
- Frequency: daily, median delta `1 days`, daily delta share `1.0`
- Missing calendar days: `0`
- Duplicate dates: `0`
- Missing OHLCV values: `0`
- Invalid OHLCV rows: `0`
- Zero-volume rows: `0`

Canonical Stage 2 columns:
- `Date` <- `Open time`
- `Open` <- `Open`
- `High` <- `High`
- `Low` <- `Low`
- `Close` <- `Close`
- `Volume` <- `Volume`

Stage 2 implication:
- No daily resampling is needed for the baseline BTC pipeline.
- `Volume` is present and usable for Re-image-style volume bars.
- `Open time` is used as the daily row date because it is the candle start date.
- BTC `I5`, `I20`, and `I60` are all feasible from this daily file.
- As recorded in 2-1, these require three window-specific CNN variants:
  `StockCNNI5`, `StockCNNI20`, and `StockCNNI60`.

Generated audit artifacts:
- `stage2_btc_extension/scripts/audit_btc_ohlcv.py`
- `stage2_btc_extension/notebooks/kaggle_stage2_btc_ohlcv_audit_one_cell.md`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.json`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_head.csv`

## 2-3 BTC Image-Generation Plan Result

Checked on: 2026-05-01

Paper/source basis:
- `자료조사/Re-image 요약.md`, line 25: MA line and volume bars are image
  elements; MA window matches image length.
- `자료조사/Re-image 요약.md`, lines 32-36: 5/20/60-day image periods, 3 pixels
  per day, and image sizes.
- `자료조사/Re-image 요약.md`, line 110: removing MA and/or volume bars is a
  reported robustness/ablation direction.

Stage 2 decisions:
- BTC has OHLCV but not MA columns.
- Stage 2 computes simple moving averages from `Close`.
- Formula: `MA_t^(K) = (1 / K) * sum_{i=0}^{K-1} Close_{t-i}`.
- `I5` uses 5-day SMA, `I20` uses 20-day SMA, and `I60` uses 60-day SMA.
- `min_periods=K`; partial moving averages are not drawn.
- MA uses only current and past close prices, never future close prices.
- Four image specs are fixed:
  - `ohlc`: OHLC only
  - `ohlc_vb`: OHLC + volume bars
  - `ohlc_ma`: OHLC + moving-average line
  - `ohlc_ma_vb`: OHLC + moving-average line + volume bars
- The four specs are compared on a common eligible sample universe within each
  window/horizon setting.

Detailed document:
- `stage2_btc_extension/docs/stage2_image_generation_plan.md`

## 한국어

이 파일은 Stage 2 구현 전에 확인해야 할 근거를 기록합니다.

| 영역 | source | Stage 2에서 쓰는 방식 |
| --- | --- | --- |
| Stage 1 CNN core | `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py` | 확인된 Stock_CNN식 baseline core를 가능한 한 재사용합니다. |
| Stage 1 data/label/eval pattern | `stage1_reimage_reproduction/src/stage1_reimage/` | 코드 스타일, output schema, leakage guard를 재사용합니다. |
| Re-image 논문 요약 | `자료조사/Re-image 요약.md` | image construction, label rule, CNN/training setting, Grad-CAM reporting style 확인. |
| Re-image 논문 PDF | `자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | 정확한 image/CNN 선택 구현 전 page-level source 확인. |
| Grad-CAM 요약 | `자료조사/Grad-CAM요약.md` | BTC Grad-CAM 구현과 해석에 사용. |
| Grad-CAM PDF | `자료조사/Grad-CAM.pdf` | Grad-CAM 원전 방법 구현 전 확인. |
| BTC OHLCV 데이터 | `https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024` | Stage 2 primary dataset. 2-2에서 local copy audit 완료. |

초기 결정:
- Stage 2는 BTC OHLCV만으로 시작합니다.
- 기본 batch size는 논문값 `128`을 유지합니다.
- Stage 2에서는 stock cross-sectional H-L decile portfolio를 사용하지 않습니다.
- Stage 2 baseline run에서도 BTC Grad-CAM을 반드시 생성합니다.

열린 확인 항목:
- Kaggle input 안의 정확한 BTC CSV filename과 column.
- dataset이 이미 daily인지, resampling이 필요한지.
- cleaning 이후 date range.
- transaction-cost-adjusted return에 쓸 trading cost 가정.
- train/validation/test date range.

## 2-1 재확인 결과

확인일: 2026-05-01

확인한 source:
- root plan: `PLAN.md`
- Stage 2 checklist: `stage2_btc_extension/checklist.md`
- Re-image 요약: `자료조사/Re-image 요약.md`
- Grad-CAM 요약: `자료조사/Grad-CAM요약.md`
- Stage 1 model: `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`
- Stage 1 label/split/normalization: `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`
- Stage 1 prediction/metrics: `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`
- Stage 1 Grad-CAM: `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`

Stage 1에서 그대로 유지할 수 있는 결정:
- binary label rule은 그대로 `future R-day return > 0 -> Up class`입니다.
- 학습 기본값은 cross-entropy, Adam, learning rate `1e-5`, dropout `0.5`,
  Xavier-style initialization, validation loss 기준 early stopping, batch size `128`을
  유지합니다. 이후 checklist에서 바꿀 이유가 기록되기 전에는 임의 변경하지 않습니다.
- pixel normalization은 training image에서만 fit해야 합니다.
- prediction output에는 date, future return, label, logits, probability, predicted class,
  correctness를 남겨야 합니다.
- Grad-CAM은 softmax 이후 probability가 아니라 target pre-softmax logit을 사용하고,
  convolution layer activation/gradient로 heatmap을 만든 뒤 입력 image 크기로 upsample합니다.

그대로 재사용하면 안 되고 BTC용 수정이 필요한 부분:
- `StockCNNI20`은 I20 전용 구현입니다. input을 `(batch, 1, 64, 60)`으로 reshape하고
  classifier input size를 `46,080`으로 고정합니다.
- BTC `I20`은 BTC image generation과 normalization을 구현한 뒤 현재 Stage 1 CNN core를
  직접 재사용할 수 있습니다.
- BTC `I5`와 `I60`은 Stage-1/Stock_CNN식 원칙을 따르는 별도 model variant 또는 model
  factory가 필요합니다. I20 model을 shape만 억지로 맞춰 조용히 재사용하지 않습니다.
- Stage 1 label code는 public stock column인 `Ret_5d`, `Ret_20d`, `Ret_60d`,
  `StockID`, `MarketCap`, `EWMA_vol`, yearly `.dat/.feather` shard에 묶여 있습니다.
  BTC는 OHLCV row alignment와 future-return construction을 새로 만들어야 합니다.
- Stage 1 prediction schema는 stock-specific입니다. BTC output은 예를 들어 `Date`,
  `entry_close`, `exit_close`, `future_return`, `label`, logits, probabilities,
  predicted class, correctness 중심으로 새로 잡습니다.
- Stage 1 Grad-CAM sample selection은 `StockID`, `year`, `shard_index`, `local_row`를
  가정합니다. BTC Grad-CAM은 BTC date와 generated image index 기준으로 sample을 선택해야 합니다.

2-1 이후 Stage 2 고정 제약:
- Stage 2는 우선 BTC OHLCV만 사용합니다. BTC news와 LLM conditioning은 Stage 4입니다.
- Stage 2는 단일 자산 time-series setting입니다. stock cross-sectional H-L decile
  portfolio를 사용하지 않습니다.
- Stage 2 기본 batch size는 `128`입니다. Stage 1 Kaggle fast diagnostic의 `1024` batch를
  가져오지 않습니다.
- Stage 2 최종 비교표는 Stage 1 full output 이후 작성하지만, Stage 2 data audit과
  구현 설계는 지금 진행할 수 있습니다.

2-2로 넘기는 열린 항목:
- Kaggle input path와 정확한 CSV filename 확인.
- BTC OHLCV column, timestamp format, frequency, duplicate, missing value, date range 확인.
- daily resampling 필요 여부 확인.
- volume column의 의미와 Re-image-style volume bar에 사용할 수 있는지 확인.
- 구현 전 BTC train/validation/test split 후보 확정.

## 2-2 BTC OHLCV 데이터 audit 결과

확인일: 2026-05-01

Primary source:
- Kaggle dataset: `https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`
- local audited folder: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV`
- audited file: `btc_1d_data_2018_to_2025.csv`

로컬 BTC 폴더에 있는 파일:
- `btc_15m_data_2018_to_2025.csv`
- `btc_1d_data_2018_to_2025.csv`
- `btc_1h_data_2018_to_2025.csv`
- `btc_4h_data_2018_to_2025.csv`

데이터 audit 요약:
- raw shape: `2997 x 12`
- date range: `2018-01-01`부터 `2026-03-16`까지
- frequency: daily, median delta `1 days`, daily delta share `1.0`
- missing calendar days: `0`
- duplicate dates: `0`
- missing OHLCV values: `0`
- invalid OHLCV rows: `0`
- zero-volume rows: `0`

Stage 2 canonical columns:
- `Date` <- `Open time`
- `Open` <- `Open`
- `High` <- `High`
- `Low` <- `Low`
- `Close` <- `Close`
- `Volume` <- `Volume`

Stage 2 implication:
- BTC baseline pipeline에서는 daily resampling이 필요 없습니다.
- `Volume` column이 있으므로 Re-image-style volume bar에 사용할 수 있습니다.
- `Open time`은 daily candle 시작일이므로 row date로 사용합니다.
- 이 daily file로 BTC `I5`, `I20`, `I60` 모두 가능합니다.
- 2-1에서 기록한 것처럼 세 window는 각각 window-specific CNN variant가 필요합니다:
  `StockCNNI5`, `StockCNNI20`, `StockCNNI60`.

생성한 audit artifact:
- `stage2_btc_extension/scripts/audit_btc_ohlcv.py`
- `stage2_btc_extension/notebooks/kaggle_stage2_btc_ohlcv_audit_one_cell.md`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.json`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`
- `stage2_btc_extension/reports/data_audit/btc_ohlcv_head.csv`

## 2-3 BTC image-generation 계획 결과

확인일: 2026-05-01

논문/source 근거:
- `자료조사/Re-image 요약.md`, line 25: MA line과 volume bar는 image 요소이고,
  MA window는 image length와 같습니다.
- `자료조사/Re-image 요약.md`, lines 32-36: 5/20/60일 image period, 하루 3픽셀,
  image size.
- `자료조사/Re-image 요약.md`, line 110: MA와/or volume bar 제거는 robustness/ablation
  방향으로 보고됩니다.

Stage 2 결정:
- BTC에는 OHLCV는 있지만 MA column은 없습니다.
- Stage 2에서는 `Close`에서 simple moving average를 계산합니다.
- 공식: `MA_t^(K) = (1 / K) * sum_{i=0}^{K-1} Close_{t-i}`.
- `I5`는 5-day SMA, `I20`은 20-day SMA, `I60`은 60-day SMA를 사용합니다.
- `min_periods=K`를 사용하며, partial moving average는 그리지 않습니다.
- MA는 현재와 과거 close price만 사용하고 미래 close price는 사용하지 않습니다.
- 네 가지 image spec을 고정합니다:
  - `ohlc`: OHLC only
  - `ohlc_vb`: OHLC + volume bars
  - `ohlc_ma`: OHLC + moving-average line
  - `ohlc_ma_vb`: OHLC + moving-average line + volume bars
- 같은 window/horizon setting 안에서는 네 spec을 공통 eligible sample universe에서
  비교합니다.

상세 문서:
- `stage2_btc_extension/docs/stage2_image_generation_plan.md`
