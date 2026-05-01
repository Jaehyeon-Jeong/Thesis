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

## 2-4 BTC Label, Split, and Normalization Plan Result

Checked on: 2026-05-01

Paper/source basis:
- `자료조사/Re-image 요약.md`, line 41: 1993-2000 train/validation,
  2001-2019 test, and 70/30 random split inside the train/validation period.
- `자료조사/Re-image 요약.md`, line 49: binary classification and train-pixel
  mean/std normalization.
- Stage 1 implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`.

Stage 2 decisions:
- BTC future return is constructed from close prices:
  `future_return_{t,R} = Close_{t+R} / Close_t - 1`.
- `label = 1` if `future_return > 0`; otherwise `0`.
- Exact zero return belongs to class `0`.
- BTC uses daily bars, so `R5/R20/R60` mean 5/20/60 daily bars.
- Primary reporting is capped at `2024-12-31`; 2025-2026 is reserved as an
  optional later holdout.
- Split:
  - train/validation pool signal dates: `2018-01-01` to `2020-12-31`
  - split the train/validation pool 70/30 at random with seed `42`
  - test signal dates: `2021-01-01` to `2024-12-31`
- Purge rule at period boundaries: `label_end_date <= split_signal_end`.
- Normalization follows Stage 1 but is stored per Stage 2 experiment tuple:
  `(image_window, image_spec, return_horizon)`.
- Normalization uses train images only.
- BTC-specific caveat: train/validation random split is paper-aligned, but BTC
  rolling samples can overlap strongly; chronological validation can be added as
  a robustness check later.

Generated artifact:
- `stage2_btc_extension/docs/stage2_label_split_normalization_plan.md`
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

## 2-5 BTC Baseline CNN Adaptation Plan Result

Checked on: 2026-05-01

Paper/source basis:
- `자료조사/Re-image 요약.md`, line 36: baseline image sizes `32x15`,
  `64x60`, and `96x180`.
- `자료조사/Re-image 요약.md`, line 47: I5/I20/I60 model depths, channel
  sequence, flatten dimensions, and parameter counts.
- `자료조사/Re-image 요약.md`, line 49: baseline training defaults.
- GitHub reference: `lich99/Stock_CNN/models/baseline.py`, commit
  `415e2acf2a5013afca67e383acd3edc61fced840`.
- Stage 1 implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`.

Source check:
- The checked GitHub model is I20-specific: it reshapes input to
  `(batch, 1, 64, 60)` and uses `Linear(46080, 2)`.
- Therefore BTC I20 can reuse the Stage 1/GitHub-style core directly.
- BTC I5 and I60 require separate model variants; they must not reuse the I20
  reshape or classifier.

Stage 2 decisions:
- `I5` -> `stock_cnn_i5`, input `(batch, 1, 32, 15)`, 2 blocks,
  flatten dim `15360`, expected params `155138`.
- `I20` -> `stock_cnn_i20`, input `(batch, 1, 64, 60)`, 3 blocks,
  flatten dim `46080`, expected params `708866`.
- `I60` -> `stock_cnn_i60`, input `(batch, 1, 96, 180)`, 4 blocks,
  flatten dim `184320`, expected params `2952962`.
- Model selection is by image window, not by return horizon.
- All four image specifications remain one-channel grayscale images.
- Default BTC baseline training uses from-scratch initialization rather than a
  stock checkpoint transfer.

Generated artifacts:
- `stage2_btc_extension/docs/stage2_baseline_cnn_adaptation_plan.md`
- `stage2_btc_extension/reports/tables/stage2_baseline_cnn_architecture_plan.csv`

## 2-6 BTC Evaluation and Trading Metric Plan Result

Checked on: 2026-05-01

Paper/source basis:
- `자료조사/Re-image 요약.md`, lines 5, 7, and 73-77: classification accuracy
  and H-L portfolio evaluation.
- `자료조사/Re-image 요약.md`, line 77: short-horizon turnover warning.
- Stage 1 evaluation implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`.
- Stage 2 constraint from 2-1: BTC is a single asset, so stock
  cross-sectional H-L decile portfolios are not directly applicable.

Stage 2 decisions:
- BTC reports classification metrics plus time-series trading metrics.
- Required sentence for reports: the original paper can form H-L decile
  portfolios because it is cross-sectional, but BTC is a single asset, so Stage
  2 reports classification and time-series trading metrics instead.
- Prediction CSV preserves BTC dates, image/window/spec metadata, future return,
  label, logits, probabilities, predicted class, and correctness.
- Classification metrics include accuracy, precision, recall, F1, ROC AUC,
  average precision, Brier score, log loss, confusion matrix, majority-class
  baseline, and probability/return correlations.
- Calibration is required as a 10-bin `prob_up` table.
- Trading strategies are `long_flat` and `long_short`.
- Trading metrics use overlapping-horizon daily backtests, annualized with
  365 daily periods.
- Report gross metrics and configurable transaction-cost-adjusted metrics.
  Initial planning value: `10 bps` per unit turnover.

Generated artifacts:
- `stage2_btc_extension/docs/stage2_evaluation_trading_metric_plan.md`
- `stage2_btc_extension/reports/tables/stage2_metric_schema.csv`

## 2-7 BTC Grad-CAM Plan Result

Checked on: 2026-05-01

Paper/source basis:
- Root `PLAN.md`: BTC Grad-CAM is required in Stage 2.
- Re-image Figure 13: original image row followed by CNN-layer Grad-CAM heatmap
  rows. The Stage 1 source map records this figure as Re-image Figure 13 and
  local Re-image summary pp.41-49.
- `자료조사/Grad-CAM요약.md`, method pp.4-6:
  - use the pre-softmax target class score.
  - compute channel weights by spatially averaging gradients.
  - compute `ReLU(sum_k alpha_k^c A^k)`.
  - upsample to the input image size with bilinear interpolation.
- Stage 1 implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`.

Stage 2 decisions:
- BTC Grad-CAM is a class-discriminative heatmap, not a raw feature map.
- The target score is the pre-softmax class logit.
- Target layers follow the image-window model variant.
  - I5: 2 convolution rows.
  - I20: 3 convolution rows.
  - I60: 4 convolution rows.
- Heatmaps are upsampled to each model input size.
- Final report figures use 10 predicted Up and 10 predicted Down samples when
  available.
- Quick smoke figures may use 2 samples per class.
- Save a fixed-date sample list for later Linear/FiLM comparisons.

Generated artifacts:
- `stage2_btc_extension/docs/stage2_gradcam_plan.md`
- `stage2_btc_extension/reports/tables/stage2_gradcam_output_schema.csv`

## 2-8 Kaggle Runner and Output Plan Result

Checked on: 2026-05-01

Paper/source basis:
- Root `PLAN.md`: full experiments use Kaggle Notebook by default, and the
  Kaggle wrapper calls shared `src/` and `scripts/` code.
- Stage 1 standard runner:
  `stage1_reimage_reproduction/notebooks/kaggle_stage1_single_horizon_one_cell.md`.
- Stage 2 audit runner:
  `stage2_btc_extension/notebooks/kaggle_stage2_btc_ohlcv_audit_one_cell.md`.

Stage 2 decisions:
- Stage 2 follows the Stage 1 one-cell Kaggle wrapper pattern.
- The Kaggle cell copies the code snapshot, patches config, and calls repo
  scripts.
- Actual implementation remains in `src/` and `scripts/`.
- Run one experiment tuple at a time.
- Strict baseline defaults are `batch_size=128`, mixed precision off,
  DataParallel off, and fast cuDNN off.
- Speed options must be recorded in `run_manifest.json` if enabled.
- Large outputs stay in Kaggle/working outputs and are not committed to GitHub.
- GitHub receives planning docs, code/config, small summary tables, and only
  small selected figures when needed.

Generated artifacts:
- `stage2_btc_extension/docs/stage2_kaggle_runner_output_plan.md`
- `stage2_btc_extension/notebooks/kaggle_stage2_btc_baseline_one_cell.md`
- `stage2_btc_extension/reports/tables/stage2_kaggle_run_matrix.csv`

## 2-I0 Implementation Readiness Review Result

Checked on: 2026-05-01

Checked sources:
- Root `PLAN.md`: confirms one-step implementation workflow and Kaggle-first
  full experiment execution.
- Stage 2 checklist: confirms planning items `2-0` through `2-8` are complete.
- Stage 2 planning docs: image generation, label/split/normalization, model
  adaptation, evaluation/trading, Grad-CAM, and Kaggle runner plans.
- Stage 2 small artifacts: data audit, split-count table, architecture table,
  metric schema, Grad-CAM schema, and Kaggle run matrix.

Readiness verdict:
- Stage 2 is ready to move from planning to implementation.
- Stage 1 full outputs are still required before final comparison tables, but
  they do not block Stage 2 implementation.
- The next checklist item is `2-I1`, shared Stage 2 config/code scaffold.

Implementation constraints carried forward:
- Keep Stage 1/Stock_CNN-style CNN core.
- Use model variants by image window: I5, I20, I60.
- Keep default Stage 2 `batch_size=128`.
- Do not use stock cross-sectional H-L decile portfolios for BTC.
- Generate BTC Grad-CAM for baseline runs.
- Keep actual logic in `src/` and `scripts/`, not in the Kaggle one-cell
  wrapper.

Generated artifacts:
- `stage2_btc_extension/docs/stage2_implementation_readiness_review.md`
- `stage2_btc_extension/reports/tables/stage2_implementation_task_map.csv`

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

## 2-4 BTC label, split, normalization 계획 결과

확인일: 2026-05-01

논문/source 근거:
- `자료조사/Re-image 요약.md`, line 41: 1993-2000 train/validation,
  2001-2019 test, train/validation period 내부 70/30 random split.
- `자료조사/Re-image 요약.md`, line 49: binary classification과 train-pixel mean/std
  normalization.
- Stage 1 implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`.

Stage 2 결정:
- BTC future return은 close price에서 만듭니다:
  `future_return_{t,R} = Close_{t+R} / Close_t - 1`.
- `future_return > 0`이면 `label = 1`, 아니면 `0`입니다.
- 정확히 0인 return은 class `0`입니다.
- BTC는 daily bar를 사용하므로 `R5/R20/R60`은 5/20/60 daily bar입니다.
- 기본 보고 기간은 `2024-12-31`까지로 제한하고, 2025-2026은 optional later holdout으로
  남깁니다.
- Split:
  - train/validation pool signal dates: `2018-01-01` to `2020-12-31`
  - train/validation pool을 seed `42`로 70/30 random split
  - test signal dates: `2021-01-01` to `2024-12-31`
- Period boundary purge rule: `label_end_date <= split_signal_end`.
- Normalization은 Stage 1 원칙을 따르되 Stage 2 experiment tuple
  `(image_window, image_spec, return_horizon)`별로 따로 저장합니다.
- Normalization은 train image에서만 fit합니다.
- BTC-specific caveat: train/validation random split은 논문 방식에 맞지만,
  BTC rolling sample은 많이 겹칠 수 있습니다. chronological validation은 나중
  robustness check로 추가할 수 있습니다.

생성 artifact:
- `stage2_btc_extension/docs/stage2_label_split_normalization_plan.md`
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

## 2-5 BTC baseline CNN adaptation 계획 결과

확인일: 2026-05-01

논문/source 근거:
- `자료조사/Re-image 요약.md`, line 36: baseline image size `32x15`,
  `64x60`, `96x180`.
- `자료조사/Re-image 요약.md`, line 47: I5/I20/I60 model depth, channel
  sequence, flatten dimension, parameter count.
- `자료조사/Re-image 요약.md`, line 49: baseline training default.
- GitHub reference: `lich99/Stock_CNN/models/baseline.py`, commit
  `415e2acf2a5013afca67e383acd3edc61fced840`.
- Stage 1 implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`.

Source 확인:
- 확인한 GitHub model은 I20 전용입니다. input을 `(batch, 1, 64, 60)`으로
  reshape하고 `Linear(46080, 2)`를 사용합니다.
- 따라서 BTC I20은 Stage 1/GitHub식 core를 직접 재사용할 수 있습니다.
- BTC I5와 I60은 별도 model variant가 필요하며, I20 reshape나 classifier를
  재사용하면 안 됩니다.

Stage 2 결정:
- `I5` -> `stock_cnn_i5`, input `(batch, 1, 32, 15)`, 2 blocks,
  flatten dim `15360`, expected params `155138`.
- `I20` -> `stock_cnn_i20`, input `(batch, 1, 64, 60)`, 3 blocks,
  flatten dim `46080`, expected params `708866`.
- `I60` -> `stock_cnn_i60`, input `(batch, 1, 96, 180)`, 4 blocks,
  flatten dim `184320`, expected params `2952962`.
- model은 return horizon이 아니라 image window로 선택합니다.
- 네 image specification은 모두 1-channel grayscale image로 유지합니다.
- 기본 BTC baseline은 stock checkpoint transfer가 아니라 from-scratch
  initialization으로 학습합니다.

생성 artifact:
- `stage2_btc_extension/docs/stage2_baseline_cnn_adaptation_plan.md`
- `stage2_btc_extension/reports/tables/stage2_baseline_cnn_architecture_plan.csv`

## 2-6 BTC evaluation과 trading metric 계획 결과

확인일: 2026-05-01

논문/source 근거:
- `자료조사/Re-image 요약.md`, lines 5, 7, 73-77: classification accuracy와
  H-L portfolio evaluation.
- `자료조사/Re-image 요약.md`, line 77: short-horizon turnover 주의.
- Stage 1 evaluation implementation:
  `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`.
- 2-1에서 확정한 Stage 2 constraint: BTC는 단일 자산이므로 stock cross-sectional
  H-L decile portfolio를 직접 적용할 수 없습니다.

Stage 2 결정:
- BTC는 classification metric과 time-series trading metric을 함께 보고합니다.
- 보고서 필수 문장: 원논문은 cross-sectional이라 H-L decile portfolio를 구성할 수
  있지만, BTC는 단일 자산이므로 Stage 2는 classification과 time-series trading
  metric을 사용합니다.
- Prediction CSV에는 BTC date, image/window/spec metadata, future return, label,
  logits, probability, predicted class, correctness를 보존합니다.
- Classification metric은 accuracy, precision, recall, F1, ROC AUC, average
  precision, Brier score, log loss, confusion matrix, majority-class baseline,
  probability/return correlation을 포함합니다.
- Calibration은 `prob_up` 10-bin table을 필수로 만듭니다.
- Trading strategy는 `long_flat`, `long_short`입니다.
- Trading metric은 overlapping-horizon daily backtest로 계산하고, annualization은
  365 daily periods를 사용합니다.
- gross metric과 configurable transaction-cost-adjusted metric을 모두 보고합니다.
  초기 계획값은 unit turnover당 `10 bps`입니다.

생성 artifact:
- `stage2_btc_extension/docs/stage2_evaluation_trading_metric_plan.md`
- `stage2_btc_extension/reports/tables/stage2_metric_schema.csv`

## 2-7 BTC Grad-CAM 계획 결과

확인일: 2026-05-01

논문/source 근거:
- Root `PLAN.md`: Stage 2에서도 BTC Grad-CAM은 필수입니다.
- Re-image Figure 13: original image row 다음에 CNN layer별 Grad-CAM heatmap row를
  둡니다. Stage 1 source-map은 이 figure를 Re-image Figure 13 및 로컬 Re-image
  summary pp.41-49로 기록했습니다.
- `자료조사/Grad-CAM요약.md`, method pp.4-6:
  - softmax 이전 target class score를 사용합니다.
  - gradient spatial average로 channel weight를 계산합니다.
  - `ReLU(sum_k alpha_k^c A^k)`로 heatmap을 만듭니다.
  - bilinear interpolation으로 input image size까지 upsample합니다.
- Stage 1 구현:
  `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`.

Stage 2 결정:
- BTC Grad-CAM은 raw feature map이 아니라 class-discriminative heatmap입니다.
- target score는 softmax 이전 class logit입니다.
- target layer는 image window model variant를 따릅니다.
  - I5: 2 convolution rows
  - I20: 3 convolution rows
  - I60: 4 convolution rows
- Heatmap은 각 model input size로 upsample합니다.
- 최종 보고 figure는 가능한 경우 predicted Up 10개와 predicted Down 10개를 사용합니다.
- quick smoke figure는 class당 2개를 사용할 수 있습니다.
- 이후 Linear/FiLM 비교를 위해 fixed-date sample list를 저장합니다.

생성 artifact:
- `stage2_btc_extension/docs/stage2_gradcam_plan.md`
- `stage2_btc_extension/reports/tables/stage2_gradcam_output_schema.csv`

## 2-8 Kaggle runner와 output 계획 결과

확인일: 2026-05-01

논문/source 근거:
- Root `PLAN.md`: full experiment 기본 실행 환경은 Kaggle Notebook이며,
  공통 `src/`/`scripts/`를 Kaggle wrapper가 호출합니다.
- Stage 1 표준 실행:
  `stage1_reimage_reproduction/notebooks/kaggle_stage1_single_horizon_one_cell.md`.
- Stage 2 audit 실행:
  `stage2_btc_extension/notebooks/kaggle_stage2_btc_ohlcv_audit_one_cell.md`.

Stage 2 결정:
- Stage 2는 Stage 1과 같은 one-cell Kaggle wrapper pattern을 따릅니다.
- Kaggle cell은 code snapshot 복사, config patch, repo script 호출만 담당합니다.
- 실제 구현은 `src/`와 `scripts/`에 둡니다.
- Experiment tuple 하나씩 실행합니다.
- strict baseline 기본값은 `batch_size=128`, mixed precision off,
  DataParallel off, fast cuDNN off입니다.
- 속도 옵션을 켜면 `run_manifest.json`에 기록합니다.
- 대용량 output은 Kaggle/working output에 남기고 GitHub에는 올리지 않습니다.
- GitHub에는 planning docs, code/config, 작은 summary table, 작은 selected figure만
  publish합니다.

생성 artifact:
- `stage2_btc_extension/docs/stage2_kaggle_runner_output_plan.md`
- `stage2_btc_extension/notebooks/kaggle_stage2_btc_baseline_one_cell.md`
- `stage2_btc_extension/reports/tables/stage2_kaggle_run_matrix.csv`

## 2-I0 구현 readiness review 결과

확인일: 2026-05-01

확인한 source:
- Root `PLAN.md`: 한 단계씩 구현하는 workflow와 Kaggle-first full experiment 실행
  원칙을 확인했습니다.
- Stage 2 checklist: planning 항목 `2-0`부터 `2-8`까지 완료된 것을 확인했습니다.
- Stage 2 planning docs: image generation, label/split/normalization, model
  adaptation, evaluation/trading, Grad-CAM, Kaggle runner 계획.
- Stage 2 small artifacts: data audit, split-count table, architecture table,
  metric schema, Grad-CAM schema, Kaggle run matrix.

Readiness 판정:
- Stage 2는 planning에서 implementation으로 넘어갈 준비가 됐습니다.
- Stage 1 full output은 최종 비교표에는 필요하지만 Stage 2 구현을 막지는 않습니다.
- 다음 checklist 항목은 `2-I1`, shared Stage 2 config/code scaffold입니다.

계속 가져갈 구현 제약:
- Stage 1/Stock_CNN식 CNN core를 유지합니다.
- image window별 model variant를 사용합니다: I5, I20, I60.
- Stage 2 기본 `batch_size=128`을 유지합니다.
- BTC에는 stock cross-sectional H-L decile portfolio를 사용하지 않습니다.
- BTC baseline run에도 Grad-CAM을 생성합니다.
- 실제 로직은 Kaggle one-cell wrapper가 아니라 `src/`와 `scripts/`에 둡니다.

생성 artifact:
- `stage2_btc_extension/docs/stage2_implementation_readiness_review.md`
- `stage2_btc_extension/reports/tables/stage2_implementation_task_map.csv`

## 2-I1 Shared Config/Code Scaffold Result

Checked on: 2026-05-01

Implementation-source basis:
- Root `PLAN.md`: local and Kaggle must use one shared codebase, with
  environment differences controlled by config.
- Stage 1 helper style:
  - `stage1_reimage_reproduction/src/stage1_reimage/config.py`
  - `stage1_reimage_reproduction/src/stage1_reimage/paths.py`
  - `stage1_reimage_reproduction/src/stage1_reimage/runtime.py`
  - `stage1_reimage_reproduction/src/stage1_reimage/seed.py`

Stage 2 scaffold created:
- `stage2_btc_extension/configs/env_local.yaml`
- `stage2_btc_extension/configs/env_kaggle.yaml`
- `stage2_btc_extension/src/stage2_btc/config.py`
- `stage2_btc_extension/src/stage2_btc/paths.py`
- `stage2_btc_extension/src/stage2_btc/runtime.py`
- `stage2_btc_extension/src/stage2_btc/seed.py`

Important note:
- This item records implementation scaffolding, not a new paper-derived model
  decision.
- Paper-derived model/image/training values are stored in config so later code
  can use them consistently.

## 2-I1 공통 Config/Code Scaffold 결과

확인일: 2026-05-01

구현 근거:
- Root `PLAN.md`: local과 Kaggle은 하나의 공통 codebase를 사용하고, 환경 차이는
  config로 관리합니다.
- Stage 1 helper style:
  - `stage1_reimage_reproduction/src/stage1_reimage/config.py`
  - `stage1_reimage_reproduction/src/stage1_reimage/paths.py`
  - `stage1_reimage_reproduction/src/stage1_reimage/runtime.py`
  - `stage1_reimage_reproduction/src/stage1_reimage/seed.py`

생성한 Stage 2 scaffold:
- `stage2_btc_extension/configs/env_local.yaml`
- `stage2_btc_extension/configs/env_kaggle.yaml`
- `stage2_btc_extension/src/stage2_btc/config.py`
- `stage2_btc_extension/src/stage2_btc/paths.py`
- `stage2_btc_extension/src/stage2_btc/runtime.py`
- `stage2_btc_extension/src/stage2_btc/seed.py`

중요:
- 이 항목은 새 논문 기반 model 결정을 추가한 것이 아니라 구현 scaffold입니다.
- 논문에서 가져온 model/image/training 값은 이후 코드가 일관되게 사용하도록 config에
  저장했습니다.

## 2-I2 to 2-I10 Implementation Result

Checked on: 2026-05-01

Implemented files:
- Data loader: `stage2_btc_extension/src/stage2_btc/data/ohlcv.py`
- Image generator: `stage2_btc_extension/src/stage2_btc/imaging/ohlcv_image.py`
- Label/split/normalization: `stage2_btc_extension/src/stage2_btc/data/label_split.py`
- Model variants: `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`
- Training loop: `stage2_btc_extension/src/stage2_btc/training/loop.py`
- Prediction metrics: `stage2_btc_extension/src/stage2_btc/evaluation/prediction.py`
- Trading metrics: `stage2_btc_extension/src/stage2_btc/evaluation/backtest.py`
- Grad-CAM: `stage2_btc_extension/src/stage2_btc/interpretability/gradcam.py`
- Kaggle runner: `stage2_btc_extension/notebooks/kaggle_stage2_btc_baseline_one_cell.md`

Source basis:
- BTC loader/image/label code follows Stage 2 planning documents from `2-2`
  through `2-4`.
- I20 model follows Stage 1/GitHub-style core.
- I5/I60 variants are Stage-1/Stock_CNN-style extensions that match the paper
  summary targets for depth, channel sequence, flatten dimension, and parameter
  count.
- BTC trading metrics replace stock H-L decile evaluation because BTC is a
  single asset.
- BTC Grad-CAM follows the Grad-CAM method plan and uses class logits.

Generated visual checks:
- Sample images: `stage2_btc_extension/reports/figures/sample_images/`
- Smoke Grad-CAM report copy:
  `stage2_btc_extension/reports/figures/gradcam/stage2_i20_ohlc_ma_vb_r20_seed_42_test_gradcam.png`

Image-generation correction after visual review:
- The first implementation reserved the volume panel even for `ohlc` and
  `ohlc_ma`, which left blank rows at the bottom of no-volume images.
- This was corrected so no-volume specs use the full image height for price
  scaling, consistent with `stage2_image_generation_plan.md`.
- Volume specs still use the upper price area and lower volume area.

Limit:
- The local smoke run verifies code execution only.
- Full Stage 2 thesis results require running the Kaggle one-cell runner.
