# Stage 2 Pipeline

## English

Purpose:
- Apply the Stage 1 Re-image/Stock_CNN-style pipeline to BTC as a single asset.

Stage 2 does not redefine the research design:
- Stage 1 remains the stock Re-image reproduction gate.
- Stage 2 changes the asset/data source to BTC OHLCV.
- Stage 2 keeps the CNN core aligned with the Stage 1 implementation unless a
  later checklist item records an explicit reason to change it.

Pipeline:
1. Audit BTC OHLCV file paths, columns, date range, frequency, missing values,
   duplicates, and volume units.
2. Generate BTC chart images for windows `5`, `20`, and `60`, and compare four
   image specifications: `OHLC`, `OHLC+Volume`, `OHLC+MA`, and
   `OHLC+MA+Volume`.
3. Use the same binary label rule:
   `label = 1 if future R-day return > 0 else 0`.
4. Construct BTC future returns from close prices:
   `Close_{t+R} / Close_t - 1`.
5. Use chronological train/validation/test split with label-horizon purging to
   avoid look-ahead leakage.
6. Fit pixel normalization on train images only.
7. Train BTC CNN baseline with paper batch size `128` by default.
8. Save prediction CSV with date, label, future return, logits, probabilities,
   predicted class, and correctness.
9. Compute classification metrics.
10. Compute BTC time-series trading metrics.
11. Generate BTC Grad-CAM figures.

Default Stage 2 batch policy:
- Use `batch_size=128` by default.
- The BTC dataset is much smaller than the Stage 1 public stock shard, so the
  Stage 1 fast diagnostic batch `1024` is not the Stage 2 default.

Evaluation policy:
- BTC is a single asset.
- Do not use cross-sectional H-L decile portfolio evaluation.
- Use classification metrics and time-series trading strategies:
  long/flat, long/short, annualized return, annualized volatility, Sharpe,
  max drawdown, turnover, and transaction-cost-adjusted return.

Dependency:
- Stage 2 preparation may proceed while Stage 1 Kaggle runs are executing.
- Stage 2 final comparison tables wait for Stage 1 full outputs.

2-1 dependency re-check:
- The current Stage 1 model class is `StockCNNI20`, which is fixed to input
  shape `(batch, 1, 64, 60)` and classifier input size `46,080`.
- Therefore, BTC `I20` can reuse the Stage 1 CNN core directly, but BTC `I5`
  and `I60` require Stage-1/Stock_CNN-style variants or a model factory.
- Stage 1 stock-specific metadata and `.dat/.feather` label handling do not
  transfer directly to BTC. BTC must construct image rows and future returns
  from OHLCV data.
- Stage 1 Grad-CAM logic is reusable as a method, but BTC sample selection and
  output metadata must be BTC-specific.

2-2 data audit result:
- Local BTC data folder: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV`.
- Baseline source file: `btc_1d_data_2018_to_2025.csv`.
- The audited daily file has `2997` rows, `12` columns, no missing OHLCV values,
  no duplicate dates, no invalid OHLCV rows, and no missing calendar days.
- Date range is `2018-01-01` to `2026-03-16`.
- No daily resampling is needed for Stage 2 baseline.
- All `I5`, `I20`, and `I60` windows are feasible.

2-3 image-generation decision:
- BTC OHLCV has no MA column, so Stage 2 computes simple moving averages from
  BTC close prices.
- `I5` uses 5-day SMA, `I20` uses 20-day SMA, and `I60` uses 60-day SMA.
- MA uses only current and past close prices.
- Four image specs are fixed: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`.
- The four specs are compared on the same eligible sample dates within each
  window/horizon setting.

2-4 label/split/normalization decision:
- BTC future return is constructed as `Close_{t+R} / Close_t - 1`.
- Label is `1` if future return is positive, else `0`; exact zero belongs to
  class `0`.
- Primary reporting is capped at `2024-12-31`; 2025-2026 remains optional
  later holdout.
- Chronological split uses train `2018-2020`, validation `2021`, and test
  `2022-2024`.
- Samples are purged at split ends unless `label_end_date <= split_signal_end`.
- Pixel normalization is fit on training images only and stored per experiment
  tuple `(image_window, image_spec, return_horizon)`.

## 한국어

목적:
- Stage 1의 Re-image/Stock_CNN식 파이프라인을 BTC 단일 자산에 적용합니다.

Stage 2는 연구 설계를 다시 정의하지 않습니다.
- Stage 1은 stock Re-image reproduction gate로 유지됩니다.
- Stage 2는 asset/data source를 BTC OHLCV로 바꿉니다.
- Stage 2는 명시적 이유가 checklist에 기록되지 않는 한 Stage 1 CNN core를 유지합니다.

파이프라인:
1. BTC OHLCV 파일 경로, column, date range, frequency, missing value, duplicate,
   volume 단위를 audit합니다.
2. window `5`, `20`, `60`에 대해 BTC chart image를 생성하고, 네 가지 image
   specification인 `OHLC`, `OHLC+Volume`, `OHLC+MA`, `OHLC+MA+Volume`을 비교합니다.
3. 같은 binary label rule을 사용합니다:
   `future R-day return > 0`이면 `label=1`, 아니면 `0`.
4. BTC future return은 close price에서 `Close_{t+R} / Close_t - 1`로 만듭니다.
5. look-ahead leakage 방지를 위해 label-horizon purge가 있는 시간순
   train/validation/test split을 사용합니다.
6. pixel normalization은 train image에서만 fit합니다.
7. BTC CNN baseline은 기본적으로 논문 batch size `128`로 학습합니다.
8. date, label, future return, logits, probability, predicted class, correctness가
   들어 있는 prediction CSV를 저장합니다.
9. classification metric을 계산합니다.
10. BTC time-series trading metric을 계산합니다.
11. BTC Grad-CAM 그림을 생성합니다.

Stage 2 기본 batch 정책:
- 기본값은 `batch_size=128`입니다.
- BTC dataset은 Stage 1 public stock shard보다 훨씬 작으므로 Stage 1 fast diagnostic의
  `1024` batch를 Stage 2 기본값으로 쓰지 않습니다.

평가 정책:
- BTC는 단일 자산입니다.
- cross-sectional H-L decile portfolio 평가는 사용하지 않습니다.
- classification metric과 time-series trading strategy를 사용합니다:
  long/flat, long/short, annualized return, annualized volatility, Sharpe,
  max drawdown, turnover, transaction-cost-adjusted return.

의존성:
- Stage 1 Kaggle run이 실행되는 동안 Stage 2 준비는 진행할 수 있습니다.
- Stage 2 최종 비교표는 Stage 1 full output 이후 작성합니다.

2-1 dependency 재확인:
- 현재 Stage 1 model class는 `StockCNNI20`이고, input shape `(batch, 1, 64, 60)`과
  classifier input size `46,080`에 고정되어 있습니다.
- 따라서 BTC `I20`은 Stage 1 CNN core를 직접 재사용할 수 있지만, BTC `I5`와 `I60`은
  Stage-1/Stock_CNN식 model variant 또는 model factory가 필요합니다.
- Stage 1의 stock-specific metadata와 `.dat/.feather` label handling은 BTC로 직접
  넘어오지 않습니다. BTC는 OHLCV에서 image row와 future return을 새로 구성해야 합니다.
- Stage 1 Grad-CAM 로직은 방법론으로 재사용할 수 있지만, BTC sample selection과 output
  metadata는 BTC용으로 바꿔야 합니다.

2-2 data audit 결과:
- local BTC data folder: `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV`.
- baseline source file: `btc_1d_data_2018_to_2025.csv`.
- audit한 daily file은 `2997` rows, `12` columns이고, OHLCV 결측, duplicate date,
  invalid OHLCV row, missing calendar day가 없습니다.
- date range는 `2018-01-01`부터 `2026-03-16`까지입니다.
- Stage 2 baseline에서는 daily resampling이 필요 없습니다.
- `I5`, `I20`, `I60` window 모두 가능합니다.

2-3 image-generation 결정:
- BTC OHLCV에는 MA column이 없으므로 Stage 2에서 BTC close price로 simple moving
  average를 계산합니다.
- `I5`는 5-day SMA, `I20`은 20-day SMA, `I60`은 60-day SMA를 사용합니다.
- MA는 현재와 과거 close price만 사용합니다.
- 네 가지 image spec은 `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`로 고정합니다.
- 같은 window/horizon setting 안에서는 네 spec을 같은 eligible sample date에서
  비교합니다.

2-4 label/split/normalization 결정:
- BTC future return은 `Close_{t+R} / Close_t - 1`로 만듭니다.
- label은 future return이 양수이면 `1`, 아니면 `0`입니다. 정확히 0이면 class `0`입니다.
- 기본 보고 기간은 `2024-12-31`까지로 제한하고, 2025-2026은 optional later holdout으로
  남깁니다.
- 시간순 split은 train `2018-2020`, validation `2021`, test `2022-2024`입니다.
- 각 split 끝에서 `label_end_date <= split_signal_end`를 만족하지 않는 sample은 purge합니다.
- pixel normalization은 train image에서만 fit하고, `(image_window, image_spec,
  return_horizon)` experiment tuple별로 따로 저장합니다.
