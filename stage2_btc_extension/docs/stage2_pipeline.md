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
5. Use the paper-style split principle with label-horizon purging:
   `2018-2020` train/validation pool, 70/30 random train/validation split, and
   `2021-2024` chronological test holdout.
6. Fit pixel normalization on train images only.
7. Train BTC CNN baseline with paper batch size `128` by default.
8. Save prediction CSV with date, label, future return, logits, probabilities,
   predicted class, and correctness.
9. Compute classification metrics.
10. Compute BTC time-series trading metrics.
11. Generate BTC Grad-CAM figures.
12. Run full experiments through the Kaggle one-cell wrapper, one experiment
    tuple at a time.

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
- The default split follows the paper-style train/validation rule:
  `2018-2020` is the train/validation pool and is split 70/30 at random with
  seed `42`; `2021-2024` is the chronological test holdout.
- Samples are purged at train/test period ends unless
  `label_end_date <= split_signal_end`.
- Pixel normalization is fit on training images only and stored per experiment
  tuple `(image_window, image_spec, return_horizon)`.
- Because BTC is a single rolling time series, adjacent train/validation samples
  can overlap strongly. This is acceptable for the paper-aligned default, but a
  chronological-validation variant can be added later as a robustness check.

2-5 baseline CNN adaptation decision:
- Model selection is by image window, not return horizon.
- `I5` uses `stock_cnn_i5`, `I20` uses `stock_cnn_i20`, and `I60` uses
  `stock_cnn_i60`.
- The checked GitHub implementation is I20-specific, so I20 reuses the exact
  Stage 1/GitHub-style core while I5 and I60 require separate paper-targeted
  variants.
- All image specs remain one-channel grayscale images; MA and volume do not add
  channels.
- Default BTC baseline training starts from scratch, not from a stock
  checkpoint.

2-6 evaluation/trading decision:
- BTC cannot use the original paper's stock cross-sectional H-L decile
  portfolios.
- Classification metrics follow the Stage 1 prediction style: accuracy,
  precision, recall, F1, ROC AUC, average precision, Brier score, log loss,
  confusion matrix, majority-class baseline, and probability/return
  correlations.
- Calibration is reported as a 10-bin `prob_up` table.
- Trading metrics use BTC single-asset time-series strategies:
  `long_flat` and `long_short`.
- R-day signals are evaluated with overlapping-horizon daily backtests.
- Annualization uses 365 daily periods.
- Report gross metrics and configurable transaction-cost-adjusted metrics.

2-7 Grad-CAM decision:
- BTC Grad-CAM is required for every baseline run.
- The heatmap is not a raw feature map. It is a class-discriminative heatmap
  made from activation and gradient with respect to the selected class logit.
- Use pre-softmax logits as target scores.
- I5 shows 2 convolution-layer heatmap rows, I20 shows 3 rows, and I60 shows
  4 rows.
- Final report figures use 10 predicted Up and 10 predicted Down samples when
  available; quick smoke checks may use 2 per class.

2-8 Kaggle runner/output decision:
- Stage 2 follows the Stage 1 one-cell Kaggle wrapper pattern.
- The notebook wrapper copies code, patches config, and calls repo scripts; it
  does not contain the actual implementation logic.
- Run one experiment tuple at a time.
- Default strict Stage 2 uses batch size `128`, no mixed precision, no
  DataParallel, and no fast cuDNN unless explicitly recorded as a speed
  diagnostic.
- Large outputs remain in Kaggle/working outputs; GitHub only receives code,
  plans, configs, and small summary/report artifacts.

2-I0 implementation readiness decision:
- Stage 2 is ready to move from planning to implementation.
- Stage 1 full output is still needed for final comparison tables, but it does
  not block Stage 2 code implementation.
- The next implementation item is `2-I1`, shared config/code scaffold.
- Implementation must proceed in checklist order from config scaffold to data
  loader, image generator, label/split, model runner, metrics, trading,
  Grad-CAM, smoke test, full Kaggle run, and report outputs.

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
5. label-horizon purge가 있는 논문식 split 원칙을 사용합니다:
   `2018-2020` train/validation pool, pool 내부 70/30 random train/validation
   split, `2021-2024` chronological test holdout.
6. pixel normalization은 train image에서만 fit합니다.
7. BTC CNN baseline은 기본적으로 논문 batch size `128`로 학습합니다.
8. date, label, future return, logits, probability, predicted class, correctness가
   들어 있는 prediction CSV를 저장합니다.
9. classification metric을 계산합니다.
10. BTC time-series trading metric을 계산합니다.
11. BTC Grad-CAM 그림을 생성합니다.
12. Kaggle one-cell wrapper로 experiment tuple 하나씩 full experiment를 실행합니다.

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
- 기본 split은 논문식 train/validation rule을 따릅니다:
  `2018-2020`을 train/validation pool로 두고 seed `42`로 70/30 random split하며,
  `2021-2024`를 chronological test holdout으로 둡니다.
- train/test period 끝에서 `label_end_date <= split_signal_end`를 만족하지 않는 sample은 purge합니다.
- pixel normalization은 train image에서만 fit하고, `(image_window, image_spec,
  return_horizon)` experiment tuple별로 따로 저장합니다.
- BTC는 단일 rolling time series라서 인접 train/validation sample이 많이 겹칠 수
  있습니다. 이는 논문 방식에 맞춘 기본값에서는 허용하되, chronological validation은
  나중 robustness check로 추가할 수 있습니다.

2-5 baseline CNN adaptation 결정:
- model은 return horizon이 아니라 image window로 선택합니다.
- `I5`는 `stock_cnn_i5`, `I20`은 `stock_cnn_i20`, `I60`은 `stock_cnn_i60`을
  사용합니다.
- 확인한 GitHub implementation은 I20 전용이므로 I20은 Stage 1/GitHub식 core를
  그대로 재사용하고, I5와 I60은 논문 target에 맞춘 별도 variant를 구현합니다.
- 모든 image spec은 1-channel grayscale image입니다. MA와 volume은 channel을
  추가하지 않습니다.
- 기본 BTC baseline은 stock checkpoint transfer 없이 from scratch로 학습합니다.

2-6 evaluation/trading 결정:
- BTC는 원논문의 stock cross-sectional H-L decile portfolio를 사용할 수 없습니다.
- Classification metric은 Stage 1 prediction style을 따릅니다: accuracy,
  precision, recall, F1, ROC AUC, average precision, Brier score, log loss,
  confusion matrix, majority-class baseline, probability/return correlation.
- Calibration은 `prob_up` 10-bin table로 보고합니다.
- Trading metric은 BTC 단일 자산 time-series strategy인 `long_flat`,
  `long_short`를 사용합니다.
- R-day signal은 overlapping-horizon daily backtest로 평가합니다.
- annualization은 365 daily periods를 사용합니다.
- gross metric과 configurable transaction-cost-adjusted metric을 모두 보고합니다.

2-7 Grad-CAM 결정:
- 모든 BTC baseline run에서 BTC Grad-CAM은 필수입니다.
- Heatmap은 raw feature map이 아닙니다. 선택한 class logit에 대한 activation과
  gradient로 만든 class-discriminative heatmap입니다.
- target score는 softmax 이전 logit을 사용합니다.
- I5는 convolution-layer heatmap row 2개, I20은 3개, I60은 4개를 표시합니다.
- 최종 보고 figure는 가능한 경우 predicted Up 10개와 predicted Down 10개를 쓰고,
  빠른 smoke check는 class당 2개를 사용할 수 있습니다.

2-8 Kaggle runner/output 결정:
- Stage 2는 Stage 1의 one-cell Kaggle wrapper pattern을 따릅니다.
- Notebook wrapper는 code copy, config patch, repo script 호출만 담당하며 실제
  구현 로직은 담지 않습니다.
- Experiment tuple 하나씩 실행합니다.
- strict Stage 2 기본값은 batch size `128`, mixed precision off, DataParallel off,
  fast cuDNN off입니다. 속도 diagnostic으로 바꾸면 기록해야 합니다.
- 대용량 output은 Kaggle/working output에 두고, GitHub에는 code, plan, config,
  작은 summary/report artifact만 올립니다.

2-I0 implementation readiness 결정:
- Stage 2는 planning에서 implementation으로 넘어갈 준비가 됐습니다.
- Stage 1 full output은 최종 비교표에는 필요하지만 Stage 2 코드 구현을 막지는
  않습니다.
- 다음 구현 항목은 `2-I1`, shared config/code scaffold입니다.
- 구현은 checklist 순서대로 진행합니다: config scaffold, data loader, image
  generator, label/split, model runner, metric, trading, Grad-CAM, smoke test,
  full Kaggle run, report output.

2-I1 shared config/code scaffold 결정:
- `configs/env_local.yaml`과 `configs/env_kaggle.yaml`을 같은 schema로 만들었습니다.
- local/Kaggle 차이는 path와 runtime config 값으로 처리하고, Python 구현은 공통
  `src/stage2_btc/`를 사용합니다.
- Stage 2 기본 batch size는 `128`로 고정했습니다.
- strict Stage 2 baseline 기본값은 mixed precision off, DataParallel off입니다.
- `src/stage2_btc/config.py`, `paths.py`, `runtime.py`, `seed.py`를 추가했습니다.
- 이 항목은 새 논문 결정이 아니라 root `PLAN.md`와 Stage 1 helper style을 따르는
  구현 scaffold입니다.

2-I2 to 2-I10 implementation status:
- `2-I2`: BTC OHLCV loader implemented and verified on `2997` daily rows.
- `2-I3`: BTC image generator implemented. Local sample images are in
  `reports/figures/sample_images/`.
- `2-I4`: BTC label/split/normalization code implemented and verified for
  `I20/ohlc_ma_vb/R20`.
- `2-I5`: I5/I20/I60 Stock_CNN-style model variants and BTC baseline training
  runner implemented.
- `2-I6`: prediction CSV and classification metric export implemented.
- `2-I7`: BTC single-asset trading metric export implemented.
- `2-I8`: BTC Grad-CAM export implemented.
- `2-I9`: local smoke test completed for `I20/ohlc_ma_vb/R20`, seed `42`.
- `2-I10`: Kaggle one-cell full runner is ready. Actual full-run result must be
  produced in Kaggle by the user.
