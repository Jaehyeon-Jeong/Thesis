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
2. Generate BTC chart images for windows `5`, `20`, and `60`.
3. Use the same binary label rule:
   `label = 1 if future R-day return > 0 else 0`.
4. Use chronological train/validation/test split to avoid look-ahead leakage.
5. Fit pixel normalization on train images only.
6. Train BTC CNN baseline with paper batch size `128` by default.
7. Save prediction CSV with date, label, future return, logits, probabilities,
   predicted class, and correctness.
8. Compute classification metrics.
9. Compute BTC time-series trading metrics.
10. Generate BTC Grad-CAM figures.

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
2. window `5`, `20`, `60`에 대해 BTC chart image를 생성합니다.
3. 같은 binary label rule을 사용합니다:
   `future R-day return > 0`이면 `label=1`, 아니면 `0`.
4. look-ahead leakage 방지를 위해 시간순 train/validation/test split을 사용합니다.
5. pixel normalization은 train image에서만 fit합니다.
6. BTC CNN baseline은 기본적으로 논문 batch size `128`로 학습합니다.
7. date, label, future return, logits, probability, predicted class, correctness가
   들어 있는 prediction CSV를 저장합니다.
8. classification metric을 계산합니다.
9. BTC time-series trading metric을 계산합니다.
10. BTC Grad-CAM 그림을 생성합니다.

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
