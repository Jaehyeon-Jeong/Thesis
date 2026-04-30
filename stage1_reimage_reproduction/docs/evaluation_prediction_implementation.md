# 1-I7 Evaluation and Prediction-output Implementation

## English

Status:
- Completed on 2026-05-01.

Purpose:
- Convert trained Stage 1 CNN checkpoints into reproducible prediction CSVs.
- Save classification metrics, majority-class baseline comparison, probability
  diagnostics, and prediction-return correlation diagnostics.
- Support both single-seed prediction files and paper-style averaged probability
  files across multiple seeds.

Implemented files:
- `src/stage1_reimage/evaluation/__init__.py`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/evaluate_stage1_predictions.py`
- Updated `src/stage1_reimage/config.py`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Implemented behavior:
- Loads `best.pt` or a user-specified checkpoint.
- Restores the checkpoint's train-only pixel normalization metadata.
- Builds deterministic evaluation dataloaders for `train`, `validation`, or
  `test` splits.
- Keeps the CNN forward output as logits.
- Applies `softmax(logits, dim=1)` only in evaluation.
- Saves `prob_down`, `prob_up`, logits, predicted class, label, target return,
  correctness, and row metadata.
- Uses `prob_up >= 0.5` as the default prediction rule. The exact tie rule is
  an implementation convention, not a separately reported paper detail.
- Computes accuracy, majority-class accuracy, precision, recall, F1, ROC AUC,
  average precision, Brier score, log loss, and confusion matrix.
- Computes global and date-wise Pearson/Spearman correlation between `prob_up`
  and `target_return`.
- Averages seed-level probabilities for full paper-style runs; it does not
  average logits.

Seed-level command:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_local.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split validation \
  --max-rows 4
```

Averaged-probability command:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_local.yaml \
  --horizon stage1_i20_r20 \
  --split validation \
  --average-seed-predictions 42
```

Validation result:
- `python -m compileall src scripts` passed.
- `python scripts/check_scaffold.py --config configs/env_local.yaml` passed.
- Local smoke seed-level prediction export passed.
- Local smoke averaged-probability export passed.
- Smoke details:
  - horizon: `stage1_i20_r20`
  - split: `validation`
  - run seed: `42`
  - rows: `4`
  - accuracy: `0.25`
  - positive rate: `0.50`
  - predicted positive rate: `0.25`

Written smoke outputs:
- `outputs/predictions/stage1_i20_r20/seed_42/validation_predictions.csv`
- `outputs/metrics/stage1_i20_r20/seed_42/validation_metrics.json`
- `outputs/metrics/stage1_i20_r20/seed_42/validation_correlation_metrics.json`
- `outputs/predictions/stage1_i20_r20/averaged/validation_predictions.csv`
- `outputs/metrics/stage1_i20_r20/averaged/validation_metrics.json`
- `outputs/metrics/stage1_i20_r20/averaged/validation_correlation_metrics.json`

Scope limits:
- Smoke metrics are not reproduction results.
- Large prediction CSVs and metric outputs under `outputs/` are excluded from
  GitHub.
- Final report tables under `reports/tables/` remain a later report assembly
  step after full Kaggle runs exist.
- Portfolio/decile H-L tables are not implemented in this gate because the
  exact paper-style convention still needs a separate source re-check before
  final reporting.
- Grad-CAM sample selection will use these prediction files, but Grad-CAM code
  itself remains `1-I8`.

## 한국어

상태:
- 2026-05-01 완료.

목적:
- 학습된 Stage 1 CNN checkpoint를 재현 가능한 prediction CSV로 변환합니다.
- classification metric, majority-class baseline 비교, probability diagnostic,
  prediction-return correlation diagnostic을 저장합니다.
- seed별 prediction file과 여러 seed의 probability 평균 파일을 모두 지원합니다.

구현한 파일:
- `src/stage1_reimage/evaluation/__init__.py`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/evaluate_stage1_predictions.py`
- `src/stage1_reimage/config.py` 업데이트
- `configs/env_local.yaml` 업데이트
- `configs/env_kaggle.yaml` 업데이트

구현한 동작:
- `best.pt` 또는 사용자가 지정한 checkpoint를 불러옵니다.
- checkpoint에 저장된 train-only pixel normalization metadata를 복원합니다.
- `train`, `validation`, `test` split에 대해 deterministic evaluation
  dataloader를 만듭니다.
- CNN forward output은 logits 그대로 유지합니다.
- evaluation에서만 `softmax(logits, dim=1)`를 적용합니다.
- `prob_down`, `prob_up`, logits, predicted class, label, target return,
  correctness, row metadata를 저장합니다.
- 기본 prediction rule은 `prob_up >= 0.5`입니다. 정확히 0.5일 때 class 1로
  보내는 tie rule은 논문에 별도로 보고된 값이 아니라 구현상 convention입니다.
- accuracy, majority-class accuracy, precision, recall, F1, ROC AUC, average
  precision, Brier score, log loss, confusion matrix를 계산합니다.
- `prob_up`과 `target_return`의 global/date-wise Pearson/Spearman correlation을
  계산합니다.
- full paper-style run에서는 seed별 softmax probability를 평균합니다. logits는
  평균하지 않습니다.

Seed별 실행 command:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_local.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split validation \
  --max-rows 4
```

Probability 평균 command:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_local.yaml \
  --horizon stage1_i20_r20 \
  --split validation \
  --average-seed-predictions 42
```

검증 결과:
- `python -m compileall src scripts` 통과.
- `python scripts/check_scaffold.py --config configs/env_local.yaml` 통과.
- local smoke seed-level prediction export 통과.
- local smoke averaged-probability export 통과.
- Smoke detail:
  - horizon: `stage1_i20_r20`
  - split: `validation`
  - run seed: `42`
  - rows: `4`
  - accuracy: `0.25`
  - positive rate: `0.50`
  - predicted positive rate: `0.25`

생성된 smoke output:
- `outputs/predictions/stage1_i20_r20/seed_42/validation_predictions.csv`
- `outputs/metrics/stage1_i20_r20/seed_42/validation_metrics.json`
- `outputs/metrics/stage1_i20_r20/seed_42/validation_correlation_metrics.json`
- `outputs/predictions/stage1_i20_r20/averaged/validation_predictions.csv`
- `outputs/metrics/stage1_i20_r20/averaged/validation_metrics.json`
- `outputs/metrics/stage1_i20_r20/averaged/validation_correlation_metrics.json`

범위 제한:
- smoke metric은 reproduction result가 아닙니다.
- `outputs/` 아래 대용량 prediction CSV와 metric output은 GitHub에서 제외합니다.
- `reports/tables/`의 최종 보고 table은 full Kaggle run 이후 report assembly
  단계에서 만듭니다.
- portfolio/decile H-L table은 이 gate에서 구현하지 않았습니다. 최종 보고 전에
  정확한 논문식 convention을 별도로 다시 확인해야 합니다.
- Grad-CAM sample selection은 이 prediction file을 사용하지만, Grad-CAM code
  자체는 `1-I8`에서 구현합니다.
