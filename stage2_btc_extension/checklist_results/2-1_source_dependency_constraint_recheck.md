# 2-1 Source, Dependency, and Constraint Re-check

## English

Status: complete.

This checklist item did not implement code. It re-checked the sources and
Stage 1 dependencies before Stage 2 BTC implementation begins.

Sources checked:
- `PLAN.md`
- `stage2_btc_extension/checklist.md`
- `stage2_btc_extension/docs/stage2_pipeline.md`
- `stage2_btc_extension/docs/source_map.md`
- `자료조사/Re-image 요약.md`
- `자료조사/Grad-CAM요약.md`
- `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`
- `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`
- `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`
- `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`

Main finding:
- Stage 2 can reuse the Stage 1 Re-image/Stock_CNN-style pipeline logic, but not
  every Stage 1 file as-is.
- The current Stage 1 model is `StockCNNI20`. It is fixed to input shape
  `(batch, 1, 64, 60)` and classifier input size `46,080`.
- BTC `I20` can reuse this CNN core directly after BTC image generation is
  implemented.
- BTC `I5` and `I60` require Stage-1/Stock_CNN-style model variants or a model
  factory. We should not silently force those windows through the I20 model.

Fixed Stage 2 constraints:
- Start with BTC OHLCV only. News/LLM conditioning remains Stage 4.
- Use the same binary rule: `future R-day return > 0 -> label 1`.
- Use chronological split to avoid look-ahead leakage.
- Fit pixel normalization on training images only.
- Keep default batch size at the paper value `128`.
- Do not use stock cross-sectional H-L decile portfolios for BTC.
- Generate Grad-CAM for BTC baseline runs.

Reusable pieces from Stage 1:
- CNN block/order and training defaults for the I20 baseline.
- Train-only pixel normalization principle.
- Prediction output pattern: date, future return, label, logits, probabilities,
  predicted class, correctness.
- Grad-CAM method: target pre-softmax logit, conv activation and gradient,
  spatial gradient average, ReLU weighted sum, input-size upsampling.

Pieces that must be rewritten for BTC:
- OHLCV loader and row cleaning.
- BTC image generator for `I5`, `I20`, and `I60`.
- BTC future-return construction.
- BTC chronological split.
- BTC prediction metadata schema.
- BTC trading metrics and single-asset backtests.
- BTC Grad-CAM sample selection by date/index instead of stock metadata.

Next checklist:
- 2-2 BTC OHLCV data audit.
- Confirm exact Kaggle input path, CSV filename, columns, timestamp format,
  daily frequency, missing values, duplicates, date range, and volume usability.

## 한국어

상태: 완료.

이번 체크리스트에서는 코드를 구현하지 않았습니다. Stage 2 BTC 구현에 들어가기 전에
근거 자료와 Stage 1 의존성을 다시 확인했습니다.

확인한 source:
- `PLAN.md`
- `stage2_btc_extension/checklist.md`
- `stage2_btc_extension/docs/stage2_pipeline.md`
- `stage2_btc_extension/docs/source_map.md`
- `자료조사/Re-image 요약.md`
- `자료조사/Grad-CAM요약.md`
- `stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py`
- `stage1_reimage_reproduction/src/stage1_reimage/data/label_split.py`
- `stage1_reimage_reproduction/src/stage1_reimage/evaluation/prediction.py`
- `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`

핵심 확인:
- Stage 2는 Stage 1의 Re-image/Stock_CNN식 pipeline 원칙을 재사용할 수 있지만,
  Stage 1 파일을 전부 그대로 가져오면 안 됩니다.
- 현재 Stage 1 model은 `StockCNNI20`입니다. input shape `(batch, 1, 64, 60)`과
  classifier input size `46,080`에 고정되어 있습니다.
- BTC `I20`은 BTC image generation이 구현되면 이 CNN core를 직접 재사용할 수 있습니다.
- BTC `I5`와 `I60`은 Stage-1/Stock_CNN식 model variant 또는 model factory가 필요합니다.
  이 두 window를 I20 model에 억지로 넣지 않습니다.

Stage 2 고정 제약:
- 우선 BTC OHLCV만 사용합니다. News/LLM conditioning은 Stage 4입니다.
- label rule은 동일하게 `future R-day return > 0 -> label 1`입니다.
- look-ahead leakage 방지를 위해 시간순 split을 사용합니다.
- pixel normalization은 training image에서만 fit합니다.
- 기본 batch size는 논문값 `128`을 유지합니다.
- BTC에서는 stock cross-sectional H-L decile portfolio를 사용하지 않습니다.
- BTC baseline run에서도 Grad-CAM을 생성합니다.

Stage 1에서 재사용할 부분:
- I20 baseline의 CNN block/order와 training default.
- train-only pixel normalization 원칙.
- prediction output 패턴: date, future return, label, logits, probabilities,
  predicted class, correctness.
- Grad-CAM 방법: target pre-softmax logit, conv activation과 gradient,
  gradient spatial average, ReLU weighted sum, input-size upsampling.

BTC용으로 다시 작성해야 하는 부분:
- OHLCV loader와 row cleaning.
- `I5`, `I20`, `I60` BTC image generator.
- BTC future-return construction.
- BTC chronological split.
- BTC prediction metadata schema.
- BTC trading metric과 single-asset backtest.
- stock metadata가 아니라 BTC date/index 기준 Grad-CAM sample selection.

다음 체크리스트:
- 2-2 BTC OHLCV data audit.
- Kaggle input path, CSV filename, column, timestamp format, daily frequency,
  missing value, duplicate, date range, volume 사용 가능성을 확인합니다.
