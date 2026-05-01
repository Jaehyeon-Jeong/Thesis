# Stage 4 Pipeline

## English

Stage 4 adds FiLM modulation to the fixed BTC chart-image pipeline.

Fixed inputs from earlier stages:
- BTC OHLCV source and chart image generation from Stage 2.
- I5/I20/I60 Stock_CNN variants from Stage 2.
- Stage 3 Linear result is a comparison point, not a dependency that changes
  the FiLM model.
- Evaluation, trading metrics, Grad-CAM, and Kaggle backup conventions remain
  the same as Stage 2/3.

Stage 4 flow:
1. Build BTC chart image sample.
2. Build or load condition vector for the selected condition-source track.
3. Use the FiLM generator to produce block/channel-level gamma and beta.
4. Run Stock_CNN block with FiLM inserted after BatchNorm.
5. Predict Up/Down.
6. Export metrics, trading metrics, Grad-CAM, and gamma/beta logs.

Current first target:
- Use the best Stage 2 single-seed configuration as the first diagnostic:
  `I60/R20/ohlc_ma_vb`, seed `42`.
- This is only a first diagnostic. The Stage 4 conclusion requires broader
  single-seed and later five-seed runs.

## 한국어

Stage 4는 고정된 BTC chart-image pipeline에 FiLM modulation을 추가합니다.

이전 단계에서 고정해서 가져오는 것:
- Stage 2의 BTC OHLCV source와 chart image generation.
- Stage 2의 I5/I20/I60 Stock_CNN variant.
- Stage 3 Linear 결과는 비교 대상일 뿐, FiLM model을 바꾸는 dependency가 아닙니다.
- Evaluation, trading metric, Grad-CAM, Kaggle backup 규칙은 Stage 2/3와
  동일하게 유지합니다.

Stage 4 흐름:
1. BTC chart image sample을 만듭니다.
2. 선택한 condition-source track에 맞는 condition vector를 만들거나 불러옵니다.
3. FiLM generator가 block/channel별 gamma와 beta를 만듭니다.
4. BatchNorm 뒤에 FiLM을 삽입한 Stock_CNN block을 실행합니다.
5. Up/Down을 예측합니다.
6. metric, trading metric, Grad-CAM, gamma/beta log를 저장합니다.

현재 첫 목표:
- 첫 diagnostic은 Stage 2 single-seed best configuration으로 시작합니다:
  `I60/R20/ohlc_ma_vb`, seed `42`.
- 이것은 첫 diagnostic일 뿐입니다. Stage 4 결론은 더 넓은 single-seed run과
  나중의 five-seed run 이후에 작성합니다.
