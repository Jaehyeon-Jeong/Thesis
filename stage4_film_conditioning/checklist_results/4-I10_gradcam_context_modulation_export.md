# 4-I10. Grad-CAM Plus Context/Gate/Gamma/Beta Export

## English

Status: complete.

Implemented files:
- `src/stage4_film/interpretability/__init__.py`
- `src/stage4_film/interpretability/gradcam_context.py`
- `scripts/generate_stage4_gradcam_context.py`

Purpose:
- Stage 2 Grad-CAM explains `model(image)`.
- Stage 4 models require `model(image, context)`.
- This step keeps the Re-image/Figure-13-style Grad-CAM convention and adds
  the missing Stage 4 interpretation artifacts: context values, gate values,
  gamma values, and beta values.

Exported artifacts:
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/btc_context_gradcam_{split}_{N}perclass.png`
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/samples.csv`
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/modulation_summary.csv`
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/modulation_values.json`

What each file means:
- `btc_context_gradcam_*.png`: original chart row plus layer-wise Grad-CAM rows,
  split into Predicted Up and Predicted Down panels.
- `samples.csv`: selected Grad-CAM samples plus prediction metadata, context
  values, and compact modulation summaries.
- `modulation_summary.csv`: compact per-sample statistics for context embedding,
  gate, gamma, beta, and delta-gamma.
- `modulation_values.json`: full arrays for context, context embedding, gate,
  gamma, beta, and top/bottom channel summaries.

Method-specific export:
- `concat`: context values and context embedding summary.
- `gating`: context values, context embedding, raw gate, final gate, top/bottom
  gate channels.
- `film_gamma`: context values, context embedding, block-wise gamma and
  delta-gamma.
- `film_full`: context values, context embedding, block-wise gamma, beta, and
  delta-gamma.

Grad-CAM target rule:
- The target score is the predicted-class pre-softmax logit.
- `concat` and `gating` use convolution target layers.
- `film_gamma` and `film_full` use post-FiLM target modules so the heatmap is
  tied to the conditioned feature map.
- Tensor-level gradient hooks are used instead of `register_full_backward_hook`
  because post-FiLM outputs are immediately followed by in-place LeakyReLU in
  the Stock_CNN block.

Local validation:
- `python -m py_compile` passed for the new interpretability module and script.
- `concat` local smoke checkpoint generated figure, samples CSV, modulation
  summary CSV, and modulation values JSON.
- `film_gamma` local smoke checkpoint generated figure, samples CSV, modulation
  summary CSV, and modulation values JSON.

Boundary:
- This step does not define the final output checker.
- `4-I11` should add a local/Kaggle smoke output check that verifies
  checkpoint, predictions, classification metrics, trading metrics, Grad-CAM,
  samples, modulation summary, and modulation JSON together.

## 한국어

상태: 완료.

구현 파일:
- `src/stage4_film/interpretability/__init__.py`
- `src/stage4_film/interpretability/gradcam_context.py`
- `scripts/generate_stage4_gradcam_context.py`

목적:
- Stage 2 Grad-CAM은 `model(image)`를 설명합니다.
- Stage 4 모델은 `model(image, context)`를 사용합니다.
- 이 단계는 Re-image/Figure-13 스타일 Grad-CAM convention을 유지하면서,
  Stage 4에서 필요한 context 값, gate 값, gamma 값, beta 값을 같이 저장합니다.

저장 산출물:
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/btc_context_gradcam_{split}_{N}perclass.png`
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/samples.csv`
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/modulation_summary.csv`
- `outputs/stage4/figures/{experiment}/seed_{seed}/gradcam/{split}/modulation_values.json`

각 파일의 의미:
- `btc_context_gradcam_*.png`: original chart row와 layer별 Grad-CAM row를
  Predicted Up / Predicted Down panel로 나눠 보여줍니다.
- `samples.csv`: 선택된 Grad-CAM sample의 prediction metadata, context 값,
  compact modulation summary를 같이 담습니다.
- `modulation_summary.csv`: context embedding, gate, gamma, beta,
  delta-gamma의 sample별 통계값입니다.
- `modulation_values.json`: context, context embedding, gate, gamma, beta의
  full array와 top/bottom channel summary를 저장합니다.

모델별 export:
- `concat`: context 값과 context embedding summary.
- `gating`: context 값, context embedding, raw gate, final gate,
  top/bottom gate channel.
- `film_gamma`: context 값, context embedding, block-wise gamma와
  delta-gamma.
- `film_full`: context 값, context embedding, block-wise gamma, beta,
  delta-gamma.

Grad-CAM target 규칙:
- target score는 predicted-class pre-softmax logit입니다.
- `concat`, `gating`은 convolution target layer를 사용합니다.
- `film_gamma`, `film_full`은 post-FiLM module을 target으로 사용해서,
  conditioned feature map 기준 heatmap을 만듭니다.
- post-FiLM output 바로 뒤에 Stock_CNN의 in-place LeakyReLU가 오기 때문에,
  PyTorch `register_full_backward_hook` 대신 tensor-level gradient hook을
  사용했습니다.

Local validation:
- 새 interpretability module과 script의 `python -m py_compile` 통과.
- `concat` local smoke checkpoint에서 figure, samples CSV,
  modulation summary CSV, modulation values JSON 생성 확인.
- `film_gamma` local smoke checkpoint에서 figure, samples CSV,
  modulation summary CSV, modulation values JSON 생성 확인.

경계:
- 이 단계는 최종 output checker를 정의하지 않습니다.
- `4-I11`에서 checkpoint, prediction, classification metric, trading metric,
  Grad-CAM, samples, modulation summary, modulation JSON을 함께 확인하는
  local/Kaggle smoke output check를 추가해야 합니다.
