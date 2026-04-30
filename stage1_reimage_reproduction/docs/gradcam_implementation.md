# 1-I8 Grad-CAM Implementation

## English

Status:
- Completed on 2026-05-01.

Purpose:
- Implement Re-image Figure 13-style Grad-CAM outputs for the Stage 1 I20 CNN
  baseline.
- Make clear that this is a class-discriminative heatmap from activations and
  gradients, not a raw feature map.

Sources:
- Root `PLAN.md`, Stage 1 Grad-CAM rule.
- `docs/gradcam_plan.md`.
- `자료조사/Grad-CAM요약.md`, method pp.4-6 mapping.
- `자료조사/Grad-CAM.pdf`.
- `자료조사/Re-image 요약.md`, interpretation section and Figure 13 mapping.
- `자료조사/Xiu-Re-Imagining-Price-Trends.pdf`.
- `lich99/Stock_CNN`, commit `415e2acf2a5013afca67e383acd3edc61fced840`.

Implemented files:
- `src/stage1_reimage/interpretability/__init__.py`
- `src/stage1_reimage/interpretability/gradcam.py`
- `scripts/generate_stage1_gradcam.py`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

Behavior:
- Selects Up and Down samples from prediction CSV.
- Uses predicted class as the Grad-CAM target class.
- Uses pre-softmax logits as the target score.
- Hooks `layer1_conv`, `layer2_conv`, and `layer3_conv`.
- Computes `alpha_k^c` by spatially averaging gradients.
- Applies `ReLU(sum_k alpha_k^c A^k)`.
- Upsamples every heatmap to `(64, 60)`.
- Saves heatmap `.npy`, `samples.csv`, `summary.json`, and a Figure 13-style PNG.

Validation:
- `python -m compileall src scripts`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`

Limitations:
- The local validation figure is a smoke output, not a reproduction result.
- The true Figure 13-style output should be generated after Kaggle full runs
  using `split=test`, `year=2019`, and `samples-per-class=10`.

## 한국어

상태:
- 2026-05-01 완료.

목적:
- Stage 1 I20 CNN baseline에 대해 Re-image Figure 13 형식의 Grad-CAM 산출물을
  생성할 수 있게 구현했습니다.
- 이 산출물이 raw feature map이 아니라 activation과 gradient로 만든
  class-discriminative heatmap임을 코드와 문서에 명확히 남겼습니다.

확인한 근거:
- 루트 `PLAN.md`의 Stage 1 Grad-CAM 규칙.
- `docs/gradcam_plan.md`.
- `자료조사/Grad-CAM요약.md`, 방법론 pp.4-6 mapping.
- `자료조사/Grad-CAM.pdf`.
- `자료조사/Re-image 요약.md`, 해석 섹션과 Figure 13 mapping.
- `자료조사/Xiu-Re-Imagining-Price-Trends.pdf`.
- `lich99/Stock_CNN`, commit `415e2acf2a5013afca67e383acd3edc61fced840`.

구현 파일:
- `src/stage1_reimage/interpretability/__init__.py`
- `src/stage1_reimage/interpretability/gradcam.py`
- `scripts/generate_stage1_gradcam.py`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

동작:
- prediction CSV에서 Up/Down 예측 sample을 고릅니다.
- predicted class를 Grad-CAM target class로 사용합니다.
- softmax 이후 probability가 아니라 softmax 이전 logit을 target score로 사용합니다.
- `layer1_conv`, `layer2_conv`, `layer3_conv`에 hook을 겁니다.
- gradient spatial average로 `alpha_k^c`를 계산합니다.
- `ReLU(sum_k alpha_k^c A^k)`로 heatmap을 만듭니다.
- 모든 heatmap을 `(64, 60)`으로 upsampling합니다.
- heatmap `.npy`, `samples.csv`, `summary.json`, Figure 13-style PNG를 저장합니다.

검증:
- `python -m compileall src scripts`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`

제한사항:
- local validation figure는 smoke output이며 재현 결과가 아닙니다.
- 진짜 Figure 13-style output은 Kaggle full run 이후 `split=test`, `year=2019`,
  `samples-per-class=10`으로 생성해야 합니다.
