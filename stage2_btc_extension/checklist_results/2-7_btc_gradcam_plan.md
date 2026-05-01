# 2-7 BTC Grad-CAM Plan

## English

Status: complete.

This checklist item fixes how BTC Grad-CAM figures will be generated and
reported.

Key decisions:
- BTC Grad-CAM is required for Stage 2 baseline outputs.
- The figure is a class-discriminative Grad-CAM heatmap, not a raw feature map.
- The target score is the pre-softmax class logit.
- Target layers depend on the BTC CNN image-window variant:
  - I5: 2 convolution blocks.
  - I20: 3 convolution blocks.
  - I60: 4 convolution blocks.
- Heatmaps are upsampled to the corresponding input image size.
- Final report figures use 10 predicted Up and 10 predicted Down samples when
  available.
- Quick smoke figures may use 2 per class.
- Later model comparisons should also save a fixed-date sample list for
  baseline/Linear/FiLM comparison.

Detailed plan:
- [Stage 2 BTC Grad-CAM plan](../docs/stage2_gradcam_plan.md)

## 한국어

상태: 완료.

이번 체크리스트에서는 BTC Grad-CAM figure를 어떻게 만들고 보고할지 고정했습니다.

핵심 결정:
- Stage 2 baseline output에는 BTC Grad-CAM이 필수입니다.
- 이 figure는 raw feature map이 아니라 class-discriminative Grad-CAM heatmap입니다.
- target score는 softmax 이전 class logit입니다.
- target layer는 BTC CNN image-window variant에 따라 달라집니다.
  - I5: convolution block 2개.
  - I20: convolution block 3개.
  - I60: convolution block 4개.
- Heatmap은 해당 input image size로 upsample합니다.
- 최종 보고 figure는 가능한 경우 predicted Up 10개와 predicted Down 10개를 사용합니다.
- 빠른 smoke figure는 class당 2개를 사용할 수 있습니다.
- 이후 baseline/Linear/FiLM 비교를 위해 fixed-date sample list도 저장합니다.

상세 계획:
- [Stage 2 BTC Grad-CAM plan](../docs/stage2_gradcam_plan.md)
