# 3-4 Grad-CAM Comparison Plan

## English

Status: done

Defined Stage 3 Grad-CAM comparison rules.

Key decisions:
- Grad-CAM remains mandatory.
- Target score is each model's pre-softmax predicted-class logit.
- Hooks stay on the same CNN convolution layers as Stage 2.
- For baseline-vs-Linear comparison, use the same date/sample whenever
  possible and save predicted class, probability, label, correctness, and
  future return metadata.
- Quick figures use `2` predicted-up and `2` predicted-down examples.
- Report figures use `10` predicted-up and `10` predicted-down examples.

Output:
- `docs/gradcam_comparison_plan.md`

## 한국어

상태: 완료

Stage 3 Grad-CAM 비교 규칙을 정의했습니다.

핵심 결정:
- Grad-CAM은 Stage 3에서도 필수입니다.
- Target score는 각 model의 softmax 이전 predicted-class logit입니다.
- Hook은 Stage 2와 같은 CNN convolution layer에 겁니다.
- baseline-vs-Linear 비교에서는 가능하면 같은 date/sample을 사용하고,
  predicted class, probability, label, correctness, future return metadata를
  저장합니다.
- Quick figure는 predicted-up 2개와 predicted-down 2개를 사용합니다.
- Report figure는 predicted-up 10개와 predicted-down 10개를 사용합니다.

산출물:
- `docs/gradcam_comparison_plan.md`
