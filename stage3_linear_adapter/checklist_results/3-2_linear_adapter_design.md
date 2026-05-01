# 3-2 Linear Adapter Design and Insertion-Point Review

## English

Status: done

Reviewed the adapter insertion point and feature dimensions.

Key decisions:
- Keep Stage 2 convolution blocks unchanged.
- Do not use `Linear(feature_dim, feature_dim)` because it is too large:
  `I60` would require about `33.97B` weights.
- Use a config-driven bias-free Linear adapter/head:
  `feature -> Linear(feature_dim, adapter_dim, bias=False) -> Linear(adapter_dim, 2, bias=False)`.
- Default first comparison uses `adapter_dim=128`.
- This is an implementation choice, not a reported Re-image paper detail.

Output:
- `docs/linear_adapter_design.md`

## 한국어

상태: 완료

adapter 삽입 위치와 feature dimension을 확인했습니다.

핵심 결정:
- Stage 2 convolution block은 바꾸지 않습니다.
- `Linear(feature_dim, feature_dim)`는 너무 커서 사용하지 않습니다:
  `I60`은 약 `33.97B`개 weight가 필요합니다.
- config 기반 bias-free Linear adapter/head를 사용합니다:
  `feature -> Linear(feature_dim, adapter_dim, bias=False) -> Linear(adapter_dim, 2, bias=False)`.
- 첫 비교의 기본값은 `adapter_dim=128`입니다.
- 이 결정은 Re-image 논문에 보고된 값이 아니라 구현상 선택입니다.

산출물:
- `docs/linear_adapter_design.md`
