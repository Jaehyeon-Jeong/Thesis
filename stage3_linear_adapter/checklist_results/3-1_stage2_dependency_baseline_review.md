# 3-1 Stage 2 Dependency and Baseline-Output Review

## English

Status: done

Checked Stage 2 dependency and baseline output status.

Key decisions:
- Stage 3 inherits Stage 2 BTC data loading, image generation, label creation,
  split, normalization, classification metrics, trading metrics, and Grad-CAM.
- Stage 3 changes only the model adapter/head path.
- Stage 2 single-seed baseline grid exists for `36` experiments with seed `42`.
- Stage 2 five-seed rerun is still pending, so Stage 3 claims remain
  provisional until stability checks are available.

Output:
- `docs/stage2_dependency_baseline_review.md`

## 한국어

상태: 완료

Stage 2 dependency와 baseline output 상태를 확인했습니다.

핵심 결정:
- Stage 3는 Stage 2의 BTC data loading, image generation, label creation,
  split, normalization, classification metric, trading metric, Grad-CAM을
  그대로 상속합니다.
- Stage 3에서 바꾸는 것은 model adapter/head path뿐입니다.
- Stage 2 single-seed baseline grid는 seed `42` 기준 `36`개 실험이 있습니다.
- Stage 2 five-seed rerun은 아직 예정이므로, Stage 3 결론도 안정성 확인 전까지는
  provisional로 둡니다.

산출물:
- `docs/stage2_dependency_baseline_review.md`
