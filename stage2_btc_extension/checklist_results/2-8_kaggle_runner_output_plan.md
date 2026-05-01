# 2-8 Kaggle Runner and Output Plan

## English

Status: complete.

This checklist item fixes the Stage 2 Kaggle execution and output policy.

Key decisions:
- Stage 2 follows the Stage 1 one-cell Kaggle wrapper pattern.
- The Kaggle cell only copies code/data, patches config, and calls repo scripts.
- The real implementation remains in `src/` and `scripts/`.
- Stage 2 runs one experiment tuple at a time.
- Default strict BTC baseline keeps `batch_size=128`.
- Mixed precision or DataParallel are not default Stage 2 settings.
- Outputs are separated into large Kaggle outputs and small report copies.
- GitHub receives planning docs, code/config, and small summary artifacts only.

Detailed plan:
- [Stage 2 Kaggle runner and output plan](../docs/stage2_kaggle_runner_output_plan.md)

Planned one-cell interface:
- [Kaggle Stage 2 BTC baseline one-cell draft](../notebooks/kaggle_stage2_btc_baseline_one_cell.md)

## 한국어

상태: 완료.

이번 체크리스트에서는 Stage 2 Kaggle 실행 방식과 output 정책을 고정했습니다.

핵심 결정:
- Stage 2는 Stage 1의 one-cell Kaggle wrapper 패턴을 따릅니다.
- Kaggle cell은 code/data 복사, config patch, repo script 호출만 담당합니다.
- 실제 구현은 `src/`와 `scripts/`에 둡니다.
- Stage 2는 experiment tuple 하나씩 실행합니다.
- strict BTC baseline 기본값은 `batch_size=128`입니다.
- Mixed precision이나 DataParallel은 Stage 2 기본값이 아닙니다.
- output은 대용량 Kaggle output과 작은 report copy로 분리합니다.
- GitHub에는 planning docs, code/config, 작은 summary artifact만 올립니다.

상세 계획:
- [Stage 2 Kaggle runner and output plan](../docs/stage2_kaggle_runner_output_plan.md)

예정 one-cell interface:
- [Kaggle Stage 2 BTC baseline one-cell draft](../notebooks/kaggle_stage2_btc_baseline_one_cell.md)
