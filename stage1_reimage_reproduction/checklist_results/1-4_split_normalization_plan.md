# 1-4 Split and Normalization Plan

## English

Result:
- Fixed train/validation pool as 1993-2000 and test period as 2001-2019.
- Fixed deterministic 70/30 train/validation split with split seed `42`.
- Fixed train-only pixel mean/std standardization per horizon.

Outputs:
- [split_normalization_plan.md](../docs/split_normalization_plan.md)
- [stage1_horizon_counts_by_period.csv](../reports/tables/stage1_horizon_counts_by_period.csv)

## 한국어

결과:
- train/validation pool은 1993-2000, test period는 2001-2019로 고정했습니다.
- split seed `42`로 deterministic 70/30 train/validation split을 사용합니다.
- horizon별 training subset에서만 pixel mean/std를 계산해 standardization합니다.

산출물:
- [split_normalization_plan.md](../docs/split_normalization_plan.md)
- [stage1_horizon_counts_by_period.csv](../reports/tables/stage1_horizon_counts_by_period.csv)
