# 1-3 Label Construction Plan

## English

Result:
- Fixed target columns: `Ret_5d`, `Ret_20d`, `Ret_60d`.
- Fixed label rule: `label = 1` if future return is greater than zero,
  otherwise `0`.
- Preserved return values and metadata for evaluation outputs.

Output:
- [label_construction_plan.md](../docs/label_construction_plan.md)

## 한국어

결과:
- target column을 `Ret_5d`, `Ret_20d`, `Ret_60d`로 고정했습니다.
- label rule은 future return이 0보다 크면 `1`, 아니면 `0`입니다.
- evaluation output을 위해 return 원값과 metadata를 보존합니다.

산출물:
- [label_construction_plan.md](../docs/label_construction_plan.md)
