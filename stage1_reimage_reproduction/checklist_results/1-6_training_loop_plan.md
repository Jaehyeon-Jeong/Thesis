# 1-6 Training Loop Plan

## English

Result:
- Fixed CrossEntropyLoss on logits, Adam with learning rate `1e-5`, batch size
  `128`, dropout `0.5`, Xavier initialization, and validation-loss early stopping.
- Fixed full paper-style seeds as `[42, 43, 44, 45, 46]`.

Output:
- [training_loop_plan.md](../docs/training_loop_plan.md)

## 한국어

결과:
- logits에 CrossEntropyLoss, Adam learning rate `1e-5`, batch size `128`,
  dropout `0.5`, Xavier initialization, validation-loss early stopping을
  사용합니다.
- full paper-style seed는 `[42, 43, 44, 45, 46]`으로 고정했습니다.

산출물:
- [training_loop_plan.md](../docs/training_loop_plan.md)
