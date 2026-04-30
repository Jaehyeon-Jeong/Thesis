# 1-2 Data Loading Plan

## English

Result:
- Fixed `.dat` image loading as `uint8`, reshaped to `(64, 60)`, converted to
  model tensor `(batch, 1, 64, 60)`.
- Fixed `.feather` label loading and row alignment by `(year, local_row)`.
- Chose memmap/lazy image loading by default.

Output:
- [data_loading_plan.md](../docs/data_loading_plan.md)

## 한국어

결과:
- `.dat` image를 `uint8`로 읽고 `(64, 60)`으로 reshape한 뒤 model tensor
  `(batch, 1, 64, 60)`으로 변환하기로 했습니다.
- `.feather` label은 `(year, local_row)` 기준으로 image와 row alignment합니다.
- 기본 image loading은 memmap/lazy 방식입니다.

산출물:
- [data_loading_plan.md](../docs/data_loading_plan.md)
