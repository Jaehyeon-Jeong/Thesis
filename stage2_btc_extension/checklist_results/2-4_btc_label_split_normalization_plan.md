# 2-4 BTC Label, Split, and Normalization Detail Plan

## English

Status: complete.

This checklist item fixes how BTC labels, chronological splits, leakage guards,
and train-only pixel normalization will work.

Key decisions:
- Future return:
  `future_return_{t,R} = Close_{t+R} / Close_t - 1`.
- Label:
  `1` if `future_return > 0`, otherwise `0`.
- Exact zero return is class `0`.
- BTC `R5/R20/R60` means 5/20/60 daily bars.
- Random split is not used for BTC.
- Primary reporting period is capped at `2024-12-31`; 2025-2026 remains an
  optional later holdout.

Chronological split:

| Split | Signal-date range |
| --- | --- |
| Train | `2018-01-01` to `2020-12-31` |
| Validation | `2021-01-01` to `2021-12-31` |
| Test | `2022-01-01` to `2024-12-31` |

Purge rule:

```text
label_end_date <= split_signal_end
```

This removes samples whose label horizon crosses into the next split.

Normalization:
- Use Stage 1-style scalar pixel normalization:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

- Fit only on training images.
- Do not use validation/test pixels.
- Save statistics per `(image_window, image_spec, return_horizon)` experiment.

Detailed plan:
- [Stage 2 BTC label/split/normalization plan](../docs/stage2_label_split_normalization_plan.md)

Split-count artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

## 한국어

상태: 완료.

이번 체크리스트에서는 BTC label, chronological split, leakage guard, train-only pixel
normalization 방식을 고정했습니다.

핵심 결정:
- Future return:
  `future_return_{t,R} = Close_{t+R} / Close_t - 1`.
- Label:
  `future_return > 0`이면 `1`, 아니면 `0`.
- 정확히 0인 return은 class `0`.
- BTC `R5/R20/R60`은 5/20/60 daily bar입니다.
- BTC에서는 random split을 사용하지 않습니다.
- 기본 보고 기간은 `2024-12-31`까지로 제한하고, 2025-2026은 optional later holdout으로
  남깁니다.

Chronological split:

| Split | Signal-date range |
| --- | --- |
| Train | `2018-01-01` to `2020-12-31` |
| Validation | `2021-01-01` to `2021-12-31` |
| Test | `2022-01-01` to `2024-12-31` |

Purge rule:

```text
label_end_date <= split_signal_end
```

label horizon이 다음 split로 넘어가는 sample은 제거합니다.

Normalization:
- Stage 1과 같은 scalar pixel normalization을 사용합니다:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

- train image에서만 fit합니다.
- validation/test pixel은 사용하지 않습니다.
- `(image_window, image_spec, return_horizon)` experiment별로 stats를 따로 저장합니다.

상세 계획:
- [Stage 2 BTC label/split/normalization plan](../docs/stage2_label_split_normalization_plan.md)

Split-count artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`
