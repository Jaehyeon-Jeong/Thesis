# 2-4 BTC Label, Split, and Normalization Detail Plan

## English

Status: complete.

This checklist item fixes how BTC labels, paper-style train/validation split,
test split, leakage guards, and train-only pixel normalization will work.

Key decisions:
- Future return:
  `future_return_{t,R} = Close_{t+R} / Close_t - 1`.
- Label:
  `1` if `future_return > 0`, otherwise `0`.
- Exact zero return is class `0`.
- BTC `R5/R20/R60` means 5/20/60 daily bars.
- Stage 2 uses the paper-style split logic:
  - first split time into a train/validation pool and a later test period;
  - then split the train/validation pool 70/30 at random.
- Primary reporting period is capped at `2024-12-31`; 2025-2026 remains an
  optional later holdout.

Primary split:

| Component | Signal-date range | Rule |
| --- | --- | --- |
| Train/validation pool | `2018-01-01` to `2020-12-31` | 70/30 random split, seed `42` |
| Test | `2021-01-01` to `2024-12-31` | chronological holdout |

Purge rule:

```text
label_end_date <= split_signal_end
```

This removes samples whose label horizon crosses from the train/validation pool
into the test period or beyond the capped test period.

BTC caveat:
- Random train/validation split is paper-aligned.
- Because BTC is one rolling time series, adjacent train and validation samples
  can overlap strongly.
- Therefore validation is used for early stopping/model selection; final
  reporting relies on the chronological test period.

Normalization:
- Use Stage 1-style scalar pixel normalization:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

- Fit only on the 70% training subset.
- Do not use validation/test pixels.
- Save statistics per `(image_window, image_spec, return_horizon)` experiment.

Detailed plan:
- [Stage 2 BTC label/split/normalization plan](../docs/stage2_label_split_normalization_plan.md)

Split-count artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`

## 한국어

상태: 완료.

이번 체크리스트에서는 BTC label, 논문식 train/validation split, test split,
leakage guard, train-only pixel normalization 방식을 고정했습니다.

핵심 결정:
- Future return:
  `future_return_{t,R} = Close_{t+R} / Close_t - 1`.
- Label:
  `future_return > 0`이면 `1`, 아니면 `0`.
- 정확히 0인 return은 class `0`.
- BTC `R5/R20/R60`은 5/20/60 daily bar입니다.
- Stage 2는 논문식 split logic을 사용합니다.
  - 먼저 train/validation pool과 이후 test period를 시간 기준으로 분리합니다.
  - 그다음 train/validation pool 내부를 70/30 random split합니다.
- 기본 보고 기간은 `2024-12-31`까지로 제한하고, 2025-2026은 optional later holdout으로
  남깁니다.

Primary split:

| Component | Signal-date range | Rule |
| --- | --- | --- |
| Train/validation pool | `2018-01-01` to `2020-12-31` | 70/30 random split, seed `42` |
| Test | `2021-01-01` to `2024-12-31` | chronological holdout |

Purge rule:

```text
label_end_date <= split_signal_end
```

label horizon이 train/validation pool에서 test period로 넘어가거나 capped test period
밖으로 넘어가는 sample은 제거합니다.

BTC caveat:
- random train/validation split은 논문 방식에 더 가깝습니다.
- 다만 BTC는 단일 rolling time series라서 인접 train/validation sample이 많이
  겹칠 수 있습니다.
- 따라서 validation은 early stopping/model selection용으로 쓰고, 최종 보고는
  chronological test period에서 합니다.

Normalization:
- Stage 1과 같은 scalar pixel normalization을 사용합니다:

```text
normalized_image = (image / 255.0 - train_pixel_mean) / train_pixel_std
```

- 70% training subset에서만 fit합니다.
- validation/test pixel은 사용하지 않습니다.
- `(image_window, image_spec, return_horizon)` experiment별로 stats를 따로 저장합니다.

상세 계획:
- [Stage 2 BTC label/split/normalization plan](../docs/stage2_label_split_normalization_plan.md)

Split-count artifact:
- `stage2_btc_extension/reports/tables/stage2_label_split_plan_counts.csv`
