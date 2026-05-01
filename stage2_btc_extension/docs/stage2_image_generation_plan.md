# Stage 2 BTC Image Generation Plan

## English

Status: planning complete for checklist 2-3. Implementation happens later in
`2-I3`.

Source basis:
- Re-image summary: `자료조사/Re-image 요약.md`
  - line 25: MA and volume-bar image elements
  - lines 32-36: 5/20/60-day input periods, 3 pixels per day, image sizes
  - line 110: robustness/ablation notes for volume bars and MA
- Stage 2 data audit:
  - `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`

Important decision:
- BTC OHLCV provides `Open`, `High`, `Low`, `Close`, and `Volume`.
- BTC OHLCV does not provide moving-average columns.
- Therefore, Stage 2 computes moving averages from BTC close prices using the
  standard simple moving average formula. This is not an arbitrary new feature;
  it is the Re-image MA image element reconstructed from available BTC data.

Moving-average formula:

```text
MA_t^(K) = (1 / K) * sum_{i=0}^{K-1} Close_{t-i}
```

Window-specific MA:

| Image window | MA used |
| --- | --- |
| `I5` | 5-day SMA |
| `I20` | 20-day SMA |
| `I60` | 60-day SMA |

Leakage rule:
- `MA_t^(K)` uses only `Close_t` and earlier close prices.
- No price after the image date can enter the image.
- Future prices are used only for label construction in the later label step.

MA availability rule:
- Use `min_periods=K`.
- A MA point is valid only when all `K` close prices exist.
- For fair comparison across image specifications, use the same eligible sample
  dates for all four image specs within a given `I/K` and `R` setting.
- Practically, this means the common universe requires full trailing MA values
  for every date drawn in the image, even for specs that do not display MA.
  This prevents `OHLC` and `OHLC+MA` results from being compared on different
  sample sets.

Image specifications:

| Spec | Code name | Components |
| --- | --- | --- |
| A | `ohlc` | OHLC chart only |
| B | `ohlc_vb` | OHLC chart + volume bars |
| C | `ohlc_ma` | OHLC chart + moving-average line |
| D | `ohlc_ma_vb` | OHLC chart + moving-average line + volume bars |

This is the Stage 2 version of the Re-image image ablation:
- no MA / no volume
- no MA / volume
- MA / no volume
- MA / volume

Image sizes:

| Window | Height x width | Days | Pixels per day |
| --- | ---: | ---: | ---: |
| `I5` | `32 x 15` | 5 | 3 |
| `I20` | `64 x 60` | 20 | 3 |
| `I60` | `96 x 180` | 60 | 3 |

Volume-region policy:

| Window | Price area with volume | Gap | Volume area |
| --- | ---: | ---: | ---: |
| `I5` | 25 px | 1 px | 6 px |
| `I20` | 51 px | 1 px | 12 px |
| `I60` | 76 px | 1 px | 19 px |

If volume is not included:
- the price chart uses the full image height.

If volume is included:
- OHLC and MA are drawn only in the price area.
- volume bars are drawn in the bottom volume area.
- volume is scaled within each image by that image window's maximum volume.

Price scaling:
- Each image is scaled independently.
- Window maximum `High` maps to the top of the price area.
- Window minimum `Low` maps to the bottom of the price area.
- OHLC and MA use the same price scaling.
- If a trailing MA value falls outside the window high-low range, clip it to the
  nearest image boundary and record this as an implementation warning/summary.

Common sample counts after full-MA eligibility:

| Image | Horizon | Samples | Positive rate | First image end date | Last image end date |
| --- | --- | ---: | ---: | --- | --- |
| I5 | R5 | 2984 | 0.5275 | 2018-01-09 | 2026-03-11 |
| I5 | R20 | 2969 | 0.5305 | 2018-01-09 | 2026-02-24 |
| I5 | R60 | 2929 | 0.5268 | 2018-01-09 | 2026-01-15 |
| I20 | R5 | 2954 | 0.5305 | 2018-02-08 | 2026-03-11 |
| I20 | R20 | 2939 | 0.5328 | 2018-02-08 | 2026-02-24 |
| I20 | R60 | 2899 | 0.5323 | 2018-02-08 | 2026-01-15 |
| I60 | R5 | 2874 | 0.5303 | 2018-04-29 | 2026-03-11 |
| I60 | R20 | 2859 | 0.5366 | 2018-04-29 | 2026-02-24 |
| I60 | R60 | 2819 | 0.5406 | 2018-04-29 | 2026-01-15 |

Implementation outputs expected later:
- sample image PNGs for each window/spec pair
- image metadata CSV with date, window, spec, source rows, scaling range, and
  MA clipping count
- tensor/memmap or generated-on-the-fly dataset implementation

## 한국어

상태: checklist 2-3 계획 완료. 실제 구현은 이후 `2-I3`에서 합니다.

근거:
- Re-image 요약: `자료조사/Re-image 요약.md`
  - line 25: MA와 volume bar image 요소
  - lines 32-36: 5/20/60일 입력 기간, 하루 3픽셀, image size
  - line 110: volume bar와 MA 제거 ablation/robustness
- Stage 2 data audit:
  - `stage2_btc_extension/reports/data_audit/btc_ohlcv_audit.md`

중요 결정:
- BTC OHLCV에는 `Open`, `High`, `Low`, `Close`, `Volume`이 있습니다.
- BTC OHLCV에는 moving-average column이 없습니다.
- 따라서 Stage 2에서는 표준 simple moving average 공식으로 BTC close에서 MA를
  계산합니다. 이것은 임의 feature 추가가 아니라, Re-image의 MA image element를
  BTC 데이터에서 재구성하는 것입니다.

Moving-average 공식:

```text
MA_t^(K) = (1 / K) * sum_{i=0}^{K-1} Close_{t-i}
```

window별 MA:

| Image window | 사용하는 MA |
| --- | --- |
| `I5` | 5-day SMA |
| `I20` | 20-day SMA |
| `I60` | 60-day SMA |

Leakage rule:
- `MA_t^(K)`는 `Close_t`와 그 이전 close만 사용합니다.
- image 날짜 이후 가격은 image에 절대 들어가지 않습니다.
- 미래 가격은 이후 label construction 단계에서만 사용합니다.

MA availability rule:
- `min_periods=K`를 사용합니다.
- MA point는 필요한 `K`개 close가 모두 있을 때만 valid합니다.
- 같은 `I/K`와 `R` setting 안에서는 네 가지 image spec이 같은 날짜 sample에서
  비교되도록 공통 eligible sample date를 사용합니다.
- 따라서 MA를 표시하지 않는 spec도, 공정한 비교를 위해 image 내부 모든 날짜에
  full trailing MA가 존재하는 sample universe를 같이 사용합니다. 이렇게 해야
  `OHLC`와 `OHLC+MA`가 서로 다른 표본에서 비교되는 문제를 피할 수 있습니다.

Image specifications:

| Spec | Code name | 구성 |
| --- | --- | --- |
| A | `ohlc` | OHLC chart only |
| B | `ohlc_vb` | OHLC chart + volume bars |
| C | `ohlc_ma` | OHLC chart + moving-average line |
| D | `ohlc_ma_vb` | OHLC chart + moving-average line + volume bars |

이 네 가지가 Stage 2의 Re-image image ablation입니다:
- MA 없음 / volume 없음
- MA 없음 / volume 있음
- MA 있음 / volume 없음
- MA 있음 / volume 있음

Image sizes:

| Window | Height x width | Days | Pixels per day |
| --- | ---: | ---: | ---: |
| `I5` | `32 x 15` | 5 | 3 |
| `I20` | `64 x 60` | 20 | 3 |
| `I60` | `96 x 180` | 60 | 3 |

Volume-region policy:

| Window | volume 포함 시 price area | Gap | Volume area |
| --- | ---: | ---: | ---: |
| `I5` | 25 px | 1 px | 6 px |
| `I20` | 51 px | 1 px | 12 px |
| `I60` | 76 px | 1 px | 19 px |

volume이 없을 때:
- price chart가 전체 image height를 사용합니다.

volume이 있을 때:
- OHLC와 MA는 상단 price area에만 그립니다.
- volume bar는 하단 volume area에 그립니다.
- volume은 각 image window 내부 최대 volume 기준으로 rescale합니다.

Price scaling:
- 각 image는 독립적으로 scaling합니다.
- window 내부 maximum `High`가 price area 맨 위에 대응합니다.
- window 내부 minimum `Low`가 price area 맨 아래에 대응합니다.
- OHLC와 MA는 같은 price scaling을 사용합니다.
- trailing MA 값이 window high-low range 밖에 있으면 가장 가까운 image boundary로
  clip하고, 구현 summary에 clipping count를 남깁니다.

Full-MA eligibility를 반영한 공통 sample 수:

| Image | Horizon | Samples | Positive rate | 첫 image 종료일 | 마지막 image 종료일 |
| --- | --- | ---: | ---: | --- | --- |
| I5 | R5 | 2984 | 0.5275 | 2018-01-09 | 2026-03-11 |
| I5 | R20 | 2969 | 0.5305 | 2018-01-09 | 2026-02-24 |
| I5 | R60 | 2929 | 0.5268 | 2018-01-09 | 2026-01-15 |
| I20 | R5 | 2954 | 0.5305 | 2018-02-08 | 2026-03-11 |
| I20 | R20 | 2939 | 0.5328 | 2018-02-08 | 2026-02-24 |
| I20 | R60 | 2899 | 0.5323 | 2018-02-08 | 2026-01-15 |
| I60 | R5 | 2874 | 0.5303 | 2018-04-29 | 2026-03-11 |
| I60 | R20 | 2859 | 0.5366 | 2018-04-29 | 2026-02-24 |
| I60 | R60 | 2819 | 0.5406 | 2018-04-29 | 2026-01-15 |

이후 구현 output:
- window/spec별 sample image PNG
- date, window, spec, source row, scaling range, MA clipping count를 담은
  image metadata CSV
- tensor/memmap 또는 generated-on-the-fly dataset implementation
