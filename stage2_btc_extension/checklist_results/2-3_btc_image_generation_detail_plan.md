# 2-3 BTC Image-Generation Detail Plan

## English

Status: complete.

This checklist item fixes the BTC image-generation design before implementation.

Main answer:
- Yes, Stage 2 must compare four image specifications:
  `OHLC`, `OHLC+Volume`, `OHLC+MA`, and `OHLC+MA+Volume`.
- This follows the Re-image design where moving-average lines and volume bars
  are image elements that can be included or removed for ablation.

MA decision:
- BTC OHLCV does not include MA columns.
- Stage 2 computes simple moving averages from BTC close prices:

```text
MA_t^(K) = (1 / K) * sum_{i=0}^{K-1} Close_{t-i}
```

Window-specific MA:
- `I5` uses 5-day SMA.
- `I20` uses 20-day SMA.
- `I60` uses 60-day SMA.

Leakage guard:
- Each MA point uses only the current date and earlier close prices.
- No future return information enters the image.
- Future close prices are used only in the label step.

Fair-comparison rule:
- The four image specs must be compared on the same sample dates within each
  window/horizon setting.
- Therefore, Stage 2 uses a common sample universe that requires full trailing
  MA availability for every date drawn in the image, even for specs that do not
  display MA.

Spec names:

| Spec | Code name | Components |
| --- | --- | --- |
| A | `ohlc` | OHLC only |
| B | `ohlc_vb` | OHLC + volume bars |
| C | `ohlc_ma` | OHLC + moving average |
| D | `ohlc_ma_vb` | OHLC + moving average + volume bars |

Detailed plan:
- [Stage 2 BTC image generation plan](../docs/stage2_image_generation_plan.md)

## 한국어

상태: 완료.

이번 체크리스트에서는 구현 전에 BTC image-generation 설계를 고정했습니다.

핵심 답변:
- 맞습니다. Stage 2에서는 네 가지 image specification을 비교해야 합니다:
  `OHLC`, `OHLC+Volume`, `OHLC+MA`, `OHLC+MA+Volume`.
- 이것은 Re-image에서 moving-average line과 volume bar를 image 요소로 넣거나 빼는
  ablation 구조를 BTC에 적용한 것입니다.

MA 결정:
- BTC OHLCV에는 MA column이 없습니다.
- Stage 2에서는 BTC close price에서 simple moving average를 계산합니다:

```text
MA_t^(K) = (1 / K) * sum_{i=0}^{K-1} Close_{t-i}
```

window별 MA:
- `I5`는 5-day SMA를 사용합니다.
- `I20`은 20-day SMA를 사용합니다.
- `I60`은 60-day SMA를 사용합니다.

Leakage guard:
- 각 MA point는 해당 날짜와 그 이전 close만 사용합니다.
- 미래 수익률 정보는 image에 들어가지 않습니다.
- 미래 close price는 label 단계에서만 사용합니다.

공정 비교 규칙:
- 같은 window/horizon setting 안에서 네 가지 image spec은 같은 sample date에서
  비교해야 합니다.
- 따라서 Stage 2는 MA를 표시하지 않는 spec에도 공통 sample universe를 적용합니다.
  이 공통 universe는 image 내부 모든 날짜에 full trailing MA가 존재하는 sample만
  사용합니다.

Spec names:

| Spec | Code name | 구성 |
| --- | --- | --- |
| A | `ohlc` | OHLC only |
| B | `ohlc_vb` | OHLC + volume bars |
| C | `ohlc_ma` | OHLC + moving average |
| D | `ohlc_ma_vb` | OHLC + moving average + volume bars |

상세 계획:
- [Stage 2 BTC image generation plan](../docs/stage2_image_generation_plan.md)
