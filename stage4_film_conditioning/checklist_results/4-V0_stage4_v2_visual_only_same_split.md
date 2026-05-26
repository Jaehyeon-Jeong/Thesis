# 4-V0. Stage 4 v2 Visual-Only Same-Split Baseline

## English

Status: runner ready, full Kaggle result pending.

This is Stage 4 v2 priority 1.

Experiment:

```text
image_window: 60
return_horizon: 20
image_spec: ohlc_ma_vb
context: none
model: Stage 2 Stock_CNN I60 visual-only baseline
primary seed: 42
```

Why this is first:

- Stage 4 v1 changed more than one variable at once: structured context,
  context-conditioned model heads, scratch training, and a context-table sample
  universe.
- Before changing FiLM, the pipeline needs a visual-only control under the same
  selected `I60/R20/ohlc_ma_vb` setup.
- This separates a possible sample-universe/split effect from the actual
  context/FiLM effect.

Interpretation rule:

- If this visual-only run matches the Stage 2 selected baseline, then the Stage
  4 v1 drop is mainly caused by context fusion/modulation or optimization
  instability.
- If this visual-only run also drops materially, then the first explanation is
  not FiLM. The split/sample universe or rerun conditions must be audited before
  new FiLM designs.

Duplicate-feature hypothesis:

- The selected `ohlc_ma_vb` image already contains trend, moving average,
  volume, volatility, and price-range information.
- BB/MFI/RV context may therefore overlap with information that the CNN can
  already infer from the chart image.
- This priority does not test that hypothesis directly. It establishes the
  control needed before testing `ohlc + context` in priorities `4-V2` to
  `4-V4`.

Execution:

- Kaggle one-cell wrapper:
  `notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`
- The wrapper uses Stage 2 BTC scripts deliberately because priority 1 has no
  context branch. This keeps the visual model identical to the Stage 2 baseline.
- The output is written under `/kaggle/working/stage2_btc_extension/outputs/stage2`
  and summarized under `/kaggle/working/stage4_v2_visual_only_reports`.

Expected comparison target:

| Model | Accuracy mean | ROC-AUC mean | Notes |
| --- | ---: | ---: | --- |
| Stage 2 selected five-seed baseline, `I60/R20/ohlc_ma_vb` | `0.5793` | `0.5849` | current strongest BTC baseline |
| Stage 4 v1 `film_full`, five-seed mean | `0.5510` | `0.5677` | best Stage 4 v1 context method, unstable |
| 4-V0 visual-only same-split | pending | pending | priority 1 result |

Next decisions after the result:

1. If 4-V0 is close to Stage 2, proceed to `4-V1` and `4-V2`.
2. If 4-V0 is below Stage 2, audit run settings, seed behavior, and output
   parity before changing FiLM.
3. Do not claim FiLM failure until the visual-only control is complete.

## 한국어

상태: runner 준비 완료, full Kaggle 결과 대기 중.

이 항목은 Stage 4 v2 우선순위 1입니다.

실험:

```text
image_window: 60
return_horizon: 20
image_spec: ohlc_ma_vb
context: 없음
model: Stage 2 Stock_CNN I60 visual-only baseline
primary seed: 42
```

이걸 먼저 하는 이유:

- Stage 4 v1에서는 structured context, context-conditioned model head,
  scratch training, context table 기반 sample universe가 동시에 들어갔습니다.
- FiLM 구조를 바꾸기 전에, 같은 `I60/R20/ohlc_ma_vb` 조건에서 context 없는
  visual-only control을 먼저 확인해야 합니다.
- 그래야 v1 성능 하락이 sample/split 문제인지, context/FiLM 문제인지 분리할 수
  있습니다.

해석 규칙:

- 이 visual-only run이 Stage 2 selected baseline과 비슷하면, Stage 4 v1 하락은
  context fusion/modulation 또는 optimization instability 쪽으로 해석합니다.
- 이 visual-only run도 크게 떨어지면, 첫 원인은 FiLM이 아닙니다. split/sample
  universe 또는 rerun 조건부터 다시 확인해야 합니다.

Duplicate-feature 가설:

- 선택된 `ohlc_ma_vb` 이미지는 이미 trend, moving average, volume, volatility,
  price range 정보를 담고 있습니다.
- 따라서 BB/MFI/RV context는 CNN이 이미지에서 이미 어느 정도 학습할 수 있는
  정보와 겹칠 수 있습니다.
- 4-V0은 이 가설을 직접 검증하지 않습니다. 4-V2부터 4-V4의 `ohlc + context`
  실험을 해석하기 위한 control입니다.

실행:

- Kaggle one-cell wrapper:
  `notebooks/kaggle_stage4_v2_p1_visual_only_same_split_one_cell.md`
- 우선순위 1은 context branch가 없으므로 Stage 2 BTC script를 의도적으로
  사용합니다. 그래야 visual model이 Stage 2 baseline과 동일하게 유지됩니다.
- output은 `/kaggle/working/stage2_btc_extension/outputs/stage2`에 저장하고,
  summary는 `/kaggle/working/stage4_v2_visual_only_reports`에 저장합니다.

비교 기준:

| Model | Accuracy mean | ROC-AUC mean | Notes |
| --- | ---: | ---: | --- |
| Stage 2 selected five-seed baseline, `I60/R20/ohlc_ma_vb` | `0.5793` | `0.5849` | 현재 가장 강한 BTC baseline |
| Stage 4 v1 `film_full`, five-seed mean | `0.5510` | `0.5677` | Stage 4 v1 best context method, seed 불안정 |
| 4-V0 visual-only same-split | pending | pending | 우선순위 1 결과 |

결과 이후 판단:

1. 4-V0이 Stage 2와 비슷하면 `4-V1`, `4-V2`로 진행합니다.
2. 4-V0도 Stage 2보다 낮으면 FiLM을 바꾸기 전에 run setting, seed behavior,
   output parity를 먼저 확인합니다.
3. Visual-only control 없이 FiLM 실패라고 결론 내리지 않습니다.
