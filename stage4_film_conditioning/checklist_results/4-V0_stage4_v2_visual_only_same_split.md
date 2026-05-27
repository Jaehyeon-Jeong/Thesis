# 4-V0. Stage 4 v2 Visual-Only Same-Split Baseline

## English

Status: complete for seed `42`.

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
| Stage 2 selected seed-42 baseline, `I60/R20/ohlc_ma_vb` | `0.6031` | `0.6170` | direct comparison target |
| Stage 4 v1 `film_full`, five-seed mean | `0.5510` | `0.5677` | best Stage 4 v1 context method, unstable |
| 4-V0 visual-only same-split, seed 42 | `0.6031` | `0.6170` | reproduces Stage 2 seed-42 |

The exported `4-V0` CSV passes the sanity check: under the visual-only
`I60/R20/ohlc_ma_vb` setting, the Stage 4 v2 diagnostic runner reproduces the
Stage 2 seed-42 result. The Stage 4 v1 drop should therefore be attributed to
the context/fusion path first, not to a broken Stage 4 sample universe.

Next decisions after the result:

1. Proceed to `4-V1`: `I60/R20/ohlc`, no context.
2. Use `4-V1` as the plain-OHLC baseline before testing `4-V2`.
3. Do not compare `ohlc + context` directly to `ohlc_ma_vb` without noting the
   image-information gap.

## 한국어

상태: seed `42` 완료.

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
| Stage 2 selected seed-42 baseline, `I60/R20/ohlc_ma_vb` | `0.6031` | `0.6170` | 직접 비교 대상 |
| Stage 4 v1 `film_full`, five-seed mean | `0.5510` | `0.5677` | Stage 4 v1 best context method, seed 불안정 |
| 4-V0 visual-only same-split, seed 42 | `0.6031` | `0.6170` | Stage 2 seed-42 재현 |

export된 `4-V0` CSV 기준으로 sanity check는 통과했습니다. visual-only
`I60/R20/ohlc_ma_vb` 조건에서 Stage 4 v2 diagnostic runner는 Stage 2 seed-42
결과를 재현했습니다. 따라서 Stage 4 v1 하락은 먼저 context/fusion 경로에서
찾는 것이 맞습니다.

결과 이후 판단:

1. `4-V1`: `I60/R20/ohlc`, context 없음으로 진행합니다.
2. `4-V1`을 plain-OHLC 기준선으로 삼은 뒤 `4-V2`를 테스트합니다.
3. `ohlc + context`를 `ohlc_ma_vb`와 직접 비교할 때는 image information gap을
   반드시 명시합니다.
