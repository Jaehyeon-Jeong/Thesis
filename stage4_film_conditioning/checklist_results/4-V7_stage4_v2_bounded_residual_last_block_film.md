# 4-V7 Stage 4 v2 Bounded/Residual Last-Block FiLM

## English

Status: ready for Kaggle execution

Purpose:
- Test whether FiLM can be stabilized by preserving the strong Stage 2 visual
  path more explicitly.
- Keep the selected visual setting fixed: `I60/R20/ohlc_ma_vb`.
- Keep the context source fixed to F&G-only so the comparison isolates the
  architecture change from `4-V6`.

Why this is the next step:
- `4-V6` showed that F&G-only `film_full` on top of `ohlc_ma_vb` was unstable
  across seeds.
- The likely risk is not only the context itself, but the fact that full FiLM
  modulated every CNN block, including low-level chart feature extraction.
- V7 therefore changes the FiLM architecture before adding news context.

Model:
- Method name: `film_full_bounded_last_block`.
- Context features:
  - `fg_value`
  - `fg_mean_60`
  - `fg_delta_60`
  - `fg_std_60`
- Context encoder: same Stage 4 MLP condition embedding.
- FiLM insertion point: final CNN block only, after BatchNorm and before
  LeakyReLU.
- Bounded/residual formula:

```text
gamma = 1 + modulation_scale * tanh(raw_gamma)
beta  =     modulation_scale * tanh(raw_beta)
F'    = gamma * F + beta
```

Default:
- `modulation_scale = 0.10`.
- Zero-initialized gamma/beta heads, so the model starts as identity
  modulation: `gamma=1`, `beta=0`.
- F&G-only parameter check: `2,987,970` parameters, `+35,008` versus the Stage
  2 I60 visual baseline.

Kaggle runner:
- `notebooks/kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`
- Fixed run:
  - `I60/R20/ohlc_ma_vb`
  - context window `60`
  - F&G-only context
  - method `film_full_bounded_last_block`
  - seeds `42, 43, 44, 45, 46`

Expected comparison:
- Primary baseline: Stage 2 `I60/R20/ohlc_ma_vb` five-seed mean.
- Direct Stage 4 comparison: `4-V6`
  `I60/R20/ohlc_ma_vb + F&G-only + film_full`.
- Success criterion is not only mean accuracy. Check:
  - mean/std accuracy;
  - ROC-AUC;
  - predicted positive rate stability;
  - whether seed collapse is reduced;
  - Grad-CAM plus bounded gamma/beta export.

## 한국어

상태: Kaggle 실행 준비 완료

목적:
- 강한 Stage 2 visual path를 더 명시적으로 보존했을 때 FiLM이 안정화되는지
  확인합니다.
- visual setting은 그대로 고정합니다: `I60/R20/ohlc_ma_vb`.
- context도 F&G-only로 고정해서 `4-V6` 대비 architecture 변경 효과만 봅니다.

왜 다음 단계인가:
- `4-V6`에서 F&G-only `film_full`은 seed별로 불안정했습니다.
- 문제는 context 자체만이 아니라, full FiLM이 low-level chart feature를 잡는
  초반 CNN block까지 모두 조절한 구조일 수 있습니다.
- 그래서 news context를 추가하기 전에 FiLM 구조를 먼저 안정화합니다.

모델:
- Method name: `film_full_bounded_last_block`.
- Context features:
  - `fg_value`
  - `fg_mean_60`
  - `fg_delta_60`
  - `fg_std_60`
- Context encoder는 기존 Stage 4 MLP condition embedding을 그대로 씁니다.
- FiLM 삽입 위치는 마지막 CNN block 하나만입니다: BatchNorm 뒤, LeakyReLU 전.
- Bounded/residual 수식:

```text
gamma = 1 + modulation_scale * tanh(raw_gamma)
beta  =     modulation_scale * tanh(raw_beta)
F'    = gamma * F + beta
```

기본값:
- `modulation_scale = 0.10`.
- gamma/beta head는 zero-initialized라서 시작점은 identity modulation입니다:
  `gamma=1`, `beta=0`.
- F&G-only parameter check: `2,987,970` parameters, Stage 2 I60 visual
  baseline 대비 `+35,008`.

Kaggle runner:
- `notebooks/kaggle_stage4_v2_p8_ohlc_ma_vb_fg_only_bounded_last_block_film_five_seed_one_cell.md`
- 고정 run:
  - `I60/R20/ohlc_ma_vb`
  - context window `60`
  - F&G-only context
  - method `film_full_bounded_last_block`
  - seeds `42, 43, 44, 45, 46`

비교 기준:
- Primary baseline: Stage 2 `I60/R20/ohlc_ma_vb` five-seed mean.
- 직접 비교 대상: `4-V6`
  `I60/R20/ohlc_ma_vb + F&G-only + film_full`.
- 성공 기준은 mean accuracy만이 아닙니다. 같이 확인해야 합니다:
  - accuracy mean/std;
  - ROC-AUC;
  - predicted positive rate 안정성;
  - seed collapse 감소 여부;
  - Grad-CAM plus bounded gamma/beta export.
