# Stage 4 Implementation Readiness Review

## English

Status: checklist `4-I0` complete.

Verdict:
- Stage 4 is ready to move from planning to implementation.
- The next checklist item should be `4-I1`: shared Stage 4 config/code
  scaffold.
- The first implementation must keep the Stage 2 BTC image pipeline fixed and
  add only the context branch, fusion/modulation models, and Stage 4 export
  logic.

## Readiness Decision

Implementation can proceed with the structured numeric-context track:

```text
I60/R20/ohlc_ma_vb image
  + 8-feature context vector
  -> concat / gating / gamma-only FiLM / full FiLM
```

The first implementation target is not the news/LLM track. News remains a
second-phase `4-N` extension after the numeric context ablation is stable.

## Checked Planning Inputs

Completed planning documents:
- `docs/stage4_pipeline.md`
- `docs/condition_track_plan.md`
- `docs/film_insertion_design.md`
- `docs/film_reference_review.md`
- `docs/news_context_plan.md`
- `docs/source_map.md`
- `docs/professor_meeting_stage4_direction_brief.md`

Completed checklist result reports:
- `4-0` scaffold
- `4-1` context fusion and news plan
- `4-2` structured context audit and leakage policy
- `4-3` news dataset audit and feasibility decision
- `4-4` Stage 2/Stage 3 dependency and baseline review
- `4-5` context encoder and normalization plan
- `4-6` concat/gating/FiLM insertion design
- `4-7` Grad-CAM plus context/gate/gamma/beta export plan
- `4-8` Kaggle runner and output backup plan

## Fixed Inputs For Implementation

Primary experiment:
- Image window: `60`
- Return horizon: `20`
- Image spec: `ohlc_ma_vb`
- Context window: `60`
- Ablations:
  - `concat`
  - `gating`
  - `film_gamma`
  - `film_full`
- First seed: `42`
- Later seeds: `42, 43, 44, 45, 46`

Primary context vector:

```text
[
  fg_value,
  fg_mean_60,
  fg_delta_60,
  fg_std_60,
  bb_percent_b_60,
  bb_bandwidth_60,
  mfi_60,
  rv_60,
]
```

Context preprocessing:
- feature-specific transform;
- train-only median imputation;
- train-only 1/99% clipping;
- train-only z-score normalization.

Context encoder:

```text
Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

## Dependency Decision

Stage 4 should reuse the Stage 2 BTC pipeline instead of copying or rewriting
it.

Required Stage 2 imports:
- BTC CSV loading;
- sample table construction;
- train/validation/test split;
- image generation;
- pixel normalization;
- DataLoader construction;
- prediction metrics;
- trading metrics;
- baseline Grad-CAM utilities where possible.

Implementation rule:
- Stage 4 config must include a `stage2_dependency` section with the Stage 2
  project root and `src` path.
- Stage 4 scripts must add both Stage 4 `src` and Stage 2 `src` to
  `sys.path`.
- Kaggle runners must either attach/copy the Stage 2 code snapshot or upload a
  larger code snapshot containing both Stage 2 and Stage 4.

This mirrors the Stage 3 dependency pattern and avoids changing the Stage 2
pipeline.

## Local Data Availability

Available locally:
- BTC daily OHLCV:
  `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- Fear & Greed daily index:
  `stage4_film_conditioning/FG_data/fear_greed_index.csv`

Implication:
- Model shape checks and non-data smoke tests can run locally.
- Full context feature construction can now be developed locally with the
  supplied F&G CSV.
- The first full execution target remains Kaggle because the intended final run
  should attach the same public F&G dataset and preserve the Kaggle output
  backup contract.

Quick F&G availability check:
- file: `fear_greed_index.csv`;
- columns: `timestamp`, `value`, `classification`, `date`;
- rows: `2,644`;
- date range: `2018-02-01` to `2025-05-02`;
- missing days inside that range: `4`
  (`2018-04-14`, `2018-04-15`, `2018-04-16`, `2024-10-26`);
- Stage 2 test period coverage: `1,460/1,461` days;
- value range: `5` to `95`;
- `historical_data.csv` in the same folder is not the first-run F&G input. It
  appears to be transaction/order data and should be excluded from the primary
  numeric context vector unless a separate audit justifies it.

## Implementation Risks And Guardrails

| Area | Risk | Guardrail |
| --- | --- | --- |
| Stage 2 reuse | Stage 4 code silently changes image/split/normalization | Import Stage 2 helpers; do not edit Stage 2 code |
| Context leakage | F&G/Bollinger/MFI/RV uses future information | Feature builder uses only rows dated at or before image end date `t` |
| Context normalization | Validation/test statistics leak into train | Fit median/clipping/z-score only on train split |
| F&G alignment | Missing dates or shifted publication date | Audit coverage and use previous available value only |
| Dataset join | Context row mismatches image sample | Join by Stage 2 `sample_index` and image end `Date`; output audit counts |
| Model identity start | Gate/FiLM destabilizes baseline at epoch 1 | Zero-init final gate/FiLM heads; gate starts `1`, gamma starts `1`, beta starts `0` |
| Output loss | Training completes but result package incomplete | 4-8 output checker is the completion gate |

## Implementation Order

Proceed in this order:

1. `4-I1`: shared config/code scaffold.
2. `4-I2`: structured context source audit and feature builder.
3. `4-I3`: context MLP encoder.
4. `4-I4`: `CNN + context concat` model.
5. `4-I5`: `CNN + context gating` model.
6. `4-I6`: FiLM layer and FiLM generator modules.
7. `4-I7`: gamma-only and full FiLM models.
8. `4-I8`: Stage 4 runner using the fixed Stage 2 data pipeline.
9. `4-I9`: prediction, classification metric, and trading metric export.
10. `4-I10`: Grad-CAM plus context/gate/gamma/beta export.
11. `4-I11`: local or small Kaggle smoke test.
12. `4-I12`: Kaggle four-ablation single-seed run.
13. `4-I13`: Kaggle selected five-seed runner.
14. `4-I14`: Stage 4 result report.

Source/task mapping:
- `reports/tables/stage4_implementation_task_map.csv`

## Decision

Proceed to `4-I1`.

## 한국어

상태: checklist `4-I0` 완료.

판정:
- Stage 4는 planning에서 implementation으로 넘어갈 준비가 됐습니다.
- 다음 체크리스트는 `4-I1`: Stage 4 공통 config/code scaffold입니다.
- 첫 구현은 Stage 2 BTC image pipeline을 고정하고, context branch,
  fusion/modulation model, Stage 4 export logic만 추가해야 합니다.

## Readiness 결정

Structured numeric-context track으로 구현을 시작할 수 있습니다.

```text
I60/R20/ohlc_ma_vb image
  + 8-feature context vector
  -> concat / gating / gamma-only FiLM / full FiLM
```

첫 구현 대상은 news/LLM track이 아닙니다. 뉴스는 numeric context ablation이 안정화된
뒤 second-phase `4-N` 확장으로 유지합니다.

## 확인한 계획 입력

완료된 planning 문서:
- `docs/stage4_pipeline.md`
- `docs/condition_track_plan.md`
- `docs/film_insertion_design.md`
- `docs/film_reference_review.md`
- `docs/news_context_plan.md`
- `docs/source_map.md`
- `docs/professor_meeting_stage4_direction_brief.md`

완료된 checklist result report:
- `4-0` scaffold
- `4-1` context fusion and news plan
- `4-2` structured context audit and leakage policy
- `4-3` news dataset audit and feasibility decision
- `4-4` Stage 2/Stage 3 dependency and baseline review
- `4-5` context encoder and normalization plan
- `4-6` concat/gating/FiLM insertion design
- `4-7` Grad-CAM plus context/gate/gamma/beta export plan
- `4-8` Kaggle runner and output backup plan

## 구현에 사용할 고정 입력

Primary experiment:
- Image window: `60`
- Return horizon: `20`
- Image spec: `ohlc_ma_vb`
- Context window: `60`
- Ablation:
  - `concat`
  - `gating`
  - `film_gamma`
  - `film_full`
- 첫 seed: `42`
- 이후 seed: `42, 43, 44, 45, 46`

Primary context vector:

```text
[
  fg_value,
  fg_mean_60,
  fg_delta_60,
  fg_std_60,
  bb_percent_b_60,
  bb_bandwidth_60,
  mfi_60,
  rv_60,
]
```

Context preprocessing:
- feature-specific transform;
- train-only median imputation;
- train-only 1/99% clipping;
- train-only z-score normalization.

Context encoder:

```text
Linear(8, 32) -> ReLU -> Dropout(0.10) -> Linear(32, 32) -> ReLU
```

## Dependency 결정

Stage 4는 Stage 2 BTC pipeline을 복사/재작성하지 않고 재사용해야 합니다.

필요한 Stage 2 import:
- BTC CSV loading;
- sample table construction;
- train/validation/test split;
- image generation;
- pixel normalization;
- DataLoader construction;
- prediction metrics;
- trading metrics;
- 가능한 범위의 baseline Grad-CAM utilities.

구현 규칙:
- Stage 4 config에는 Stage 2 project root와 `src` path를 담는
  `stage2_dependency` section이 필요합니다.
- Stage 4 script는 Stage 4 `src`와 Stage 2 `src`를 모두 `sys.path`에 추가해야 합니다.
- Kaggle runner는 Stage 2 code snapshot을 함께 attach/copy하거나, Stage 2와 Stage 4가
  같이 들어 있는 더 큰 code snapshot을 업로드해야 합니다.

이 방식은 Stage 3 dependency pattern과 같고, Stage 2 pipeline을 바꾸지 않게 해줍니다.

## 로컬 데이터 가용성

로컬에 있는 것:
- BTC daily OHLCV:
  `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- Fear & Greed daily index:
  `stage4_film_conditioning/FG_data/fear_greed_index.csv`

의미:
- Model shape check와 non-data smoke test는 로컬에서 가능합니다.
- Full context feature construction은 이제 제공된 F&G CSV로 로컬 개발이 가능합니다.
- 첫 full 실행 target은 여전히 Kaggle입니다. 최종 run에서는 같은 public F&G dataset을
  attach하고 Kaggle output backup 계약을 유지해야 합니다.

빠른 F&G availability check:
- file: `fear_greed_index.csv`;
- columns: `timestamp`, `value`, `classification`, `date`;
- rows: `2,644`;
- date range: `2018-02-01` to `2025-05-02`;
- 해당 범위 안의 missing day: `4`
  (`2018-04-14`, `2018-04-15`, `2018-04-16`, `2024-10-26`);
- Stage 2 test period coverage: `1,460/1,461` days;
- value range: `5` to `95`;
- 같은 폴더의 `historical_data.csv`는 첫 run의 F&G input이 아닙니다. 거래/order
  data로 보이므로 별도 audit 없이 primary numeric context vector에 넣지 않습니다.

## 구현 리스크와 guardrail

| 영역 | 리스크 | Guardrail |
| --- | --- | --- |
| Stage 2 재사용 | Stage 4 code가 image/split/normalization을 조용히 변경 | Stage 2 helper를 import하고 Stage 2 code는 수정하지 않음 |
| Context leakage | F&G/Bollinger/MFI/RV가 미래 정보를 사용 | feature builder는 image end date `t` 또는 그 이전 row만 사용 |
| Context normalization | validation/test 통계가 train에 섞임 | median/clipping/z-score는 train split에서만 fit |
| F&G alignment | missing date 또는 publication date shift | coverage audit 후 previous available value만 사용 |
| Dataset join | context row와 image sample mismatch | Stage 2 `sample_index`와 image end `Date`로 join하고 audit count 저장 |
| Model identity start | gate/FiLM이 epoch 1부터 baseline scale을 망침 | final gate/FiLM head zero-init; gate `1`, gamma `1`, beta `0`에서 시작 |
| Output loss | 학습은 끝났는데 결과 패키지가 불완전 | 4-8 output checker를 완료 gate로 사용 |

## 구현 순서

이 순서대로 진행합니다.

1. `4-I1`: shared config/code scaffold.
2. `4-I2`: structured context source audit and feature builder.
3. `4-I3`: context MLP encoder.
4. `4-I4`: `CNN + context concat` model.
5. `4-I5`: `CNN + context gating` model.
6. `4-I6`: FiLM layer and FiLM generator modules.
7. `4-I7`: gamma-only and full FiLM models.
8. `4-I8`: Stage 4 runner using the fixed Stage 2 data pipeline.
9. `4-I9`: prediction, classification metric, and trading metric export.
10. `4-I10`: Grad-CAM plus context/gate/gamma/beta export.
11. `4-I11`: local or small Kaggle smoke test.
12. `4-I12`: Kaggle four-ablation single-seed run.
13. `4-I13`: Kaggle selected five-seed runner.
14. `4-I14`: Stage 4 result report.

Source/task mapping:
- `reports/tables/stage4_implementation_task_map.csv`

## 결정

`4-I1`로 진행합니다.
