# 교수님 제출용 GitHub 안내 및 실험 요약

작성일: 2026-05-28  
Repository: `Jaehyeon-Jeong/Thesis`

## 1. 전체 연구 흐름

본 repository는 *Re-Imag(in)ing Price Trends*의 chart-image CNN 파이프라인을 재현한 뒤, 이를 BTC와 market-context conditioning으로 확장하는 실험 과정을 정리한 것입니다.

전체 흐름은 다음과 같습니다.

```text
Stage 0 Data/source audit
    -> Stage 1 Original Re-image pipeline reproduction
    -> Stage 2 BTC asset-class extension
    -> Stage 3 Linear adapter ablation
    -> Stage 4 Market-context FiLM conditioning
```

전체 구조는 아래 문서에서 먼저 확인할 수 있습니다.

- Root overview: [README.md](../README.md)
- Overall pipeline diagram: [docs/overall_pipeline_diagram.md](../docs/overall_pipeline_diagram.md)

## 2. GitHub를 보는 순서

처음 볼 때는 아래 순서가 가장 빠릅니다.

| 순서 | 볼 문서 | 목적 |
| --- | --- | --- |
| 1 | [README.md](../README.md) | 전체 stage별 현재 상태 확인 |
| 2 | [docs/overall_pipeline_diagram.md](../docs/overall_pipeline_diagram.md) | 전체 실험 흐름을 다이어그램으로 확인 |
| 3 | 각 stage의 `README.md` | 해당 stage의 goal, workflow, result, code map 확인 |
| 4 | 각 stage의 `checklist.md`와 `checklist_results/` | 계획 단계와 구현/리뷰 기록 확인 |
| 5 | 각 stage의 `reports/`와 `reports/tables/` | 실제 결과 요약과 CSV table 확인 |
| 6 | 각 stage의 `src/`, `scripts/`, `notebooks/` | 모델 코드, 실행 코드, Kaggle runner 확인 |

Stage별 README:

- [Stage 1 README](../stage1_reimage_reproduction/README.md)
- [Stage 2 README](../stage2_btc_extension/README.md)
- [Stage 3 README](../stage3_linear_adapter/README.md)
- [Stage 4 README](../stage4_film_conditioning/README.md)

## 3. Stage별 요약

### Stage 1: Original Re-image pipeline reproduction

목적:
- 원 논문 기반 I20 chart-image CNN pipeline을 먼저 구현하고 검증했습니다.
- 공개 `monthly_20d` stock chart image shard를 사용했습니다.
- `I20/R5`, `I20/R20`, `I20/R60`을 실행했습니다.

주요 결과:

| Experiment | Samples | Accuracy | Majority | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| `I20/R5` | 1,399,933 | 0.5273 | 0.5078 | 0.5373 |
| `I20/R20` | 1,393,845 | 0.5285 | 0.5222 | 0.5339 |
| `I20/R60` | 1,376,215 | 0.5312 | 0.5408 | 0.5298 |

중요하게 볼 파일:
- Result report: [stage1_current_status_report.md](../stage1_reimage_reproduction/reports/stage1_current_status_report.md)
- Result CSV: [stage1_seed42_current_status.csv](../stage1_reimage_reproduction/reports/tables/stage1_seed42_current_status.csv)
- Workflow/checklist: [stage1 checklist](../stage1_reimage_reproduction/checklist.md)

해석:
- Stage 1은 이후 실험의 원형 pipeline을 확인하는 단계입니다.
- 결과 수치는 원 논문 pipeline을 end-to-end로 구현하고 실행할 수 있음을 확인하는 baseline 역할입니다.

### Stage 2: BTC asset-class extension

목적:
- Stage 1의 chart-image CNN pipeline을 BTC OHLCV로 옮겼습니다.
- BTC raw OHLCV에서 chart image를 직접 생성했습니다.
- `I5/I20/I60 x R5/R20/R60 x 4 image specs` 전체를 seed 1개로 먼저 선별했습니다.
- 효과가 있던 후보만 seed 5개로 재검증했습니다.

중요한 실험 구조:

| Axis | Values |
| --- | --- |
| Image window/model | `I5`, `I20`, `I60` |
| Return horizon | `R5`, `R20`, `R60` |
| Image spec | `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb` |
| First screening | 36 runs, seed `42` |
| Selected robustness check | `I20/R20`, `I60/R20`, four image specs, seeds `42-46` |

핵심 결과:
- 현재 가장 안정적인 BTC visual baseline은 `I60/R20/ohlc_ma_vb`입니다.

| Image window | Return horizon | Image spec | Accuracy mean | Accuracy std | ROC-AUC mean |
| ---: | ---: | --- | ---: | ---: | ---: |
| 60 | 20 | `ohlc_ma_vb` | 0.5793 | 0.0182 | 0.5849 |

중요하게 볼 파일:
- Single-seed screening report: [stage2_single_seed_result_report.md](../stage2_btc_extension/reports/stage2_single_seed_result_report.md)
- Selected five-seed report: [stage2_i20_i60_r20_five_seed_result_report.md](../stage2_btc_extension/reports/stage2_i20_i60_r20_five_seed_result_report.md)
- Single-seed full table: [stage2_single_seed_seed_level_results.csv](../stage2_btc_extension/reports/tables/stage2_single_seed_seed_level_results.csv)
- Selected five-seed mean/std table: [stage2_i20_i60_r20_five_seed_mean_std_results.csv](../stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv)

해석:
- BTC에서는 짧은 window보다 `I60`의 긴 chart history가 더 유효했습니다.
- `R20` horizon이 가장 안정적이었습니다.
- MA와 volume이 포함된 `ohlc_ma_vb`가 Stage 4의 기준 visual baseline으로 선택되었습니다.

### Stage 3: Linear adapter ablation

목적:
- Stage 2의 CNN feature 뒤에 단순 Linear adapter를 붙였습니다.
- FiLM으로 가기 전, 단순 parameter 증가만으로 성능이 좋아지는지 확인했습니다.

핵심 결과:

| Model | Setting | Accuracy | Majority | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Stage 2 visual baseline | `I60/R20/ohlc_ma_vb`, seed `42` | 0.6031 | 0.5413 | 0.6170 |
| Stage 3 Linear adapter | same setting, adapter dim `128` | 0.5413 | 0.5413 | 0.5221 |

중요하게 볼 파일:
- Stage 3 README: [stage3 README](../stage3_linear_adapter/README.md)
- Linear adapter design: [linear_adapter_design.md](../stage3_linear_adapter/docs/linear_adapter_design.md)
- Stage 3 vs Stage 2 table: [stage3_smoke_vs_stage2.csv](../stage3_linear_adapter/reports/tables/stage3_smoke_vs_stage2.csv)

해석:
- Linear adapter는 Stage 2 visual baseline보다 크게 낮았습니다.
- 따라서 Stage 4는 “단순히 layer를 하나 더 붙이는 실험”이 아니라, market context가 CNN feature를 조건부로 조절하는지 보는 실험으로 정리해야 합니다.

### Stage 4: Market-context FiLM conditioning

목적:
- Stage 2의 가장 강한 BTC visual baseline에 market context를 별도 vector로 넣었습니다.
- Context를 이미지에 그리는 것이 아니라, numeric context vector로 인코딩했습니다.
- `concat`, `gating`, `FiLM gamma-only`, `FiLM full`을 비교했습니다.

사용한 context:

| Group | Features |
| --- | --- |
| Fear and Greed | `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60` |
| Technical context | `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60` |

중요한 Stage 4 결과:

| Experiment | Result | Interpretation |
| --- | --- | --- |
| v1 `film_full`, `ohlc_ma_vb` + all context, five seeds | Accuracy mean `0.5510`, ROC-AUC mean `0.5677` | Stage 2 baseline보다 낮고 seed instability가 있음 |
| v2 `ohlc`, visual-only | Accuracy `0.5420` | OHLC-only는 `ohlc_ma_vb`보다 약함 |
| v2 `ohlc` + all context + `film_full`, seed 42 | Accuracy `0.5725` | OHLC-only 대비 partial recovery |
| v2 `ohlc` + F&G-only, five seeds | Accuracy mean `0.5586` | F&G alone은 충분하지 않음 |
| v2 `ohlc` + technical-only, five seeds | Accuracy mean `0.5603` | technical context alone도 약함 |
| v2 `ohlc_ma_vb` + F&G-only, five seeds | Accuracy mean `0.5524` | 강한 visual baseline 위에서도 full FiLM은 불안정 |

중요하게 볼 파일:
- Stage 4 README: [stage4 README](../stage4_film_conditioning/README.md)
- Stage 4 pipeline: [stage4_pipeline.md](../stage4_film_conditioning/docs/stage4_pipeline.md)
- FiLM insertion design: [film_insertion_design.md](../stage4_film_conditioning/docs/film_insertion_design.md)
- v1 interpretation report: [stage4_v1_interpretation_report.md](../stage4_film_conditioning/reports/stage4_v1_interpretation/stage4_v1_interpretation_report.md)
- v2 reviews: [checklist_results/](../stage4_film_conditioning/checklist_results/)

현재 해석:
- `ohlc_ma_vb` image 자체가 이미 많은 technical information을 포함합니다.
- 그래서 Bollinger/MFI/RV처럼 OHLCV에서 파생되는 context는 중복 정보가 되어 noise가 될 수 있습니다.
- F&G는 image-external context지만, 현재 full FiLM 구조에서는 seed별 collapse가 발생합니다.
- 다음 단계는 context를 더 복잡하게 만드는 것이 아니라, 강한 visual baseline을 보존하는 bounded/residual last-block FiLM 구조를 시험하는 것입니다.

## 4. 중요한 코드 위치

### Stage 1 model

- [stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py](../stage1_reimage_reproduction/src/stage1_reimage/models/stock_cnn.py)

역할:
- 원 논문/Stock_CNN 스타일 I20 baseline CNN 구현.
- Stage 1 재현 실험의 기준 모델입니다.

### Stage 2 BTC model and data/image pipeline

- BTC model variants: [stage2_btc_extension/src/stage2_btc/models/stock_cnn.py](../stage2_btc_extension/src/stage2_btc/models/stock_cnn.py)
- BTC OHLCV loader/label/split: [stage2_btc_extension/src/stage2_btc/data/](../stage2_btc_extension/src/stage2_btc/data/)
- BTC chart image generation: [stage2_btc_extension/src/stage2_btc/imaging/ohlcv_image.py](../stage2_btc_extension/src/stage2_btc/imaging/ohlcv_image.py)

역할:
- `I5`, `I20`, `I60` CNN variant 구현.
- BTC OHLCV에서 chart image와 future-return label을 생성합니다.

### Stage 3 Linear adapter

- [stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py](../stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py)

역할:
- Stage 2 CNN feature 뒤에 Linear adapter를 붙이는 negative ablation 모델입니다.

### Stage 4 context and FiLM models

- Context feature builder: [stage4_film_conditioning/src/stage4_film/context/features.py](../stage4_film_conditioning/src/stage4_film/context/features.py)
- Context normalization: [stage4_film_conditioning/src/stage4_film/context/normalization.py](../stage4_film_conditioning/src/stage4_film/context/normalization.py)
- Context encoder MLP: [stage4_film_conditioning/src/stage4_film/conditions/context_encoder.py](../stage4_film_conditioning/src/stage4_film/conditions/context_encoder.py)
- FiLM generator: [stage4_film_conditioning/src/stage4_film/conditions/film_generator.py](../stage4_film_conditioning/src/stage4_film/conditions/film_generator.py)
- FiLM layer: [stage4_film_conditioning/src/stage4_film/layers/film.py](../stage4_film_conditioning/src/stage4_film/layers/film.py)
- Concat/gating models: [stage4_film_conditioning/src/stage4_film/models/context_stock_cnn.py](../stage4_film_conditioning/src/stage4_film/models/context_stock_cnn.py)
- FiLM models: [stage4_film_conditioning/src/stage4_film/models/film_stock_cnn.py](../stage4_film_conditioning/src/stage4_film/models/film_stock_cnn.py)

역할:
- F&G, Bollinger, MFI, realized volatility를 numeric context vector로 만듭니다.
- Context MLP가 condition embedding을 만들고, 이 embedding이 concat/gating/FiLM 방식으로 CNN feature를 조절합니다.

## 5. 실행 코드 위치

각 stage는 같은 구조를 따릅니다.

| Folder | Meaning |
| --- | --- |
| `configs/` | Local/Kaggle path and runtime config |
| `scripts/` | CLI 실행 파일: audit, train, evaluate, Grad-CAM, summarize |
| `notebooks/` | Kaggle one-cell runner |
| `reports/` | 결과 보고서, tables, figures |
| `checklist_results/` | 각 작업 단계별 review/result note |
| `src/` | 실제 Python package/source code |

주요 Kaggle runner:
- Stage 1: [kaggle_stage1_single_horizon_one_cell.md](../stage1_reimage_reproduction/notebooks/kaggle_stage1_single_horizon_one_cell.md)
- Stage 2 full grid: [kaggle_stage2_grid_single_seed_one_cell.md](../stage2_btc_extension/notebooks/kaggle_stage2_grid_single_seed_one_cell.md)
- Stage 2 selected five-seed: [kaggle_stage2_i20_i60_r20_five_seed_one_cell.md](../stage2_btc_extension/notebooks/kaggle_stage2_i20_i60_r20_five_seed_one_cell.md)
- Stage 3 single config: [kaggle_stage3_linear_single_config_one_cell.md](../stage3_linear_adapter/notebooks/kaggle_stage3_linear_single_config_one_cell.md)
- Stage 4 v2 runners: [stage4 notebooks](../stage4_film_conditioning/notebooks/)

## 6. 현재 결론과 다음 작업

현재까지의 결론:

1. Stage 1에서 원 논문 기반 chart-image CNN pipeline을 구현하고 실행했습니다.
2. Stage 2에서 같은 pipeline을 BTC로 옮겼고, `I60/R20/ohlc_ma_vb`가 가장 안정적인 BTC visual baseline으로 확인되었습니다.
3. Stage 3 Linear adapter는 성능을 개선하지 못했으므로 negative ablation으로 정리합니다.
4. Stage 4에서는 context-conditioned FiLM을 실험 중이지만, 단순 full FiLM은 아직 Stage 2 baseline보다 안정적이지 않습니다.
5. 현재 다음 방향은 bounded/residual last-block FiLM처럼 visual baseline을 보존하면서 context modulation을 제한하는 구조입니다.

교수님께 확인받고 싶은 핵심 방향:

- Stage 4를 “단순 feature 추가”가 아니라 “market context가 CNN visual feature를 조건부로 조절하고, gamma/beta/gate를 통해 해석 가능한 경로를 제공하는가?”로 정리하는 것이 적절한지.
- 다음 실험을 bounded/residual last-block FiLM으로 진행하는 것이 교수님이 의도한 FiLM 방향성과 맞는지.
- News context는 현재 구조가 안정화된 뒤 second-phase context track으로 두는 것이 적절한지.

방향이 확정되면, 남은 Stage 4 architecture 실험을 정리하고 논문 초안 작성 단계로 넘어갈 예정입니다.
