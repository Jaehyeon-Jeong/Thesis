# Thesis

## English

This repository contains the experiment pipeline for a thesis extending the chart-image CNN framework from *Re-Imag(in)ing Price Trends* toward BTC and market-context conditioning.

Large source data, checkpoints, paper PDFs, and large prediction files are not tracked. Stage folders contain the detailed implementation notes, results, and reproducibility records.

### Current Status

| Stage | Scope | Status | Main links |
| --- | --- | --- | --- |
| `stage0_data_check` | Data/source audit | Completed | [README](stage0_data_check/README.md), [checklist](stage0_data_check/checklist.md) |
| `stage1_reimage_reproduction` | Original Re-image I20 stock-chart pipeline | Completed for current seed-42 full-data fast diagnostics on `I20/R5`, `I20/R20`, `I20/R60`; strict paper-style batch/five-seed remains later | [README](stage1_reimage_reproduction/README.md), [status report](stage1_reimage_reproduction/reports/stage1_current_status_report.md), [CSV](stage1_reimage_reproduction/reports/tables/stage1_seed42_current_status.csv) |
| `stage2_btc_extension` | Asset change from stock charts to BTC OHLCV | Completed for current scope: first screened all `I5/I20/I60 x R5/R20/R60 x 4 image specs` with one seed, then reran selected effective candidates with five seeds; best stable baseline is `I60/R20/ohlc_ma_vb` | [README](stage2_btc_extension/README.md), [single-seed screening report](stage2_btc_extension/reports/stage2_single_seed_result_report.md), [selected five-seed report](stage2_btc_extension/reports/stage2_i20_i60_r20_five_seed_result_report.md) |
| `stage3_linear_adapter` | Linear adapter ablation | Negative result; adapter underperformed the Stage 2 visual baseline | [README](stage3_linear_adapter/README.md), [checklist](stage3_linear_adapter/checklist.md) |
| `stage4_film_conditioning` | Market-context concat/gating/FiLM | In progress; v1/v2 diagnostics show simple context injection is not yet robust; next direction is bounded/residual last-block FiLM | [README](stage4_film_conditioning/README.md), [checklist](stage4_film_conditioning/checklist.md) |

### Short Result Summary

- **Stage 1:** the original I20 pipeline is implemented and was run end-to-end for `R5`, `R20`, and `R60` under the same seed-42 fast diagnostic setting. Detailed metrics are in the Stage 1 report.
- **Stage 2:** the pipeline was transferred to BTC. The first pass used one seed to screen all image-window, return-horizon, and image-spec candidates; selected effective candidates were then checked with five seeds. `I60/R20/ohlc_ma_vb` is the strongest stable BTC baseline so far.
- **Stage 3:** the Linear adapter did not improve the BTC visual baseline and is treated as a failed ablation.
- **Stage 4:** market-context conditioning is under active refinement. Current results suggest context must be added without destabilizing the strong chart-image baseline.

### Main Documents

- [PLAN.md](PLAN.md)
- [Professor one-page update](reports/professor_one_page_update_2026-05-27.md)
- [Stage 4 decision report](reports/professor_stage4_decision_report_2026-05-21.md)
- [Overall pipeline diagram](docs/overall_pipeline_diagram.md)
- [Execution environment diagram](docs/execution_environment_diagram.md)

### Data Policy

Tracked:
- Markdown plans/reports
- diagrams
- configs
- source maps
- small CSV summaries
- small sample figures

Not tracked:
- paper PDFs
- raw `.dat` image shards
- raw `.feather` labels
- checkpoints
- large prediction CSVs
- Kaggle output zip archives

## 한국어

이 저장소는 *Re-Imag(in)ing Price Trends*의 chart-image CNN 파이프라인을 재현하고, 이를 BTC와 market-context conditioning으로 확장하는 논문 실험 저장소입니다.

대용량 원본 데이터, checkpoint, 논문 PDF, 대용량 prediction 파일은 GitHub에 올리지 않습니다. 자세한 구현 설명과 결과는 각 stage 폴더에 정리합니다.

### 현재 상태

| 단계 | 범위 | 상태 | 주요 링크 |
| --- | --- | --- | --- |
| `stage0_data_check` | 데이터/source audit | 완료 | [README](stage0_data_check/README.md), [checklist](stage0_data_check/checklist.md) |
| `stage1_reimage_reproduction` | 원 논문 I20 stock-chart pipeline 재현 | 현재 범위 완료: `I20/R5`, `I20/R20`, `I20/R60` seed-42 full-data fast diagnostic 완료; strict paper-style batch/five-seed는 later | [README](stage1_reimage_reproduction/README.md), [status report](stage1_reimage_reproduction/reports/stage1_current_status_report.md), [CSV](stage1_reimage_reproduction/reports/tables/stage1_seed42_current_status.csv) |
| `stage2_btc_extension` | 자산군을 BTC OHLCV로 교체 | 현재 범위 완료: `I5/I20/I60 x R5/R20/R60 x 4 image specs`를 seed 1개로 1차 선별한 뒤, 효과가 있던 후보만 seed 5개로 재검증; 가장 안정적인 baseline은 `I60/R20/ohlc_ma_vb` | [README](stage2_btc_extension/README.md), [single-seed screening report](stage2_btc_extension/reports/stage2_single_seed_result_report.md), [selected five-seed report](stage2_btc_extension/reports/stage2_i20_i60_r20_five_seed_result_report.md) |
| `stage3_linear_adapter` | Linear adapter ablation | 실패/negative result; Stage 2 visual baseline보다 낮음 | [README](stage3_linear_adapter/README.md), [checklist](stage3_linear_adapter/checklist.md) |
| `stage4_film_conditioning` | Market-context concat/gating/FiLM | 진행 중; 단순 context injection은 아직 robust하지 않음; 다음 방향은 bounded/residual last-block FiLM | [README](stage4_film_conditioning/README.md), [checklist](stage4_film_conditioning/checklist.md) |

### 짧은 결과 요약

- **Stage 1:** 원 논문 I20 pipeline을 구현했고 `R5`, `R20`, `R60` 전체를 seed-42 fast diagnostic 조건으로 실행했습니다. 자세한 수치는 Stage 1 report에 있습니다.
- **Stage 2:** 같은 pipeline을 BTC로 옮겨 실험했습니다. 먼저 seed 1개로 전체 후보를 1차 선별했고, 효과가 있던 후보만 seed 5개로 재검증했습니다. 현재는 `I60/R20/ohlc_ma_vb`가 가장 안정적인 BTC baseline입니다.
- **Stage 3:** Linear adapter는 성능을 개선하지 못해 실패한 ablation으로 정리했습니다.
- **Stage 4:** market context를 FiLM/gating/concat으로 붙이는 실험을 진행 중입니다. 현재 방향은 강한 chart baseline을 보존하면서 context modulation을 안정화하는 것입니다.

### 주요 문서

- [PLAN.md](PLAN.md)
- [교수님 1페이지 진행 보고](reports/professor_one_page_update_2026-05-27.md)
- [Stage 4 방향 확정 요청 보고서](reports/professor_stage4_decision_report_2026-05-21.md)
- [전체 파이프라인 다이어그램](docs/overall_pipeline_diagram.md)
- [실행 환경 다이어그램](docs/execution_environment_diagram.md)

### 데이터 정책

GitHub에 올리는 것:
- Markdown 계획/결과 보고
- 다이어그램
- config
- source map
- 작은 CSV summary
- 작은 sample figure

GitHub에 올리지 않는 것:
- 논문 PDF
- 원본 `.dat` image shard
- 원본 `.feather` label
- checkpoint
- 대용량 prediction CSV
- Kaggle output zip archive
