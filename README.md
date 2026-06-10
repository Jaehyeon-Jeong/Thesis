# Thesis

## English

This repository contains the experiment pipeline for a thesis extending the chart-image CNN framework from *Re-Imag(in)ing Price Trends* toward BTC and market-context conditioning.

Large source data, checkpoints, paper PDFs, and large prediction files are not tracked. Stage folders contain the detailed implementation notes, results, and reproducibility records.

### Current Status

| Stage | Scope | Status | Main links |
| --- | --- | --- | --- |
| `stage0_data_check` | Data/source audit | Completed | [README](stage0_data_check/README.md), [checklist](stage0_data_check/checklist.md) |
| `stage1_reimage_reproduction` | Original Re-image I20 stock-chart pipeline | Completed for current seed-42 full-data fast diagnostics on `I20/R5`, `I20/R20`, `I20/R60` | [README](stage1_reimage_reproduction/README.md), [status report](stage1_reimage_reproduction/reports/stage1_current_status_report.md), [CSV](stage1_reimage_reproduction/reports/tables/stage1_seed42_current_status.csv) |
| `stage2_btc_extension` | Asset change from stock charts to BTC OHLCV | Completed for current scope: first screened all `I5/I20/I60 x R5/R20/R60 x 4 image specs` with one seed, then reran selected effective candidates with five seeds; best stable baseline is `I60/R20/ohlc_ma_vb` | [README](stage2_btc_extension/README.md), [single-seed screening report](stage2_btc_extension/reports/stage2_single_seed_result_report.md), [selected five-seed report](stage2_btc_extension/reports/stage2_i20_i60_r20_five_seed_result_report.md) |
| `stage3_linear_adapter` | Linear adapter ablation | Negative result; adapter underperformed the Stage 2 visual baseline | [README](stage3_linear_adapter/README.md), [checklist](stage3_linear_adapter/checklist.md) |
| `stage4_film_conditioning` | Market-context concat/gating/FiLM | Main experiment cycle complete: numeric context, news TF-IDF/SVD, F&G, technical context, FSI/RORO, derivatives/leverage, and image-spec complement checks are done; only optional conditional-regime interpretation remains | [README](stage4_film_conditioning/README.md), [checklist](stage4_film_conditioning/checklist.md), [final report](stage4_film_conditioning/checklist_results/4-N14_final_stage4_interpretability_report.md), [N16 result](stage4_film_conditioning/checklist_results/4-N16-5_derivatives_interpretability_export.md) |
| `stage5_llm_news_embedding` | LLM-derived news embedding/sentiment context | Completed for first thesis draft: OpenAI embedding, FinBERT-only, FinBERT+F&G, correction/regression, Grad-CAM/modulation export, final report, and gamma/beta relaxation diagnostics are complete | [README](stage5_llm_news_embedding/README.md), [checklist](stage5_llm_news_embedding/checklist.md), [5-13 final report](stage5_llm_news_embedding/checklist_results/5-13_final_stage5_report_title_decision.md), [5-14 gamma/beta relaxation](stage5_llm_news_embedding/checklist_results/5-14_finbert_fg_gamma_beta_relaxation.md) |

### Short Result Summary

- **Stage 1:** the original I20 pipeline is implemented and was run end-to-end for `R5`, `R20`, and `R60` under the same seed-42 fast diagnostic setting. Detailed metrics are in the Stage 1 report.
- **Stage 2:** the pipeline was transferred to BTC. The first pass used one seed to screen all image-window, return-horizon, and image-spec candidates; selected effective candidates were then checked with five seeds. `I60/R20/ohlc_ma_vb` is the strongest stable BTC baseline so far.
- **Stage 3:** the Linear adapter did not improve the BTC visual baseline and is treated as a failed ablation.
- **Stage 4:** the full market-context cycle is complete for the current thesis scope. Scratch-trained FiLM was unstable, so the final defensible protocol preserves the Stage 2 visual baseline with frozen pretrained checkpoints and trains only bounded final-block context-FiLM heads. F&G is the best compact external regime signal on the strongest image baseline, public FSI/RORO is stable but weak, chart-derived technical context is mostly redundant, and derivatives/leverage context gives the clearest interpretable same-image positive case: `ohlc_vb + funding_plus_cftc_oi` improves the `ohlc_vb` baseline by `+0.002082` accuracy with net `+15` corrections. Remaining Stage 4 work is optional conditional-regime interpretation, not another modeling branch.
- **Stage 5:** the LLM/news representation track is complete for the first thesis draft. OpenAI headline embeddings and FinBERT headline sentiment were converted into numeric context for the same frozen Stage2 + bounded FiLM protocol. Embedding-only and FinBERT-only do not beat the visual baseline on accuracy. FinBERT + F&G is a small positive/near-tie result: mean accuracy `0.580569`, ROC-AUC `0.585843`, `+0.001249` accuracy over Stage2 `ohlc_ma_vb`, and `+0.000278` over F&G-only. Correction/regression and Grad-CAM/modulation exports support a conservative calibration interpretation. Gamma/beta relaxation was tested as a final diagnostic and did not improve the bounded FiLM result.

### Main Documents

- [PLAN.md](PLAN.md)
- [Overall pipeline diagram](docs/overall_pipeline_diagram.md)
- [Execution environment diagram](docs/execution_environment_diagram.md)
- [Thesis LaTeX draft source](thesis_draft/Thesis_Jaehyeon_Jeong.tex)
- [Thesis evidence map](thesis_draft/evidence_map.md)

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
| `stage1_reimage_reproduction` | 원 논문 I20 stock-chart pipeline 재현 | 현재 범위 완료: `I20/R5`, `I20/R20`, `I20/R60` seed-42 full-data fast diagnostic 완료 | [README](stage1_reimage_reproduction/README.md), [status report](stage1_reimage_reproduction/reports/stage1_current_status_report.md), [CSV](stage1_reimage_reproduction/reports/tables/stage1_seed42_current_status.csv) |
| `stage2_btc_extension` | 자산군을 BTC OHLCV로 교체 | 현재 범위 완료: `I5/I20/I60 x R5/R20/R60 x 4 image specs`를 seed 1개로 1차 선별한 뒤, 효과가 있던 후보만 seed 5개로 재검증; 가장 안정적인 baseline은 `I60/R20/ohlc_ma_vb` | [README](stage2_btc_extension/README.md), [single-seed screening report](stage2_btc_extension/reports/stage2_single_seed_result_report.md), [selected five-seed report](stage2_btc_extension/reports/stage2_i20_i60_r20_five_seed_result_report.md) |
| `stage3_linear_adapter` | Linear adapter ablation | 실패/negative result; Stage 2 visual baseline보다 낮음 | [README](stage3_linear_adapter/README.md), [checklist](stage3_linear_adapter/checklist.md) |
| `stage4_film_conditioning` | Market-context concat/gating/FiLM | 주요 실험 cycle 완료: numeric context, news TF-IDF/SVD, F&G, technical context, FSI/RORO, derivatives/leverage, image-spec complement check까지 완료; 선택적 conditional-regime 해석만 남음 | [README](stage4_film_conditioning/README.md), [checklist](stage4_film_conditioning/checklist.md), [final report](stage4_film_conditioning/checklist_results/4-N14_final_stage4_interpretability_report.md), [N16 result](stage4_film_conditioning/checklist_results/4-N16-5_derivatives_interpretability_export.md) |
| `stage5_llm_news_embedding` | LLM 기반 news embedding/sentiment context | 1차 논문 초안 기준 완료: OpenAI embedding, FinBERT-only, FinBERT+F&G, correction/regression, Grad-CAM/modulation export, final report, gamma/beta relaxation diagnostics 완료 | [README](stage5_llm_news_embedding/README.md), [checklist](stage5_llm_news_embedding/checklist.md), [5-13 final report](stage5_llm_news_embedding/checklist_results/5-13_final_stage5_report_title_decision.md), [5-14 gamma/beta relaxation](stage5_llm_news_embedding/checklist_results/5-14_finbert_fg_gamma_beta_relaxation.md) |

### 짧은 결과 요약

- **Stage 1:** 원 논문 I20 pipeline을 구현했고 `R5`, `R20`, `R60` 전체를 seed-42 fast diagnostic 조건으로 실행했습니다. 자세한 수치는 Stage 1 report에 있습니다.
- **Stage 2:** 같은 pipeline을 BTC로 옮겨 실험했습니다. 먼저 seed 1개로 전체 후보를 1차 선별했고, 효과가 있던 후보만 seed 5개로 재검증했습니다. 현재는 `I60/R20/ohlc_ma_vb`가 가장 안정적인 BTC baseline입니다.
- **Stage 3:** Linear adapter는 성능을 개선하지 못해 실패한 ablation으로 정리했습니다.
- **Stage 4:** 현재 논문 범위의 market-context cycle은 완료했습니다. Scratch 학습 FiLM은 불안정했기 때문에 최종적으로 Stage 2 pretrained checkpoint를 불러오고 visual CNN/classifier를 freeze한 뒤 bounded final-block context-FiLM head만 학습하는 protocol로 정리했습니다. F&G는 가장 강한 image baseline에서 가장 compact한 외부 regime signal이고, public FSI/RORO는 안정적이지만 약했으며, chart-derived technical context는 대부분 중복 신호였습니다. 가장 해석 가능한 positive case는 derivatives/leverage context로, `ohlc_vb + funding_plus_cftc_oi`가 `ohlc_vb` baseline 대비 accuracy `+0.002082`, net correction `+15`를 만들었습니다. 남은 Stage 4 작업은 새 모델 실험이 아니라 선택적 conditional-regime 해석입니다.
- **Stage 5:** LLM/news representation track은 1차 논문 초안 기준 완료했습니다. OpenAI headline embedding과 FinBERT headline sentiment를 같은 frozen Stage2 + bounded FiLM protocol의 numeric context로 변환했습니다. Embedding-only와 FinBERT-only는 accuracy 기준으로 visual baseline을 넘지 못했습니다. FinBERT + F&G는 mean accuracy `0.580569`, ROC-AUC `0.585843`로 Stage2 `ohlc_ma_vb` 대비 `+0.001249`, F&G-only 대비 `+0.000278`의 small positive/near-tie 결과입니다. Correction/regression과 Grad-CAM/modulation export는 conservative calibration 해석을 뒷받침합니다. 마지막으로 gamma/beta relaxation diagnostic을 수행했지만 bounded FiLM 결과를 개선하지 못했습니다.

### 주요 문서

- [PLAN.md](PLAN.md)
- [전체 파이프라인 다이어그램](docs/overall_pipeline_diagram.md)
- [실행 환경 다이어그램](docs/execution_environment_diagram.md)
- [논문 LaTeX 초안 source](thesis_draft/Thesis_Jaehyeon_Jeong.tex)
- [논문 evidence map](thesis_draft/evidence_map.md)

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
