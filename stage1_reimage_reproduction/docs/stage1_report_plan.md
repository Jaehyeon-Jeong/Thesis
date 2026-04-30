# Stage 1 Report Plan

## English

Status:
- Planning gate `1-9` completed on 2026-04-30.
- This document defines the final Stage 1 reporting structure only. No report
  results have been generated yet.

## Purpose

Define the final Stage 1 report, tables, figures, limitations, and paper
comparison rules before implementation starts.

Stage 1 report identity:
- Stage 1 is a public-data reproduction of the Re-image 20-day full-spec stock
  image CNN pipeline.
- It is not a complete reproduction of every Re-image window, image
  specification, or ablation.
- Final reproduction status requires code, smoke tests, Kaggle full runs,
  predictions, metrics, tables, and Grad-CAM figures.

## Sources Checked

| Source | Report use |
| --- | --- |
| `../PLAN.md` | Fixed stage order, Kaggle-first full-run rule, GitHub-first model-core rule, Grad-CAM requirement. |
| `../stage0_data_check/docs/monthly20_data_check.md` | Data feasibility, file inventory, image shape/spec, label columns. |
| `../stage0_data_check/docs/source_reference_check.md` | GitHub reference commit and paper/GitHub mismatch log. |
| `../자료조사/Re-image 요약.md` | Report page map, paper result anchors, split/training/model details, interpretation section. |
| `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | Must be visually rechecked before final report citations and code comments. |
| `docs/data_loading_plan.md` | Data loading and image tensor convention. |
| `docs/label_construction_plan.md` | Label rule and horizon valid counts. |
| `docs/split_normalization_plan.md` | Split, positive rates by period, train-only normalization. |
| `docs/baseline_cnn_implementation_plan.md` | GitHub-style I20 model structure and known dilation mismatch. |
| `docs/training_loop_plan.md` | Training settings, seeds, checkpoint rules, paper-unreported choices. |
| `docs/kaggle_runner_plan.md` | Kaggle run modes, manifests, dataset/code snapshot requirements. |
| `docs/evaluation_prediction_plan.md` | Metrics, prediction schema, stock decile/H-L output plan. |
| `docs/gradcam_plan.md` | Figure 13-style Grad-CAM output plan. |

Local Re-image summary page map to cite in the final report:
- Image construction: pp.8-11.
- CNN structure and training: pp.12-21.
- U.S. stock experiments: pp.21-33.
- Transfer learning: pp.34-40.
- Interpretation section: pp.41-49.
- Internet appendix robustness/alternative models/simulations: pp.55-67.

PDF extraction note:
- As of this planning step, the local environment lacks `pdftotext/pdfinfo`.
- Final report page numbers must be visually checked in the PDF before the
  report is treated as citation-ready.

## Report Output

Primary report file:

```text
reports/stage1_reproduction_report.md
```

Supporting folders:

```text
reports/tables/
reports/figures/sample_images/
reports/figures/gradcam/
reports/figures/training/
reports/figures/calibration/
```

The report should be generated only in implementation gate `1-I12`, after the
actual outputs exist.

## Required Report Outline

1. Executive summary
2. What Stage 1 reproduces
3. Data audit and feasible scope
4. Label construction and leakage controls
5. Split and normalization
6. Model architecture and GitHub reference
7. Training and reproducibility setup
8. Classification results
9. Stock ranking, decile, and H-L portfolio results
10. Grad-CAM interpretation figures
11. Comparison to Re-image paper values
12. Known mismatches and limitations
13. Artifact inventory and run manifest summary
14. Next step: Stage 2 BTC extension

## Required Tables

Generate these CSV files under `reports/tables/`:

| File | Purpose |
| --- | --- |
| `stage1_dataset_summary.csv` | Dataset period, image spec, window, horizons, file counts, sample counts. |
| `stage1_split_summary.csv` | Train/validation/test rows and positive rates per horizon. |
| `stage1_training_summary.csv` | Seeds, epochs, best validation loss, checkpoint paths, training time if available. |
| `stage1_classification_metrics.csv` | Accuracy, precision, recall, F1, AUC, Brier, log loss per horizon/run mode. |
| `stage1_majority_baseline_comparison.csv` | Accuracy vs majority-class baseline per horizon/run mode. |
| `stage1_correlation_metrics.csv` | Global and date-wise Pearson/Spearman prediction-return correlations. |
| `stage1_portfolio_metrics.csv` | Equal-weight/value-weight decile and H-L return, volatility, Sharpe, turnover if implemented. |
| `stage1_paper_comparison.csv` | Our results beside comparable paper cells, with a match-status column. |
| `stage1_gradcam_samples.csv` | Samples used for Figure 13-style Grad-CAM outputs. |
| `stage1_artifact_manifest.csv` | Paths to configs, checkpoints, predictions, metrics, figures, and run manifests. |

Table rules:
- Every table must include `experiment_name`, `image_window`, `target_horizon`,
  `target_return_name`, and `run_mode`.
- Paper comparison tables must include `paper_source`, `paper_cell`, and
  `comparison_status`.
- If a metric is unavailable, store `null` or `NA` and include a reason column.

## Paper Comparison Rules

Compare only like-for-like cells:
- Same image window where possible.
- Same target horizon.
- Same stock cross-sectional setting.
- Same full-spec interpretation where the public data supports it.

Closest direct Stage 1 comparisons from the local summary:

| Stage 1 experiment | Paper comparison source | Paper value from local summary | Report handling |
| --- | --- | --- | --- |
| `I20/R20` classification | Re-image U.S. results/Table 2, pp.21-33 | Accuracy `53.3%`; Corr. about `3.4%` range for main cells | Direct comparison after PDF visual check. |
| `I20/R60` classification | Re-image U.S. results/Table 2, pp.21-33 | Accuracy `53.2%` | Direct comparison after PDF visual check. |
| `I20/R5` classification | Re-image paper, exact accuracy cell not extracted in current local summary | Pending PDF check | Do not invent a paper value. |
| `I20/R20` EW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.21`, Sharpe `2.16` | Compare if our decile implementation matches convention. |
| `I20/R20` VW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.05`, Sharpe `0.49` | Compare if value-weighting and annualization match. |
| `I20/R60` EW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.05`, Sharpe `0.37` | Compare after annualization convention check. |
| `I20/R5` EW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.84`, Sharpe `6.75` | Compare after turnover/annualization convention check. |
| `I20/R5` VW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.22`, Sharpe `1.74` | Compare after value-weighting convention check. |

Important:
- Do not claim full-paper reproduction from the current public data.
- Do not compare `I20` results against paper `I5` or `I60` cells as if they
  were the same experiment.
- Do not compare local smoke-test metrics to paper values.
- If the PDF check changes a value or page location, update this plan and
  `docs/source_map.md`.

## Required Limitation Wording

The final report must include these statements or close equivalents:

1. Stage 1 uses author/public `monthly_20d` rendered image shards. The available
   images are 20-day full-spec images interpreted as OHLC + 20-day MA + volume.
2. Therefore Stage 1 directly supports `I20/R5`, `I20/R20`, and `I20/R60`.
3. Current files do not directly support stock `I5`, stock `I60`, or A/B/C/D
   image-spec ablations.
4. `Ret_5d`, `Ret_20d`, and `Ret_60d` are individual stock future holding-period
   returns, not feature maps and not portfolio returns.
5. Binary labels are created from future return sign: `return > 0` is class `1`,
   otherwise class `0`.
6. H-L portfolio results, if reported, are constructed after prediction by
   cross-sectional ranking within date.
7. BTC is a single asset and cannot reuse the stock H-L decile setup directly.
8. The GitHub I20 model applies dilation to all three convolution layers,
   while the local paper summary emphasizes first-layer dilation. Stage 1
   follows the checked GitHub model core unless the user changes the decision.
9. Kaggle full `full_paper_style` runs are the primary reproduction runs. Local
   smoke-test metrics are not reproduction results.
10. Grad-CAM figures are post-hoc explanations from trained checkpoints. They
    are class-discriminative heatmaps, not raw feature maps or causal proof.

## Required Figures

| Figure | Source output |
| --- | --- |
| Sample public I20 images | `outputs/figures/sample_images/` |
| Training/validation loss curves | `outputs/figures/training/` |
| Calibration curves if implemented | `outputs/figures/calibration/` |
| Figure 13-style Grad-CAM for `I20/R20` | `outputs/figures/gradcam/stage1_i20_r20/...` |
| Extended Grad-CAM for `I20/R5` and `I20/R60` | `outputs/figures/gradcam/stage1_i20_r5/...`, `stage1_i20_r60/...` |

## Artifact Inventory

The final report must link or list:
- configs used for the run
- Kaggle run manifest
- dataset root/name/version
- code snapshot or Git commit
- split and normalization JSON files
- checkpoints
- per-seed prediction CSVs
- averaged prediction CSVs
- metrics JSON files
- report tables
- Grad-CAM figures

## What Is Deferred

Deferred to implementation/execution gates:
- Actually generating report tables.
- Running paper-style 5-seed Kaggle jobs.
- Computing final metrics and portfolio outputs.
- Generating Grad-CAM figures.
- Writing `reports/stage1_reproduction_report.md`.

## 한국어

상태:
- 2026-04-30에 planning gate `1-9`를 완료했습니다.
- 이 문서는 최종 1단계 보고 구조만 정의합니다. 아직 report 결과는 만들지 않았습니다.

## 목적

구현 시작 전에 최종 1단계 보고서, 표, 그림, limitation, 논문 비교 규칙을 정의합니다.

1단계 보고서의 정체성:
- 1단계는 Re-image의 20-day full-spec stock image CNN pipeline에 대한
  public-data reproduction입니다.
- Re-image의 모든 window, image specification, ablation을 완전히 재현한
  것은 아닙니다.
- 최종 재현 완료는 code, smoke test, Kaggle full run, prediction, metric,
  table, Grad-CAM figure가 모두 생성된 뒤에만 판단합니다.

## 확인한 근거

| 자료 | 보고서에서 쓰는 부분 |
| --- | --- |
| `../PLAN.md` | 고정 단계 순서, Kaggle-first full-run 원칙, GitHub-first model-core 원칙, Grad-CAM 필수. |
| `../stage0_data_check/docs/monthly20_data_check.md` | 데이터 가능 범위, 파일 inventory, image shape/spec, label columns. |
| `../stage0_data_check/docs/source_reference_check.md` | GitHub reference commit, paper/GitHub mismatch log. |
| `../자료조사/Re-image 요약.md` | report page map, paper result anchor, split/training/model detail, interpretation section. |
| `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | 최종 보고서 인용과 코드 주석 전에 눈으로 다시 확인해야 하는 원문 PDF. |
| `docs/data_loading_plan.md` | data loading과 image tensor convention. |
| `docs/label_construction_plan.md` | label rule과 horizon별 valid count. |
| `docs/split_normalization_plan.md` | split, period별 positive rate, train-only normalization. |
| `docs/baseline_cnn_implementation_plan.md` | GitHub식 I20 model 구조와 dilation mismatch. |
| `docs/training_loop_plan.md` | training setting, seed, checkpoint, paper-unreported choice. |
| `docs/kaggle_runner_plan.md` | Kaggle run mode, manifest, dataset/code snapshot 요구사항. |
| `docs/evaluation_prediction_plan.md` | metric, prediction schema, stock decile/H-L output plan. |
| `docs/gradcam_plan.md` | Figure 13 스타일 Grad-CAM output plan. |

최종 보고서에서 인용할 로컬 Re-image 요약 page map:
- image construction: pp.8-11.
- CNN structure and training: pp.12-21.
- U.S. stock experiments: pp.21-33.
- transfer learning: pp.34-40.
- interpretation section: pp.41-49.
- internet appendix robustness/alternative models/simulations: pp.55-67.

PDF extraction 제한:
- 현재 로컬 환경에는 `pdftotext/pdfinfo`가 없습니다.
- 최종 보고서 page number는 PDF를 눈으로 다시 확인한 뒤 citation-ready로 봅니다.

## 보고서 산출물

주 보고서 파일:

```text
reports/stage1_reproduction_report.md
```

보조 폴더:

```text
reports/tables/
reports/figures/sample_images/
reports/figures/gradcam/
reports/figures/training/
reports/figures/calibration/
```

이 보고서는 실제 산출물이 생긴 뒤 구현 gate `1-I12`에서 생성합니다.

## 필수 보고서 목차

1. Executive summary
2. 1단계에서 재현한 것
3. Data audit와 가능한 범위
4. Label construction과 leakage control
5. Split and normalization
6. Model architecture와 GitHub reference
7. Training and reproducibility setup
8. Classification results
9. Stock ranking, decile, H-L portfolio results
10. Grad-CAM interpretation figures
11. Re-image paper value와 비교
12. Known mismatches and limitations
13. Artifact inventory와 run manifest summary
14. 다음 단계: Stage 2 BTC extension

## 필수 Table

`reports/tables/` 아래 다음 CSV를 생성합니다.

| 파일 | 목적 |
| --- | --- |
| `stage1_dataset_summary.csv` | dataset period, image spec, window, horizon, file count, sample count. |
| `stage1_split_summary.csv` | horizon별 train/validation/test rows와 positive rate. |
| `stage1_training_summary.csv` | seed, epoch, best validation loss, checkpoint path, 가능하면 training time. |
| `stage1_classification_metrics.csv` | horizon/run mode별 accuracy, precision, recall, F1, AUC, Brier, log loss. |
| `stage1_majority_baseline_comparison.csv` | horizon/run mode별 accuracy와 majority-class baseline 비교. |
| `stage1_correlation_metrics.csv` | global/date-wise Pearson/Spearman prediction-return correlation. |
| `stage1_portfolio_metrics.csv` | equal-weight/value-weight decile과 H-L return, volatility, Sharpe, 구현 가능하면 turnover. |
| `stage1_paper_comparison.csv` | 비교 가능한 paper cell과 우리 결과를 나란히 배치. |
| `stage1_gradcam_samples.csv` | Figure 13 스타일 Grad-CAM에 사용한 sample metadata. |
| `stage1_artifact_manifest.csv` | config, checkpoint, prediction, metric, figure, run manifest 경로. |

Table 규칙:
- 모든 table에는 `experiment_name`, `image_window`, `target_horizon`,
  `target_return_name`, `run_mode`를 포함합니다.
- paper comparison table에는 `paper_source`, `paper_cell`,
  `comparison_status`를 포함합니다.
- metric이 없으면 `null` 또는 `NA`로 저장하고 reason column을 둡니다.

## 논문 비교 규칙

같은 조건끼리만 비교합니다.
- 가능한 경우 같은 image window.
- 같은 target horizon.
- 같은 stock cross-sectional setting.
- public data가 지원하는 범위 안에서 같은 full-spec 해석.

로컬 요약에서 확인된 직접 비교 후보:

| 1단계 실험 | Paper comparison source | 로컬 요약의 paper 값 | 보고서 처리 |
| --- | --- | --- | --- |
| `I20/R20` classification | Re-image U.S. results/Table 2, pp.21-33 | Accuracy `53.3%`; Corr.는 main cell 기준 약 `3.4%` 범위 | PDF 눈검증 뒤 직접 비교. |
| `I20/R60` classification | Re-image U.S. results/Table 2, pp.21-33 | Accuracy `53.2%` | PDF 눈검증 뒤 직접 비교. |
| `I20/R5` classification | Re-image paper, 현재 로컬 요약에서 정확한 accuracy cell 미추출 | PDF 확인 필요 | paper 값을 임의로 만들지 않음. |
| `I20/R20` EW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.21`, Sharpe `2.16` | decile 구현 convention이 맞을 때 비교. |
| `I20/R20` VW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.05`, Sharpe `0.49` | value-weighting과 annualization 확인 뒤 비교. |
| `I20/R60` EW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.05`, Sharpe `0.37` | annualization convention 확인 뒤 비교. |
| `I20/R5` EW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.84`, Sharpe `6.75` | turnover/annualization convention 확인 뒤 비교. |
| `I20/R5` VW H-L portfolio | Re-image U.S. portfolio results, pp.21-33 | Annual return `0.22`, Sharpe `1.74` | value-weighting convention 확인 뒤 비교. |

중요:
- 현재 public data만으로 full-paper reproduction이라고 주장하지 않습니다.
- `I20` 결과를 paper의 `I5` 또는 `I60` cell과 같은 실험처럼 비교하지 않습니다.
- local smoke-test metric은 paper value와 비교하지 않습니다.
- PDF 확인 후 값이나 page 위치가 달라지면 이 문서와 `docs/source_map.md`를 갱신합니다.

## 필수 Limitation 문구

최종 보고서에는 아래 문장을 그대로 또는 같은 의미로 포함합니다.

1. 1단계는 저자/public `monthly_20d` rendered image shard를 사용합니다. 사용 가능한
   이미지는 OHLC + 20-day MA + volume으로 해석되는 20-day full-spec image입니다.
2. 따라서 1단계가 직접 지원하는 실험은 `I20/R5`, `I20/R20`, `I20/R60`입니다.
3. 현재 파일은 stock `I5`, stock `I60`, A/B/C/D image-spec ablation을 직접 지원하지 않습니다.
4. `Ret_5d`, `Ret_20d`, `Ret_60d`는 개별 주식의 미래 보유수익률이지, feature map이나 portfolio return이 아닙니다.
5. Binary label은 future return sign에서 만듭니다. `return > 0`이면 class `1`, 아니면 class `0`입니다.
6. H-L portfolio 결과를 보고한다면, 이는 예측 이후 date별 cross-sectional ranking으로 구성한 것입니다.
7. BTC는 단일 자산이므로 stock H-L decile setup을 그대로 재사용할 수 없습니다.
8. GitHub I20 model은 세 convolution layer 모두에 dilation을 적용하지만, 로컬 논문 요약은 first-layer dilation을 강조합니다. 사용자가 바꾸지 않는 한 1단계는 확인한 GitHub model core를 따릅니다.
9. Kaggle full `full_paper_style` run이 primary reproduction run입니다. Local smoke-test metric은 reproduction result가 아닙니다.
10. Grad-CAM figure는 학습된 checkpoint에서 만든 post-hoc explanation입니다. 이것은 class-discriminative heatmap이지 raw feature map이나 causal proof가 아닙니다.

## 필수 Figure

| Figure | Source output |
| --- | --- |
| Public I20 sample image | `outputs/figures/sample_images/` |
| Train/validation loss curves | `outputs/figures/training/` |
| Calibration curve, 구현 시 | `outputs/figures/calibration/` |
| `I20/R20` Figure 13 스타일 Grad-CAM | `outputs/figures/gradcam/stage1_i20_r20/...` |
| `I20/R5`, `I20/R60` 확장 Grad-CAM | `outputs/figures/gradcam/stage1_i20_r5/...`, `stage1_i20_r60/...` |

## Artifact Inventory

최종 보고서에는 다음을 link 또는 list합니다.
- 사용 config
- Kaggle run manifest
- dataset root/name/version
- code snapshot 또는 Git commit
- split and normalization JSON
- checkpoints
- seed별 prediction CSV
- averaged prediction CSV
- metrics JSON
- report tables
- Grad-CAM figures

## 구현 단계로 넘길 것

- 실제 report table 생성.
- paper-style 5-seed Kaggle job 실행.
- 최종 metrics와 portfolio output 계산.
- Grad-CAM figure 생성.
- `reports/stage1_reproduction_report.md` 작성.
