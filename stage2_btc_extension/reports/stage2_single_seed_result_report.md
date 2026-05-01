# Stage 2 Single-Seed BTC Baseline Result Report

## English

### Scope

This report summarizes the Stage 2 BTC extension baseline grid for one seed.

- Run seed: `42`
- Dataset: BTC daily OHLCV from Kaggle
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Test period: `2021-01-01` to `2024-12-31`
- Model family: Stage-1/Stock_CNN-style CNN variants
- Grid size: `36` runs
  - Image windows/models: `I5`, `I20`, `I60`
  - Return horizons: `R5`, `R20`, `R60`
  - Image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`

Important limitation: this is a single-seed result. The five-seed run is still
pending, so the current result should be treated as a first diagnostic, not as a
final stability claim.

### Main Result

The strongest configuration is:

`I60 / R20 / ohlc_ma_vb`

| Rank | Image window | Return horizon | Image spec | Accuracy | AUC | Accuracy - majority | Long/flat Sharpe net |
|---:|---:|---:|:---|---:|---:|---:|---:|
| 1 | 60 | 20 | `ohlc_ma_vb` | 0.6031 | 0.6169 | 0.0618 | 3.9342 |
| 2 | 60 | 20 | `ohlc_vb` | 0.5947 | 0.5828 | 0.0534 | 3.7747 |
| 3 | 60 | 20 | `ohlc_ma` | 0.5711 | 0.5608 | 0.0298 | 3.5635 |
| 4 | 20 | 20 | `ohlc_vb` | 0.5455 | 0.5309 | 0.0042 | 3.2430 |
| 5 | 60 | 60 | `ohlc` | 0.5432 | 0.4891 | 0.0000 | 5.0593 |

The first three accuracy and AUC leaders are all `I60/R20` variants. This is the
clearest signal in the single-seed run.

### Aggregate Patterns

Average by image window:

| Image window | Accuracy | Accuracy - majority | AUC | Long/flat Sharpe net | Long/short Sharpe net |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.5140 | -0.0229 | 0.5075 | 2.4606 | 0.6271 |
| 20 | 0.5173 | -0.0196 | 0.5120 | 2.7100 | 1.0938 |
| 60 | 0.5439 | 0.0071 | 0.5323 | 3.1815 | 2.3824 |

Average by return horizon:

| Return horizon | Accuracy | Accuracy - majority | AUC | Long/flat Sharpe net | Long/short Sharpe net |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.5157 | -0.0104 | 0.5094 | 1.3350 | 0.5892 |
| 20 | 0.5427 | 0.0014 | 0.5277 | 3.0834 | 2.3398 |
| 60 | 0.5167 | -0.0265 | 0.5147 | 3.9338 | 1.1743 |

Average by image specification:

| Image spec | Accuracy | Accuracy - majority | AUC | Long/flat Sharpe net | Long/short Sharpe net |
|:---|---:|---:|---:|---:|---:|
| `ohlc` | 0.5183 | -0.0186 | 0.5010 | 2.6913 | 1.4255 |
| `ohlc_ma` | 0.5253 | -0.0116 | 0.5167 | 2.8659 | 1.7049 |
| `ohlc_ma_vb` | 0.5263 | -0.0105 | 0.5312 | 2.6894 | 0.8293 |
| `ohlc_vb` | 0.5303 | -0.0066 | 0.5201 | 2.8896 | 1.5115 |

### Interpretation

- `I60` is stronger than `I5` and `I20` on average.
- `R20` is the most favorable return horizon in this single-seed run.
- `ohlc_ma_vb` gives the best individual result and the best average AUC, but
  `ohlc_vb` has the best average accuracy.
- Only `5` of `36` configurations beat the majority-class baseline, so the
  result is concentrated rather than uniformly strong.
- Trading metrics are useful as a BTC time-series supplement, but they should
  not be read without classification metrics. For example, `I60/R60/ohlc` has
  the highest long/flat Sharpe net, but its AUC is below `0.50`, so that result
  may reflect BTC exposure/regime effects rather than a robust classifier.

### Figure-13-Style Grad-CAM

The report should include a Re-Imagining Figure-13-style Grad-CAM for the best
single-seed configuration:

`I60 / R20 / ohlc_ma_vb / seed 42`

Required figure layout:

- `Predicted Up` (`pred_class = 1`): 10 BTC chart images
- `Predicted Down` (`pred_class = 0`): 10 BTC chart images
- Total: 20 original chart images plus layer-wise Grad-CAM rows

Generation cell:
[`../notebooks/kaggle_stage2_best_gradcam_10_one_cell.md`](../notebooks/kaggle_stage2_best_gradcam_10_one_cell.md)

Expected final artifacts after running the Kaggle cell:

- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass.png`
- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass_samples.csv`

Current local limitation: the downloaded `stage2_result_save` folder contains
only the quick `2`-per-class Grad-CAM preview and does not include the checkpoint
or prediction CSV needed to regenerate `10` per class locally. Therefore the
10-per-class figure must be generated in Kaggle from the saved output backup zip,
then copied into `reports/figures/gradcam/` before this Stage 2 report is treated
as final.

The heatmap is computed from the predicted class logit. The true label and
correctness are displayed as interpretation metadata, not as the Grad-CAM target.

### Tracked Result Artifacts

- [`tables/stage2_single_seed_seed_level_results.csv`](tables/stage2_single_seed_seed_level_results.csv)
- [`tables/stage2_single_seed_summary_sorted_by_accuracy.csv`](tables/stage2_single_seed_summary_sorted_by_accuracy.csv)
- [`tables/stage2_single_seed_pivot_accuracy_mean.csv`](tables/stage2_single_seed_pivot_accuracy_mean.csv)
- [`tables/stage2_single_seed_top20_accuracy.csv`](tables/stage2_single_seed_top20_accuracy.csv)
- [`tables/stage2_single_seed_top20_roc_auc.csv`](tables/stage2_single_seed_top20_roc_auc.csv)
- [`tables/stage2_single_seed_top20_long_flat_sharpe_net.csv`](tables/stage2_single_seed_top20_long_flat_sharpe_net.csv)

### Next Step

Before treating Stage 2 as final, run five seeds for at least the leading group:

1. `I60/R20/ohlc_ma_vb`
2. `I60/R20/ohlc_vb`
3. `I60/R20/ohlc_ma`
4. `I60/R20/ohlc`

Then compare mean/std and check whether the `I60/R20` advantage survives seed
variation.

## 한국어

### 범위

이 문서는 Stage 2 BTC 확장 baseline grid의 single-seed 결과를 정리합니다.

- Run seed: `42`
- Dataset: Kaggle BTC daily OHLCV
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Test period: `2021-01-01` to `2024-12-31`
- Model family: Stage-1/Stock_CNN-style CNN variants
- Grid size: `36` runs
  - Image window/model: `I5`, `I20`, `I60`
  - Return horizon: `R5`, `R20`, `R60`
  - Image spec: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`

중요한 제한사항: 이 결과는 seed `42` 한 번만 돌린 결과입니다. 5-seed run은 아직
예정이므로, 현재 결과는 최종 안정성 결론이 아니라 1차 diagnostic result로 봐야
합니다.

### 핵심 결과

가장 강한 조합은 다음입니다.

`I60 / R20 / ohlc_ma_vb`

| Rank | Image window | Return horizon | Image spec | Accuracy | AUC | Accuracy - majority | Long/flat Sharpe net |
|---:|---:|---:|:---|---:|---:|---:|---:|
| 1 | 60 | 20 | `ohlc_ma_vb` | 0.6031 | 0.6169 | 0.0618 | 3.9342 |
| 2 | 60 | 20 | `ohlc_vb` | 0.5947 | 0.5828 | 0.0534 | 3.7747 |
| 3 | 60 | 20 | `ohlc_ma` | 0.5711 | 0.5608 | 0.0298 | 3.5635 |
| 4 | 20 | 20 | `ohlc_vb` | 0.5455 | 0.5309 | 0.0042 | 3.2430 |
| 5 | 60 | 60 | `ohlc` | 0.5432 | 0.4891 | 0.0000 | 5.0593 |

Accuracy와 AUC 상위 3개가 모두 `I60/R20` variant입니다. single-seed 기준으로는 이
부분이 가장 뚜렷한 signal입니다.

### 축별 패턴

Image window별 평균:

| Image window | Accuracy | Accuracy - majority | AUC | Long/flat Sharpe net | Long/short Sharpe net |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.5140 | -0.0229 | 0.5075 | 2.4606 | 0.6271 |
| 20 | 0.5173 | -0.0196 | 0.5120 | 2.7100 | 1.0938 |
| 60 | 0.5439 | 0.0071 | 0.5323 | 3.1815 | 2.3824 |

Return horizon별 평균:

| Return horizon | Accuracy | Accuracy - majority | AUC | Long/flat Sharpe net | Long/short Sharpe net |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.5157 | -0.0104 | 0.5094 | 1.3350 | 0.5892 |
| 20 | 0.5427 | 0.0014 | 0.5277 | 3.0834 | 2.3398 |
| 60 | 0.5167 | -0.0265 | 0.5147 | 3.9338 | 1.1743 |

Image specification별 평균:

| Image spec | Accuracy | Accuracy - majority | AUC | Long/flat Sharpe net | Long/short Sharpe net |
|:---|---:|---:|---:|---:|---:|
| `ohlc` | 0.5183 | -0.0186 | 0.5010 | 2.6913 | 1.4255 |
| `ohlc_ma` | 0.5253 | -0.0116 | 0.5167 | 2.8659 | 1.7049 |
| `ohlc_ma_vb` | 0.5263 | -0.0105 | 0.5312 | 2.6894 | 0.8293 |
| `ohlc_vb` | 0.5303 | -0.0066 | 0.5201 | 2.8896 | 1.5115 |

### 해석

- 평균적으로 `I60`이 `I5`, `I20`보다 강합니다.
- 현재 single-seed 결과에서는 `R20` horizon이 가장 유리합니다.
- 개별 최고 성능과 평균 AUC는 `ohlc_ma_vb`가 좋고, 평균 accuracy는 `ohlc_vb`가
  조금 더 높습니다.
- 36개 조합 중 majority-class baseline을 이긴 조합은 5개뿐입니다. 따라서 모든
  설정에서 성능이 좋아진 것이 아니라, signal이 `I60/R20` 주변에 집중되어 있다고
  해석해야 합니다.
- Trading metric은 BTC 단일 자산 time-series setting에서 보조 지표로 유용하지만,
  classification metric과 분리해서 보면 위험합니다. 예를 들어 `I60/R60/ohlc`는
  long/flat Sharpe net이 가장 높지만 AUC가 `0.50`보다 낮으므로, 강한 분류 signal이라기보다
  BTC exposure/regime 효과일 수 있습니다.

### Figure 13 스타일 Grad-CAM

결과 보고에는 single-seed 기준 best configuration에 대해 Re-Imagining Figure 13
스타일 Grad-CAM을 포함해야 합니다.

`I60 / R20 / ohlc_ma_vb / seed 42`

필수 figure 구성:

- `Predicted Up` (`pred_class = 1`): BTC chart image 10개
- `Predicted Down` (`pred_class = 0`): BTC chart image 10개
- 총 20개 원본 chart image와 layer별 Grad-CAM row

생성 cell:
[`../notebooks/kaggle_stage2_best_gradcam_10_one_cell.md`](../notebooks/kaggle_stage2_best_gradcam_10_one_cell.md)

Kaggle cell 실행 후 들어와야 하는 최종 산출물:

- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass.png`
- `reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_10perclass_samples.csv`

현재 로컬 제한사항: 내려받은 `stage2_result_save`에는 빠른 확인용 `2`-per-class
Grad-CAM preview만 있고, 로컬에서 `10`개씩 다시 생성하는 데 필요한 checkpoint와
prediction CSV는 없습니다. 따라서 Kaggle에 남아 있는 saved output backup zip에서
10-per-class figure를 생성한 뒤 `reports/figures/gradcam/`에 복사해야 이 Stage 2
결과 보고를 최종본으로 볼 수 있습니다.

Heatmap은 predicted class logit 기준으로 계산합니다. true label과 correct 여부는
Grad-CAM target이 아니라 해석용 metadata로 함께 표시합니다.

### GitHub에 포함한 결과 파일

- [`tables/stage2_single_seed_seed_level_results.csv`](tables/stage2_single_seed_seed_level_results.csv)
- [`tables/stage2_single_seed_summary_sorted_by_accuracy.csv`](tables/stage2_single_seed_summary_sorted_by_accuracy.csv)
- [`tables/stage2_single_seed_pivot_accuracy_mean.csv`](tables/stage2_single_seed_pivot_accuracy_mean.csv)
- [`tables/stage2_single_seed_top20_accuracy.csv`](tables/stage2_single_seed_top20_accuracy.csv)
- [`tables/stage2_single_seed_top20_roc_auc.csv`](tables/stage2_single_seed_top20_roc_auc.csv)
- [`tables/stage2_single_seed_top20_long_flat_sharpe_net.csv`](tables/stage2_single_seed_top20_long_flat_sharpe_net.csv)

### 다음 단계

Stage 2를 최종화하기 전에는 최소한 아래 leading group에 대해 5-seed를 먼저 확인하는
것이 좋습니다.

1. `I60/R20/ohlc_ma_vb`
2. `I60/R20/ohlc_vb`
3. `I60/R20/ohlc_ma`
4. `I60/R20/ohlc`

그 다음 mean/std를 비교해서 `I60/R20` 우위가 seed 변화에도 유지되는지 확인합니다.
