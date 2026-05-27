# Stage 2 Selected Five-Seed Robustness Result Report

## English

### Scope

This report summarizes the selected Stage 2 five-seed robustness check.

- Dataset: BTC daily OHLCV from Kaggle
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Test period: `2021-01-01` to `2024-12-31`
- Selected image windows: `I20`, `I60`
- Selected return horizon: `R20`
- Image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- Seeds: `42`, `43`, `44`, `45`, `46`
- Total runs: `40`
- Run status: `40/40 ok`
- Total Kaggle runtime recorded in the run summary: about `61.5` minutes

This is not the full Stage 2 five-seed grid. It is a targeted robustness check
for the strongest single-seed family found earlier: `R20`, especially `I60/R20`.
The `I5` family was not included in this five-seed expansion because the
earlier seed-42 screening was weak: `I5` was below the majority-class baseline
on average, and the best `I5` accuracy was only about `0.524`.

### Main Result

The best five-seed mean configuration is still:

`I60 / R20 / ohlc_ma_vb`

| Image window | Image spec | Accuracy mean | Accuracy std | Accuracy - majority | ROC-AUC mean | ROC-AUC std | F1 mean | Long/flat Sharpe net | Long/short Sharpe net |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | `ohlc_ma_vb` | 0.5793 | 0.0182 | 0.0380 | 0.5849 | 0.0233 | 0.6511 | 3.4423 | 2.4078 |
| 60 | `ohlc_vb` | 0.5674 | 0.0173 | 0.0261 | 0.5612 | 0.0186 | 0.6586 | 3.1368 | 2.1340 |
| 60 | `ohlc` | 0.5581 | 0.0152 | 0.0168 | 0.5602 | 0.0156 | 0.6810 | 3.0834 | 2.4596 |
| 60 | `ohlc_ma` | 0.5575 | 0.0233 | 0.0162 | 0.5645 | 0.0163 | 0.6274 | 3.0741 | 1.8055 |
| 20 | `ohlc_ma_vb` | 0.5117 | 0.0211 | -0.0296 | 0.5047 | 0.0297 | 0.5277 | 2.3655 | 0.4074 |
| 20 | `ohlc_vb` | 0.5113 | 0.0259 | -0.0300 | 0.5039 | 0.0283 | 0.5112 | 2.3688 | 0.3540 |
| 20 | `ohlc_ma` | 0.5067 | 0.0178 | -0.0346 | 0.4993 | 0.0271 | 0.5417 | 2.3378 | 0.4817 |
| 20 | `ohlc` | 0.5059 | 0.0145 | -0.0354 | 0.4986 | 0.0285 | 0.5286 | 2.2067 | 0.1758 |

### Stability Check

Accuracy by seed:

| Config | 42 | 43 | 44 | 45 | 46 |
|:---|---:|---:|---:|---:|---:|
| `I20/R20/ohlc` | 0.5239 | 0.4913 | 0.4913 | 0.5156 | 0.5073 |
| `I20/R20/ohlc_ma` | 0.5246 | 0.5031 | 0.4948 | 0.5253 | 0.4858 |
| `I20/R20/ohlc_ma_vb` | 0.5406 | 0.5239 | 0.4858 | 0.5017 | 0.5066 |
| `I20/R20/ohlc_vb` | 0.5455 | 0.5316 | 0.4851 | 0.4934 | 0.5010 |
| `I60/R20/ohlc` | 0.5420 | 0.5413 | 0.5697 | 0.5656 | 0.5718 |
| `I60/R20/ohlc_ma` | 0.5711 | 0.5323 | 0.5579 | 0.5378 | 0.5885 |
| `I60/R20/ohlc_ma_vb` | 0.6031 | 0.5746 | 0.5628 | 0.5628 | 0.5933 |
| `I60/R20/ohlc_vb` | 0.5947 | 0.5642 | 0.5468 | 0.5628 | 0.5684 |

ROC-AUC by seed:

| Config | 42 | 43 | 44 | 45 | 46 |
|:---|---:|---:|---:|---:|---:|
| `I20/R20/ohlc` | 0.4905 | 0.4662 | 0.5446 | 0.4939 | 0.4978 |
| `I20/R20/ohlc_ma` | 0.5100 | 0.4661 | 0.5390 | 0.4908 | 0.4907 |
| `I20/R20/ohlc_ma_vb` | 0.5378 | 0.4808 | 0.5028 | 0.4708 | 0.5314 |
| `I20/R20/ohlc_vb` | 0.5309 | 0.4725 | 0.5070 | 0.4774 | 0.5316 |
| `I60/R20/ohlc` | 0.5441 | 0.5442 | 0.5794 | 0.5676 | 0.5657 |
| `I60/R20/ohlc_ma` | 0.5608 | 0.5438 | 0.5704 | 0.5594 | 0.5880 |
| `I60/R20/ohlc_ma_vb` | 0.6169 | 0.5833 | 0.5610 | 0.5651 | 0.5979 |
| `I60/R20/ohlc_vb` | 0.5828 | 0.5573 | 0.5468 | 0.5412 | 0.5782 |

### Interpretation

- The `I60/R20` advantage survives the five-seed check. All four `I60/R20`
  image specs beat the majority-class baseline on average.
- The `I5` family was screened out before this robustness run because the
  single-seed grid did not show enough signal to justify the immediate
  five-seed cost.
- The `I20/R20` group does not survive the five-seed check. All four `I20/R20`
  image specs are below the majority-class baseline on average.
- The earlier single-seed `I60/R20/ohlc_ma_vb` result with accuracy `0.6031`
  was high, but it was not an isolated one-off. Its five-seed mean is still the
  best in this selected robustness check: `0.5793`.
- `ohlc_ma_vb` is the strongest selected configuration by both accuracy mean
  and ROC-AUC mean. `ohlc_vb` is the strongest simpler alternative if the moving
  average overlay is removed.
- This result supports using `I60/R20/ohlc_ma_vb` as the primary Stage 2
  baseline for Stage 4 FiLM comparisons, with `I60/R20/ohlc_vb` as a useful
  secondary ablation candidate.

### Remaining Limitations

- This is a selected `40`-run five-seed check, not the full `180`-run Stage 2
  five-seed grid.
- Only `R20` was tested here. The earlier `R5` and `R60` groups still have only
  single-seed coverage.
- The dataset is still one BTC time series. The result is stronger than the
  single-seed diagnostic, but final thesis claims should still avoid language
  stronger than "selected five-seed robustness check" unless the full grid is
  later completed.

### Tracked Result Artifacts

- [`tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv`](tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv)
- [`tables/stage2_i20_i60_r20_five_seed_seed_results.csv`](tables/stage2_i20_i60_r20_five_seed_seed_results.csv)
- [`tables/stage2_i20_i60_r20_five_seed_run_summary.csv`](tables/stage2_i20_i60_r20_five_seed_run_summary.csv)

## 한국어

### 범위

이 문서는 Stage 2 BTC baseline 중 선별한 5-seed robustness check 결과를 정리합니다.

- Dataset: Kaggle BTC daily OHLCV
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`
- Test period: `2021-01-01` to `2024-12-31`
- 선택한 image window: `I20`, `I60`
- 선택한 return horizon: `R20`
- Image spec: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- Seeds: `42`, `43`, `44`, `45`, `46`
- 총 run 수: `40`
- Run status: `40/40 ok`
- Run summary 기준 총 Kaggle 실행 시간: 약 `61.5`분

이 결과는 Stage 2 전체 5-seed grid가 아닙니다. 이전 single-seed에서 가장 강했던
`R20`, 특히 `I60/R20` 계열이 seed가 바뀌어도 유지되는지 확인한 선별 robustness
check입니다.
`I5` 계열은 이 five-seed 확장에 포함하지 않았습니다. 이전 seed `42`
screening에서 `I5`는 평균적으로 majority-class baseline보다 낮았고, 가장 좋은
`I5` accuracy도 약 `0.524` 수준이었습니다.

### 핵심 결과

5-seed 평균 기준 가장 좋은 조합은 여전히 다음입니다.

`I60 / R20 / ohlc_ma_vb`

| Image window | Image spec | Accuracy mean | Accuracy std | Accuracy - majority | ROC-AUC mean | ROC-AUC std | F1 mean | Long/flat Sharpe net | Long/short Sharpe net |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | `ohlc_ma_vb` | 0.5793 | 0.0182 | 0.0380 | 0.5849 | 0.0233 | 0.6511 | 3.4423 | 2.4078 |
| 60 | `ohlc_vb` | 0.5674 | 0.0173 | 0.0261 | 0.5612 | 0.0186 | 0.6586 | 3.1368 | 2.1340 |
| 60 | `ohlc` | 0.5581 | 0.0152 | 0.0168 | 0.5602 | 0.0156 | 0.6810 | 3.0834 | 2.4596 |
| 60 | `ohlc_ma` | 0.5575 | 0.0233 | 0.0162 | 0.5645 | 0.0163 | 0.6274 | 3.0741 | 1.8055 |
| 20 | `ohlc_ma_vb` | 0.5117 | 0.0211 | -0.0296 | 0.5047 | 0.0297 | 0.5277 | 2.3655 | 0.4074 |
| 20 | `ohlc_vb` | 0.5113 | 0.0259 | -0.0300 | 0.5039 | 0.0283 | 0.5112 | 2.3688 | 0.3540 |
| 20 | `ohlc_ma` | 0.5067 | 0.0178 | -0.0346 | 0.4993 | 0.0271 | 0.5417 | 2.3378 | 0.4817 |
| 20 | `ohlc` | 0.5059 | 0.0145 | -0.0354 | 0.4986 | 0.0285 | 0.5286 | 2.2067 | 0.1758 |

### Seed별 안정성

Accuracy by seed:

| Config | 42 | 43 | 44 | 45 | 46 |
|:---|---:|---:|---:|---:|---:|
| `I20/R20/ohlc` | 0.5239 | 0.4913 | 0.4913 | 0.5156 | 0.5073 |
| `I20/R20/ohlc_ma` | 0.5246 | 0.5031 | 0.4948 | 0.5253 | 0.4858 |
| `I20/R20/ohlc_ma_vb` | 0.5406 | 0.5239 | 0.4858 | 0.5017 | 0.5066 |
| `I20/R20/ohlc_vb` | 0.5455 | 0.5316 | 0.4851 | 0.4934 | 0.5010 |
| `I60/R20/ohlc` | 0.5420 | 0.5413 | 0.5697 | 0.5656 | 0.5718 |
| `I60/R20/ohlc_ma` | 0.5711 | 0.5323 | 0.5579 | 0.5378 | 0.5885 |
| `I60/R20/ohlc_ma_vb` | 0.6031 | 0.5746 | 0.5628 | 0.5628 | 0.5933 |
| `I60/R20/ohlc_vb` | 0.5947 | 0.5642 | 0.5468 | 0.5628 | 0.5684 |

ROC-AUC by seed:

| Config | 42 | 43 | 44 | 45 | 46 |
|:---|---:|---:|---:|---:|---:|
| `I20/R20/ohlc` | 0.4905 | 0.4662 | 0.5446 | 0.4939 | 0.4978 |
| `I20/R20/ohlc_ma` | 0.5100 | 0.4661 | 0.5390 | 0.4908 | 0.4907 |
| `I20/R20/ohlc_ma_vb` | 0.5378 | 0.4808 | 0.5028 | 0.4708 | 0.5314 |
| `I20/R20/ohlc_vb` | 0.5309 | 0.4725 | 0.5070 | 0.4774 | 0.5316 |
| `I60/R20/ohlc` | 0.5441 | 0.5442 | 0.5794 | 0.5676 | 0.5657 |
| `I60/R20/ohlc_ma` | 0.5608 | 0.5438 | 0.5704 | 0.5594 | 0.5880 |
| `I60/R20/ohlc_ma_vb` | 0.6169 | 0.5833 | 0.5610 | 0.5651 | 0.5979 |
| `I60/R20/ohlc_vb` | 0.5828 | 0.5573 | 0.5468 | 0.5412 | 0.5782 |

### 해석

- `I60/R20` 우위는 5-seed 확인에서도 유지됩니다. `I60/R20`의 네 가지 image spec은
  모두 평균적으로 majority-class baseline을 넘었습니다.
- `I5` 계열은 single-seed grid에서 충분한 signal을 보이지 않았기 때문에 이번
  selected five-seed robustness run에서는 우선순위에서 제외했습니다.
- `I20/R20`은 5-seed에서 살아남지 못했습니다. 네 가지 image spec 모두 평균적으로
  majority-class baseline보다 낮습니다.
- 이전 single-seed에서 `I60/R20/ohlc_ma_vb`가 accuracy `0.6031`로 매우 높게
  나왔는데, 이번 결과를 보면 완전히 우연한 한 번의 spike로 보기는 어렵습니다.
  5-seed 평균도 `0.5793`으로 이번 선별 실험에서 가장 높습니다.
- `ohlc_ma_vb`는 accuracy mean과 ROC-AUC mean 모두에서 가장 좋은 조합입니다.
  단순한 대안으로는 `ohlc_vb`가 가장 좋습니다.
- Stage 4 FiLM 비교의 primary baseline은 `I60/R20/ohlc_ma_vb`로 두는 것이 현재
  결과 기준으로 가장 방어 가능합니다. 보조 ablation 후보로는 `I60/R20/ohlc_vb`가
  적절합니다.

### 남은 제한사항

- 이 결과는 선별한 `40`개 run의 5-seed 확인이지, 전체 `180`개 Stage 2 5-seed grid가
  아닙니다.
- 여기서는 `R20`만 확인했습니다. 기존 `R5`, `R60` 그룹은 아직 single-seed 결과입니다.
- 데이터는 여전히 BTC 단일 시계열입니다. single-seed보다는 훨씬 강한 근거가 되었지만,
  전체 grid를 돌리기 전에는 최종 논문 문장에서 "selected five-seed robustness check"라는
  표현을 유지하는 것이 안전합니다.

### 저장된 결과 파일

- [`tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv`](tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv)
- [`tables/stage2_i20_i60_r20_five_seed_seed_results.csv`](tables/stage2_i20_i60_r20_five_seed_seed_results.csv)
- [`tables/stage2_i20_i60_r20_five_seed_run_summary.csv`](tables/stage2_i20_i60_r20_five_seed_run_summary.csv)
