# 4-4 Stage 2/Stage 3 Dependency and Baseline Output Review

## English

Status: complete for planning.

Purpose:
- lock the Stage 4 baseline input/model/result target;
- identify exactly which Stage 2 code should be reused;
- define how Stage 3 Linear should be used in the Stage 4 argument;
- avoid using provisional or missing artifacts as hard dependencies.

## Files Reviewed

Stage 2 result tables:
- `stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv`
- `stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_seed_results.csv`
- `stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_run_summary.csv`

Stage 2 selected single-seed metadata:
- `stage2_btc_extension/stage2_result_save/metadata/outputs/stage2/run_manifests/stage2_i60_ohlc_ma_vb_r20/seed_42/run_manifest.json`
- `stage2_btc_extension/stage2_result_save/metadata/outputs/stage2/metrics/stage2_i60_ohlc_ma_vb_r20/seed_42/train_metadata.json`
- `stage2_btc_extension/stage2_result_save/metadata/outputs/stage2/metrics/stage2_i60_ohlc_ma_vb_r20/seed_42/test_metrics.json`

Stage 2 code dependencies:
- `stage2_btc_extension/src/stage2_btc/data/ohlcv.py`
- `stage2_btc_extension/src/stage2_btc/data/label_split.py`
- `stage2_btc_extension/src/stage2_btc/imaging/ohlcv_image.py`
- `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`
- `stage2_btc_extension/src/stage2_btc/evaluation/prediction.py`
- `stage2_btc_extension/src/stage2_btc/evaluation/backtest.py`
- `stage2_btc_extension/src/stage2_btc/interpretability/gradcam.py`

Stage 3 review files:
- `stage3_linear_adapter/README.md`
- `stage3_linear_adapter/docs/linear_adapter_design.md`
- `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py`

## Stage 2 Baseline Lock

Stage 4 primary baseline is locked to:

```text
image_window = 60
return_horizon = 20
image_spec = ohlc_ma_vb
experiment = stage2_i60_ohlc_ma_vb_r20
```

Reason:
- It is the strongest selected five-seed Stage 2 configuration among the
  reviewed `I20/I60 x R20 x 4 image spec` grid.
- It is stronger than using only the seed-42 single-run result as the selection
  basis.

Five-seed summary:

| Metric | Value |
| --- | ---: |
| Accuracy mean | `0.579320` |
| Accuracy std | `0.018218` |
| Accuracy minus majority mean | `0.038029` |
| ROC-AUC mean | `0.584862` |
| ROC-AUC std | `0.023250` |
| Average precision mean | `0.611256` |
| F1 mean | `0.651071` |
| Brier score mean | `0.274337` |
| Long/flat Sharpe net mean | `3.442312` |
| Long/short Sharpe net mean | `2.407759` |

Seed-level sanity check for the selected baseline:

| Seed | Accuracy | ROC-AUC | Accuracy - majority |
| ---: | ---: | ---: | ---: |
| 42 | `0.603053` | `0.616950` | `0.061763` |
| 43 | `0.574601` | `0.583347` | `0.033310` |
| 44 | `0.562804` | `0.560969` | `0.021513` |
| 45 | `0.562804` | `0.565132` | `0.021513` |
| 46 | `0.593338` | `0.597911` | `0.052047` |

Interpretation:
- Seed 42 is strong, but not the only evidence.
- The five-seed mean remains above majority-class accuracy.
- Stage 4 should compare against the five-seed Stage 2 summary when possible.

## Stage 2 Single-Seed Metadata Needed for Stage 4

The seed-42 run manifest for `stage2_i60_ohlc_ma_vb_r20` records:

| Item | Value |
| --- | --- |
| Source file | `/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv` |
| Device | `cuda` |
| Model params | `2,952,962` |
| Train rows | `671` |
| Validation rows | `287` |
| Test rows | `1,441` |
| Test date range | `2021-01-01` to `2024-12-11` |
| Test positive rate / majority accuracy | `0.541291` |
| Train pixel mean | `0.0760317485` |
| Train pixel std | `0.2650489044` |
| Batch size | `128` |
| Learning rate | `1e-5` |
| Early stopping | `val_loss`, patience `2` |
| Best epoch | `10` |
| Stopped epoch | `12` |

Stage 4 must keep the Stage 2 data rules fixed:
- same BTC source file;
- same `I60/R20/ohlc_ma_vb` image construction;
- same label definition, split, and train-only image normalization;
- same classification/trading metric functions;
- same Grad-CAM convention.

## Stage 2 Code Reuse Decision

Stage 4 should not fork/rewrite the BTC pipeline. Reuse or mirror the Stage 2
modules.

Required fixed pieces:
- OHLCV parser and canonical column handling.
- BTC image generator.
- Label and split builder.
- `StockCNN` I60 variant.
- Prediction metric and trading metric code.
- Grad-CAM hook policy.

Stage 4-specific additions:
- context feature builder;
- train-only context scaler;
- context MLP encoder;
- concat/gating/FiLM model wrappers;
- context/gate/gamma/beta logging.

## Stage 2 Model Dependency

For `I60`, Stage 2 uses:

```text
input_shape = (1, 96, 180)
channels = (64, 128, 256, 512)
expected_flatten_dim = 184,320
expected_num_params = 2,952,962
```

The CNN block is:

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

Stage 4 FiLM should modify the block as planned:

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

This means Stage 4 modulation is not a post-flatten adapter. It is block-level
feature modulation.

## Stage 3 Dependency Review

Stage 3 Linear is a comparison result, not a code dependency for Stage 4.

Stage 3 model insertion:

```text
image
  -> fixed Stage 2 Stock_CNN convolution blocks
  -> flatten
  -> Linear(flatten_dim, adapter_dim=128, bias=False)
  -> Linear(adapter_dim, 2, bias=False)
  -> logits
```

Preliminary Stage 3 result for the Stage 2 best single-seed configuration:

| Model | Config | Seed | Accuracy | Majority accuracy | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage 2 baseline | `I60/R20/ohlc_ma_vb` | 42 | `0.603053` | `0.541291` | `0.616950` |
| Stage 3 Linear | `I60/R20/ohlc_ma_vb`, adapter dim `128` | 42 | `0.541291` | `0.541291` | `0.522101` |

Interpretation:
- The first Linear test drops to majority-class-level accuracy.
- This supports the Stage 4 argument that simply adding a dense post-flatten
  parameter block is not enough.
- This is still a preliminary Stage 3 result because the full Stage 3 grid is
  not complete.
- Stage 4 should reference Stage 3 as a negative/simple-parameter ablation, not
  as a model component to reuse.

## Artifact Availability

Available locally:
- Stage 2 selected five-seed tables.
- Stage 2 seed-42 run metadata and metrics for `I60/R20/ohlc_ma_vb`.
- Stage 2 selected Grad-CAM preview:
  `stage2_btc_extension/reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam.png`
- Stage 2 selected Grad-CAM sample metadata:
  `stage2_btc_extension/reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_samples.csv`
- Stage 3 Linear code and preliminary result documentation.

Not required for GitHub:
- `.pt` checkpoints;
- large prediction CSVs;
- raw output zips.

For Stage 4 Kaggle execution, checkpoints can be generated by the Stage 4
runner. Stage 4 should not require local checkpoint files inside GitHub.

## 4-4 Decision

Proceed to 4-5 with these locked dependencies:

```text
Stage 4 primary baseline:
    Stage 2 I60/R20/ohlc_ma_vb StockCNN

Primary comparison target:
    Stage 2 selected five-seed mean
    accuracy = 0.579320
    ROC-AUC = 0.584862

Stage 3 role:
    negative/simple-parameter ablation
    not a Stage 4 architecture dependency

Stage 4 model changes:
    add context branch and concat/gating/FiLM modulation
    keep Stage 2 data/image/label/split/evaluation fixed
```

## 한국어

상태: 계획 단계 완료.

목적:
- Stage 4 baseline input/model/result target을 고정합니다.
- Stage 2에서 어떤 코드를 재사용해야 하는지 명확히 합니다.
- Stage 3 Linear를 Stage 4 논리에서 어떻게 쓸지 정합니다.
- provisional 결과나 없는 artifact를 hard dependency로 쓰지 않게 합니다.

## 확인한 파일

Stage 2 결과 표:
- `stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_mean_std_results.csv`
- `stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_seed_results.csv`
- `stage2_btc_extension/reports/tables/stage2_i20_i60_r20_five_seed_run_summary.csv`

Stage 2 selected single-seed metadata:
- `stage2_btc_extension/stage2_result_save/metadata/outputs/stage2/run_manifests/stage2_i60_ohlc_ma_vb_r20/seed_42/run_manifest.json`
- `stage2_btc_extension/stage2_result_save/metadata/outputs/stage2/metrics/stage2_i60_ohlc_ma_vb_r20/seed_42/train_metadata.json`
- `stage2_btc_extension/stage2_result_save/metadata/outputs/stage2/metrics/stage2_i60_ohlc_ma_vb_r20/seed_42/test_metrics.json`

Stage 2 code dependency:
- `stage2_btc_extension/src/stage2_btc/data/ohlcv.py`
- `stage2_btc_extension/src/stage2_btc/data/label_split.py`
- `stage2_btc_extension/src/stage2_btc/imaging/ohlcv_image.py`
- `stage2_btc_extension/src/stage2_btc/models/stock_cnn.py`
- `stage2_btc_extension/src/stage2_btc/evaluation/prediction.py`
- `stage2_btc_extension/src/stage2_btc/evaluation/backtest.py`
- `stage2_btc_extension/src/stage2_btc/interpretability/gradcam.py`

Stage 3 확인 파일:
- `stage3_linear_adapter/README.md`
- `stage3_linear_adapter/docs/linear_adapter_design.md`
- `stage3_linear_adapter/src/stage3_linear/models/linear_stock_cnn.py`

## Stage 2 baseline 고정

Stage 4 primary baseline은 다음으로 고정합니다.

```text
image_window = 60
return_horizon = 20
image_spec = ohlc_ma_vb
experiment = stage2_i60_ohlc_ma_vb_r20
```

이유:
- 확인한 `I20/I60 x R20 x 4 image spec` selected five-seed grid에서 가장 강한
  Stage 2 configuration입니다.
- seed 42 단일 결과만 보고 고르는 것보다 더 방어 가능합니다.

Five-seed summary:

| Metric | Value |
| --- | ---: |
| Accuracy mean | `0.579320` |
| Accuracy std | `0.018218` |
| Accuracy minus majority mean | `0.038029` |
| ROC-AUC mean | `0.584862` |
| ROC-AUC std | `0.023250` |
| Average precision mean | `0.611256` |
| F1 mean | `0.651071` |
| Brier score mean | `0.274337` |
| Long/flat Sharpe net mean | `3.442312` |
| Long/short Sharpe net mean | `2.407759` |

선택 baseline의 seed-level sanity check:

| Seed | Accuracy | ROC-AUC | Accuracy - majority |
| ---: | ---: | ---: | ---: |
| 42 | `0.603053` | `0.616950` | `0.061763` |
| 43 | `0.574601` | `0.583347` | `0.033310` |
| 44 | `0.562804` | `0.560969` | `0.021513` |
| 45 | `0.562804` | `0.565132` | `0.021513` |
| 46 | `0.593338` | `0.597911` | `0.052047` |

해석:
- Seed 42는 강하지만, seed 42만 근거인 것은 아닙니다.
- five-seed 평균도 majority-class accuracy보다 높습니다.
- Stage 4는 가능하면 Stage 2 five-seed summary와 비교해야 합니다.

## Stage 4에 필요한 Stage 2 single-seed metadata

`stage2_i60_ohlc_ma_vb_r20` seed 42 run manifest 기준:

| Item | Value |
| --- | --- |
| Source file | `/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv` |
| Device | `cuda` |
| Model params | `2,952,962` |
| Train rows | `671` |
| Validation rows | `287` |
| Test rows | `1,441` |
| Test date range | `2021-01-01` to `2024-12-11` |
| Test positive rate / majority accuracy | `0.541291` |
| Train pixel mean | `0.0760317485` |
| Train pixel std | `0.2650489044` |
| Batch size | `128` |
| Learning rate | `1e-5` |
| Early stopping | `val_loss`, patience `2` |
| Best epoch | `10` |
| Stopped epoch | `12` |

Stage 4는 Stage 2 data rule을 고정해야 합니다.
- 같은 BTC source file;
- 같은 `I60/R20/ohlc_ma_vb` image construction;
- 같은 label definition, split, train-only image normalization;
- 같은 classification/trading metric function;
- 같은 Grad-CAM convention.

## Stage 2 code reuse 결정

Stage 4에서 BTC pipeline을 다시 작성하거나 fork하지 않습니다. Stage 2 module을
재사용하거나 mirror합니다.

고정해서 가져올 것:
- OHLCV parser와 canonical column handling.
- BTC image generator.
- Label/split builder.
- `StockCNN` I60 variant.
- Prediction metric과 trading metric code.
- Grad-CAM hook policy.

Stage 4에서 추가할 것:
- context feature builder;
- train-only context scaler;
- context MLP encoder;
- concat/gating/FiLM model wrapper;
- context/gate/gamma/beta logging.

## Stage 2 model dependency

`I60`에서 Stage 2는 다음을 사용합니다.

```text
input_shape = (1, 96, 180)
channels = (64, 128, 256, 512)
expected_flatten_dim = 184,320
expected_num_params = 2,952,962
```

CNN block은 다음입니다.

```text
Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d
```

Stage 4 FiLM은 계획대로 block을 다음처럼 바꿉니다.

```text
Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d
```

즉 Stage 4 modulation은 post-flatten adapter가 아닙니다. Block-level feature
modulation입니다.

## Stage 3 dependency review

Stage 3 Linear는 비교 결과이지 Stage 4 code dependency가 아닙니다.

Stage 3 model insertion:

```text
image
  -> fixed Stage 2 Stock_CNN convolution blocks
  -> flatten
  -> Linear(flatten_dim, adapter_dim=128, bias=False)
  -> Linear(adapter_dim, 2, bias=False)
  -> logits
```

Stage 2 best single-seed configuration에 대한 preliminary Stage 3 결과:

| Model | Config | Seed | Accuracy | Majority accuracy | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage 2 baseline | `I60/R20/ohlc_ma_vb` | 42 | `0.603053` | `0.541291` | `0.616950` |
| Stage 3 Linear | `I60/R20/ohlc_ma_vb`, adapter dim `128` | 42 | `0.541291` | `0.541291` | `0.522101` |

해석:
- 첫 Linear test는 majority-class-level accuracy까지 하락했습니다.
- 이는 단순히 post-flatten dense parameter block을 추가하는 것만으로는 충분하지
  않다는 Stage 4 논리를 뒷받침합니다.
- 다만 이것은 아직 preliminary Stage 3 결과입니다. Stage 3 full grid는 완료되지
  않았습니다.
- Stage 4는 Stage 3를 negative/simple-parameter ablation으로만 참조하고, Stage 4
  model component로 재사용하지 않습니다.

## Artifact availability

로컬에 있는 것:
- Stage 2 selected five-seed tables.
- `I60/R20/ohlc_ma_vb` seed 42 Stage 2 metadata/metrics.
- Stage 2 selected Grad-CAM preview:
  `stage2_btc_extension/reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam.png`
- Stage 2 selected Grad-CAM sample metadata:
  `stage2_btc_extension/reports/figures/gradcam/stage2_single_seed_best_i60_ohlc_ma_vb_r20_gradcam_samples.csv`
- Stage 3 Linear code와 preliminary result documentation.

GitHub에 필요 없는 것:
- `.pt` checkpoints;
- large prediction CSVs;
- raw output zips.

Stage 4 Kaggle 실행에서는 runner가 checkpoint를 다시 생성하면 됩니다. Stage 4가
GitHub 안의 local checkpoint file을 요구하면 안 됩니다.

## 4-4 결정

다음 의존성을 고정하고 4-5로 넘어갑니다.

```text
Stage 4 primary baseline:
    Stage 2 I60/R20/ohlc_ma_vb StockCNN

Primary comparison target:
    Stage 2 selected five-seed mean
    accuracy = 0.579320
    ROC-AUC = 0.584862

Stage 3 role:
    negative/simple-parameter ablation
    not a Stage 4 architecture dependency

Stage 4 model changes:
    add context branch and concat/gating/FiLM modulation
    keep Stage 2 data/image/label/split/evaluation fixed
```
