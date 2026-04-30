# Stage 1 Source Map

## English

Status:
- Initial source map created.
- Stage 1-1 source and constraint re-check completed on 2026-04-30.
- Stage 1-2 data loading detail plan completed on 2026-04-30.
- Stage 1-3 label construction detail plan completed on 2026-04-30.
- Stage 1-4 split and normalization detail plan completed on 2026-04-30.
- Stage 1-5 baseline CNN implementation detail plan completed on 2026-04-30.
- Stage 1-6 training loop detail plan completed on 2026-04-30.
- Stage 1 implementation/execution gates added to `docs/stage1_checklist.md` on 2026-04-30.
- Stage 1-6K Kaggle runner and environment config detail plan completed on 2026-04-30.
- Stage 1-7 evaluation and prediction-output detail plan completed on 2026-04-30.
- Stage 1-8 Grad-CAM detail plan completed on 2026-04-30.
- Stage 1-9 report plan completed on 2026-04-30.
- Stage 1-I0 implementation readiness review completed on 2026-04-30.
- Stage 1-I1 shared code/config scaffold implementation completed on 2026-04-30.
- Stage 1-I2 data loading implementation completed on 2026-04-30.
- Stage 1-I3 label, split, and normalization implementation completed on 2026-04-30.
- Stage 1-I4 baseline CNN model implementation completed on 2026-04-30.
- Stage 1-I5 training loop and checkpoint implementation completed on 2026-04-30.
- Stage 1-I6 Kaggle/local runner implementation completed on 2026-05-01.

Stage 1 gate structure:
- Planning/design gates: `1-0` through `1-9`.
- Implementation/execution gates: `1-I0` through `1-I12`.
- Stage 1 reproduction is not complete until implementation, smoke testing,
  Kaggle full runs, prediction outputs, metrics, and Grad-CAM figures exist.

## Global Process Source

| Source | Path / URL | Stage 1 use |
| --- | --- | --- |
| Root plan | `../PLAN.md` | Defines stage order, GitHub-first model rule, Grad-CAM rule, and one-step-at-a-time workflow. |
| Stage 0 data audit | `../stage0_data_check/docs/monthly20_data_check.md` | Defines feasible data scope: public I20 full-spec images and `I20/R5`, `I20/R20`, `I20/R60`. |
| Stage 0 source audit | `../stage0_data_check/docs/source_reference_check.md` | Defines Re-image paper references, GitHub commit, and paper/GitHub mismatches. |

## Paper Sources

| Topic | Source | Stage 1 use |
| --- | --- | --- |
| Re-image pipeline | `../자료조사/Re-image 요약.md` | Image construction, CNN architecture, training setup, split, interpretation notes. |
| Re-image original PDF | `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | Visual confirmation of Figure 7, Figure 13, and paper details before implementation. |
| Grad-CAM method | `../자료조사/Grad-CAM요약.md` | Grad-CAM equations, logit target, activation/gradient heatmap interpretation. |
| Grad-CAM original PDF | `../자료조사/Grad-CAM.pdf` | Visual/method confirmation before Grad-CAM implementation. |

## GitHub Sources

| Topic | Source | Stage 1 use |
| --- | --- | --- |
| Stock CNN reference repo | `https://github.com/lich99/Stock_CNN` | Stage 1 I20 model core reference. |
| Checked commit | `415e2acf2a5013afca67e383acd3edc61fced840` | Commit to cite in model code comments. |
| Checked model file | `models/baseline.py` | I20 CNN layer structure and forward behavior. |
| Local copy | `../stage0_data_check/references/stock_cnn_baseline_415e2ac.py.txt` | Offline reference copy used before implementation. |

## Stage 1 Decisions Carried from Stage 0

| Decision | Current value | Reason |
| --- | --- | --- |
| Direct stock experiments | `I20/R5`, `I20/R20`, `I20/R60` | Current local data is 20-day only but has 5/20/60 future-return labels. |
| Image spec | Full `OHLC + MA + Volume` | Current `.dat` files are already rendered with `20_ma` and `has_vb`. |
| Tensor convention | `(batch, channel, height=64, width=60)` | Local `.dat` files reshape to `(N, 64, 60)`. |
| Label rule | future return `> 0` -> `1`; otherwise `0` | Re-image binary classification setup. |
| Loss interface | `CrossEntropyLoss` on logits | GitHub model returns logits and comments out softmax. |
| I20 model core | Follow `Stock_CNN/models/baseline.py` unless user decides otherwise | User instructed to follow GitHub core implementation. |
| Grad-CAM status | Required after training | Re-image Figure 13 style output is part of Stage 1 interpretation. |

## Known Mismatches to Keep Visible

| Mismatch | Current handling |
| --- | --- |
| Paper summary emphasizes first-layer dilation; GitHub applies I20 dilation to all three conv layers. | Follow GitHub for I20 core unless user changes the decision; document in code comments. |
| Paper discusses softmax probabilities; GitHub returns logits. | Train on logits; compute softmax only for prediction/evaluation outputs. |
| Paper/table dimension notation can mix width/height order. | Use explicit tensor convention `(batch, 1, height=64, width=60)`. |
| Current data cannot support all paper windows/spec ablations. | Report Stage 1 as public I20 full-spec reproduction, not full paper-wide reproduction. |

## Implementation File References

| Gate | File | Purpose |
| --- | --- | --- |
| 1-I1 | `src/stage1_reimage/config.py` | Load and validate environment YAML config sections. |
| 1-I1 | `src/stage1_reimage/paths.py` | Build explicit local/Kaggle path objects and output-directory helpers. |
| 1-I1 | `src/stage1_reimage/runtime.py` | Select runtime device from config. |
| 1-I1 | `src/stage1_reimage/seed.py` | Apply one selected run seed to Python, NumPy, and PyTorch. |
| 1-I1 | `scripts/check_scaffold.py` | Local smoke check for package import, config, paths, seed, and device selection. |
| 1-I2 | `src/stage1_reimage/data/monthly20.py` | Discover/validate public `monthly_20d` shards and provide memmap-backed image loading. |
| 1-I2 | `scripts/check_data_loading.py` | Local smoke check for row alignment, shard count, and sample tensor shape. |
| 1-I3 | `src/stage1_reimage/data/label_split.py` | Build horizon labels, deterministic splits, split summaries, and train-only pixel normalization metadata. |
| 1-I3 | `scripts/check_label_split_normalization.py` | Local smoke check for labels, split counts, and normalization JSON writing. |
| 1-I4 | `src/stage1_reimage/models/stock_cnn.py` | GitHub-style I20 baseline CNN with hookable convolution layers and logits output. |
| 1-I4 | `scripts/check_model.py` | Local smoke check for model shapes, parameter count, logits output, and Grad-CAM layer lookup. |
| 1-I5 | `src/stage1_reimage/training/loop.py` | Xavier initialization, CrossEntropyLoss, Adam, training/validation loop, early stopping, checkpoints, history, and metadata. |
| 1-I5 | `scripts/check_training_loop.py` | Synthetic local smoke check for forward/backward, best/last checkpoints, history CSV, and metadata JSON. |
| 1-I6 | `src/stage1_reimage/runners/stage1_baseline.py` | Config-driven local/Kaggle baseline runner, dataloaders, training matrix, and run manifest. |
| 1-I6 | `scripts/run_stage1_baseline.py` | CLI runner for local smoke and Kaggle full modes. |
| 1-I6 | `notebooks/kaggle_stage1_runner.md` | Kaggle command skeleton and expected input layout. |

1-I1 source note:
- These files implement only the shared execution scaffold required by root
  `PLAN.md`.
- No Re-image model architecture, label rule, loss, optimizer, evaluation, or
  Grad-CAM algorithm is implemented in 1-I1.

1-I2 source note:
- Data loading follows `docs/data_loading_plan.md`.
- The module only returns images and metadata. It does not construct labels,
  splits, normalization statistics, model inputs beyond the image tensor, or
  Grad-CAM outputs.

1-I3 source note:
- Label construction follows `docs/label_construction_plan.md`.
- Split and normalization follow `docs/split_normalization_plan.md`.
- The implementation still does not include model architecture, training loop,
  evaluation metrics, prediction CSV writing, or Grad-CAM.

1-I4 source note:
- Model implementation follows `lich99/Stock_CNN/models/baseline.py`, commit
  `415e2acf2a5013afca67e383acd3edc61fced840`.
- Re-image local summary maps CNN architecture/training details to pp. 12-21
  and Figure 7 to p. 18.
- The known paper/GitHub dilation mismatch remains documented in code and docs.
- The implementation still does not include loss, optimizer, training loop,
  checkpointing, evaluation metrics, prediction CSV writing, or Grad-CAM.

1-I5 source note:
- Training loop follows `docs/training_loop_plan.md`.
- Re-image local summary maps training settings to CNN/training pp. 12-21.
- The implementation uses the reported choices where available:
  CrossEntropyLoss, Adam, learning rate `1e-5`, batch size `128`, Xavier
  initialization, and validation-loss early stopping patience `2`.
- Adam betas/eps, weight decay, exact seeds, and epoch cap remain explicit
  implementation choices because the paper summary does not report them.
- The implementation still does not include final evaluation metrics,
  prediction CSV writing, portfolio outputs, or Grad-CAM.

1-I6 source note:
- Runner implementation follows `docs/kaggle_runner_plan.md`.
- One runner script serves local and Kaggle execution through config.
- Local non-smoke runs are blocked by default.
- The implementation still does not include final evaluation metrics,
  prediction CSV writing, portfolio outputs, or Grad-CAM.

## 1-1 Source and Constraint Re-check

Checked on:
- 2026-04-30

Checked required sources:

| Source | Check result |
| --- | --- |
| `../PLAN.md` | Confirms one-step workflow, Stage 1 before BTC, GitHub-first model core, and Grad-CAM as required output. |
| `../stage0_data_check/docs/monthly20_data_check.md` | Confirms current direct data scope remains public `monthly_20d` full-spec images only. |
| `../stage0_data_check/docs/source_reference_check.md` | Confirms `Stock_CNN` commit and I20 paper/GitHub mismatch log. |
| `../자료조사/Re-image 요약.md` | Confirms image construction, 20-day image size, CNN/training setup, split, and interpretation requirements. |
| `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | File exists locally; exact page/figure visual checks should still be done when writing code comments tied to a page/figure. |
| `../자료조사/Grad-CAM요약.md` | Confirms Grad-CAM uses target class score/logit, activation gradients, spatial-average channel weights, ReLU heatmap, and bilinear upsampling. |
| `../자료조사/Grad-CAM.pdf` | File exists locally; method details are tracked through the local summary before Grad-CAM coding. |

Confirmed Stage 1 constraints:

| Constraint | 1-1 result |
| --- | --- |
| Direct experiments | Still only `I20/R5`, `I20/R20`, and `I20/R60`. |
| Image source | Still author-provided rendered `.dat` images from `monthly_20d`. |
| Image spec | Still full `OHLC + MA + Volume`; A/B/C/D ablation cannot be recovered from the current `.dat`. |
| Tensor convention | Still `(batch, 1, height=64, width=60)`. |
| Labels | Still binary labels from selected future return `> 0`, with NaN filtering per horizon. |
| Model core | Still follow `lich99/Stock_CNN/models/baseline.py` for I20 unless the user changes the decision. |
| Dilation mismatch | Still visible: paper summary emphasizes first-layer dilation; GitHub uses I20 dilation in all three conv layers. |
| Loss/softmax mismatch | Still visible: paper discusses softmax probability; GitHub returns logits. |
| Grad-CAM | Still required after training; it is class-discriminative heatmap, not raw feature map. |

1-1 conclusion:
- Stage 1 remains a public-data reproduction of the author-provided I20 full-spec pipeline.
- No implementation should start until the next checklist item, `1-2. Data loading detail plan`, is planned and confirmed.

## 1-2 Data Loading Detail Plan

Checked on:
- 2026-04-30

Output:
- `docs/data_loading_plan.md`

Data loading decisions:

| Topic | Decision |
| --- | --- |
| Data root | `../테스트/Test/img_data/monthly_20d` |
| File discovery | Deterministic year pairs from 1993 through 2019. |
| Image pattern | `20d_month_has_vb_20_ma_{year}_images.dat` |
| Label pattern | `20d_month_has_vb_20_ma_{year}_labels_w_delay.feather` |
| Image dtype | Read `.dat` as `uint8`. |
| Pixel scale | Convert to `float32` and divide by `255.0` for model tensors. |
| Image shape | Per sample `(1, 64, 60)`, batch `(batch_size, 1, 64, 60)`. |
| Label reader | `pandas.read_feather`; local `pyarrow` is available. |
| Row alignment | Same year and same local row index define one sample. |
| Memory strategy | Use memory-mapped image loading by default because image files total about `7.9G`. |
| Shuffling | No base-data shuffling before horizon filtering, split assignment, and normalization decisions. |
| Deferred decisions | Horizon filtering to 1-3; split/normalization to 1-4; DataLoader performance settings to 1-6. |

1-2 conclusion:
- Data loading should be implemented as shard-aware lazy image loading with strict per-year row-alignment validation.
- No data-loading code has been written yet.

## 1-3 Label Construction Detail Plan

Checked on:
- 2026-04-30

Output:
- `docs/label_construction_plan.md`

Label decisions:

| Topic | Decision |
| --- | --- |
| Main targets | `Ret_5d`, `Ret_20d`, `Ret_60d`. |
| Experiments | `stage1_i20_r5`, `stage1_i20_r20`, `stage1_i20_r60`. |
| Label rule | `label = 1` if selected future return `> 0`; otherwise `0`. |
| Zero returns | Treated as class `0`, because Up requires strictly positive return. |
| NaN handling | Drop rows with missing selected target return separately for each horizon. |
| Model input | Image tensor only; return columns and metadata are not input features. |
| Required metadata | Preserve `Date`, `StockID`, `MarketCap`, return columns, `EWMA_vol`, `year`, `local_row`, `target_return_name`, `target_return`, and `label`. |
| `Ret_month` | Preserve as metadata or optional future experiment; not a fixed Stage 1 target. |
| Leakage rule | Future returns are labels/evaluation targets only, never model inputs. |

Valid row counts:

| Target | Valid rows | Positive rate among valid rows |
| --- | ---: | ---: |
| `Ret_5d` | 2,190,724 | 50.48% |
| `Ret_20d` | 2,180,610 | 51.23% |
| `Ret_60d` | 2,151,213 | 53.17% |

1-3 conclusion:
- Stage 1 will create three horizon-specific filtered datasets from the same I20 image rows.
- Split and normalization remain deferred to `1-4`.

## 1-4 Split and Normalization Detail Plan

Checked on:
- 2026-04-30

Outputs:
- `docs/split_normalization_plan.md`
- `data_inventory/stage1_horizon_counts_by_year.csv`
- `data_inventory/stage1_horizon_counts_by_period.csv`

Split decisions:

| Topic | Decision |
| --- | --- |
| Train/validation pool | 1993-2000. |
| Test period | 2001-2019. |
| Train/validation split | Deterministic 70/30 random split after horizon-specific NaN filtering. |
| Split seed | `42`, because the paper summary does not report a seed. |
| Split unit | Sample row, identified by `(year, local_row)`. |
| Stratification | `false` by default, because the paper summary only says random 70/30 split. |
| Test leakage | Test rows are never used for training, early stopping, or normalization-stat fitting. |

Normalization decisions:

| Topic | Decision |
| --- | --- |
| Pixel scaling | `uint8 / 255.0` to `[0, 1]`. |
| Standardization | Fit scalar pixel mean/std on training subset only. |
| Application | Use train mean/std for train, validation, and test. |
| Scope | Per horizon, because filtered training rows differ across `Ret_5d`, `Ret_20d`, and `Ret_60d`. |
| Saved metadata | Save `normalization.json` and `split_summary.json` per horizon. |

1-4 conclusion:
- Stage 1 follows the paper summary split: 1993-2000 train/validation and 2001-2019 test.
- The seed is an explicit reproduction choice because the paper does not provide one.
- Model implementation can now be planned in `1-5`, but no model code has been written yet.

## 1-5 Baseline CNN Implementation Detail Plan

Checked on:
- 2026-04-30

Output:
- `docs/baseline_cnn_implementation_plan.md`

Reference implementation:

| Item | Decision |
| --- | --- |
| Repository | `https://github.com/lich99/Stock_CNN` |
| Commit | `415e2acf2a5013afca67e383acd3edc61fced840` |
| File | `models/baseline.py` |
| Local copy | `../stage0_data_check/references/stock_cnn_baseline_415e2ac.py.txt` |

Model decisions:

| Topic | Decision |
| --- | --- |
| Planned file | `src/models/stock_cnn.py` |
| Planned class | `StockCNNI20` |
| Direct scope | Public `I20/R5`, `I20/R20`, `I20/R60` reproduction. |
| Input convention | `(batch_size, 1, height=64, width=60)`. |
| GitHub compatibility | Keep `layer1`, `layer2`, `layer3`, `fc1`, and logits-returning forward behavior. |
| Softmax | Do not apply inside `forward`; apply only in evaluation/prediction output. |
| Parameter count target | `708,866`. |

I20 architecture fixed for implementation:

| Block | Layer values |
| --- | --- |
| `layer1` | `Conv2d(1,64,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`, BN, LeakyReLU `0.01`, MaxPool `(2,1)` |
| `layer2` | `Conv2d(64,128,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`, BN, LeakyReLU `0.01`, MaxPool `(2,1)` |
| `layer3` | `Conv2d(128,256,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`, BN, LeakyReLU `0.01`, MaxPool `(2,1)` |
| `fc1` | Dropout `0.5`, Linear `46080 -> 2` |

Shape contract:

| Step | Shape |
| --- | --- |
| Input | `(batch_size, 1, 64, 60)` |
| After `layer1` | `(batch_size, 64, 13, 60)` |
| After `layer2` | `(batch_size, 128, 5, 60)` |
| After `layer3` | `(batch_size, 256, 3, 60)` |
| Flatten | `(batch_size, 46080)` |
| Logits | `(batch_size, 2)` |

Grad-CAM readiness:
- Model `forward` will not compute Grad-CAM.
- Keep convolution modules hookable for later `1-8`.
- Planned Grad-CAM target names:
  - `layer1_conv`: `model.layer1[0]`
  - `layer2_conv`: `model.layer2[0]`
  - `layer3_conv`: `model.layer3[0]`

1-5 conclusion:
- Stage 1 baseline model implementation should now be written exactly around the GitHub I20 core.
- The known paper/GitHub dilation mismatch remains documented.
- Training behavior is still deferred to `1-6`; no model code has been written yet.

## 1-6 Training Loop Detail Plan

Checked on:
- 2026-04-30

Output:
- `docs/training_loop_plan.md`

Paper/source training settings:

| Topic | Decision |
| --- | --- |
| Loss | `CrossEntropyLoss` on logits. |
| Optimizer | Adam. |
| Learning rate | `1e-5`. |
| Batch size | `128`. |
| Initialization | Xavier; implementation variant fixed as `xavier_uniform_`. |
| Dropout | `0.5`, already fixed in the model classifier. |
| Early stopping | Validation loss, patience `2`, `min_delta=0.0`. |
| Train/test split | Use `1-4`: 1993-2000 train/validation, 2001-2019 test. |
| Normalization | Use `1-4`: fit train-only pixel mean/std per horizon. |
| Independent runs | 5 per horizon, then average test `prob_up`. |

Reproduction choices for paper-unreported details:

| Topic | Decision |
| --- | --- |
| Split seed | `42`, already fixed in `1-4`. |
| Run seeds | `[42, 43, 44, 45, 46]`. |
| Epoch cap | `100`, safety cap because paper does not report exact epochs. |
| Weight decay | `0.0`, because paper does not report weight decay. |
| LR schedule | None by default, because paper does not report one. |
| Mixed precision | Disabled by default. |
| Gradient clipping | Disabled by default. |
| Device | `auto`; full run target is Kaggle CUDA GPU. |
| Determinism | Seed Python, NumPy, PyTorch CPU/CUDA; use deterministic cuDNN defaults when possible. |

Checkpoint/logging decisions:

| Topic | Decision |
| --- | --- |
| Best checkpoint | `outputs/checkpoints/{experiment_name}/seed_{run_seed}/best.pt`. |
| Last checkpoint | `outputs/checkpoints/{experiment_name}/seed_{run_seed}/last.pt`. |
| Best criterion | Lowest validation loss. |
| Test checkpoint | Use `best.pt`. |
| Train history | `outputs/metrics/{experiment_name}/seed_{run_seed}/train_history.csv`. |
| Train metadata | `outputs/metrics/{experiment_name}/seed_{run_seed}/train_metadata.json`. |

1-6 conclusion:
- Stage 1 full reproduction should train 5 independent `StockCNNI20` models per horizon and average probabilities.
- Local smoke tests should use one seed and tiny subsets only, and must not be reported as reproduction results.
- Kaggle path/performance details remain deferred to `1-6K`; no training code has been written yet.

## 1-6K Kaggle Runner and Environment Config Detail Plan

Checked on:
- 2026-04-30

Outputs:
- `docs/kaggle_runner_plan.md`
- `configs/env_kaggle.yaml`
- `configs/env_local.yaml`
- `notebooks/README.md`

Environment decisions:

| Topic | Decision |
| --- | --- |
| Full run environment | Kaggle Notebook. |
| Codebase rule | One shared `src/`; no separate Kaggle-only/local-only codebase. |
| Kaggle data input | Upload/attach `monthly_20d` as a Kaggle Dataset. |
| Recommended dataset name | `reimage-monthly-20d`. |
| Kaggle data root | `/kaggle/input/reimage-monthly-20d/monthly_20d`. |
| Kaggle output root | `/kaggle/working/stage1_reimage_reproduction/outputs`. |
| Local role | Smoke tests only by default. |
| Local data root | `/Users/jaehyeonjeong/Desktop/논문/테스트/Test/img_data/monthly_20d`. |

Kaggle runtime defaults:

| Topic | Decision |
| --- | --- |
| Device | `cuda` for full run. |
| CUDA unavailable behavior | Fail clearly for full run. |
| DataLoader workers | `num_workers: 2` conservative default. |
| Pin memory | `true` on Kaggle/CUDA. |
| Mixed precision | `false`, because the paper does not report AMP. |
| Internet | Not required in preferred code snapshot mode. |

Run modes:

| Mode | Seeds | Use |
| --- | --- | --- |
| `smoke` | `[42]` | Shape/path/checkpoint smoke test only. |
| `full_single_seed` | `[42]` | First full baseline check. |
| `full_paper_style` | `[42, 43, 44, 45, 46]` | Paper-style 5-run averaged reproduction. |

Manifest requirement:
- Every Kaggle run must save `outputs/run_manifests/run_manifest.json`.
- The manifest must record dataset/code versions, config files, seeds, package versions, CUDA/GPU info, and source references.

1-6K conclusion:
- Stage 1 can now be implemented against environment configs instead of hardcoded local/Kaggle paths.
- Actual runner notebook and training commands remain deferred until implementation begins.
- Evaluation schema is still deferred to `1-7`; Grad-CAM execution is still deferred to `1-8`.

## 1-7 Evaluation and Prediction-output Detail Plan

Checked on:
- 2026-04-30

Output:
- `docs/evaluation_prediction_plan.md`

Prediction decisions:

| Topic | Decision |
| --- | --- |
| Probability | `softmax(logits, dim=1)`; paper source: Re-image summary maps softmax Up probability to CNN/training pp.20-22. |
| Up probability | `prob_up`, class `1`. |
| Prediction threshold | 50% threshold; paper source: Re-image summary maps threshold to CNN/training pp.20-22. Implementation uses `prob_up >= 0.5`. |
| Tie rule | exact `0.5` goes to class `1`; paper does not report tie behavior, so this is an implementation choice. |
| 5-run averaging | average probabilities, not logits. |
| Per-seed file | `outputs/predictions/{experiment_name}/seed_{run_seed}/test_predictions.csv`. |
| Averaged file | `outputs/predictions/{experiment_name}/averaged/test_predictions.csv`. |

Metric decisions:

| Topic | Decision |
| --- | --- |
| Core classification | accuracy, precision, recall, F1, ROC AUC, average precision, Brier, log loss, confusion matrix. |
| Class balance | positive rate, negative rate, predicted positive rate. |
| Baseline comparison | majority-class accuracy and accuracy minus majority-class accuracy. |
| Return diagnostics | global and date-wise Pearson/Spearman correlation between prediction signal and target return. |
| Paper-style output | decile ranking and H-L portfolio metrics are planned for stock cross-sectional Stage 1 evaluation. |

1-7 conclusion:
- Prediction files must preserve `Date`, `StockID`, `MarketCap`, target returns, labels, logits/probabilities, and correctness.
- Stage 1 evaluation includes both classification metrics and stock cross-sectional ranking outputs.
- Exact implementation remains deferred to `1-I7`; Grad-CAM sample selection remains deferred to `1-8`.

## 1-I7 Evaluation and Prediction-output Implementation

Implemented on:
- 2026-05-01

Output:
- `docs/evaluation_prediction_implementation.md`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/evaluate_stage1_predictions.py`

Implementation source mapping:

| Topic | Source | Code location |
| --- | --- | --- |
| Model output interface | `lich99/Stock_CNN/models/baseline.py`, commit `415e2acf2a5013afca67e383acd3edc61fced840`; local model implementation in `src/stage1_reimage/models/stock_cnn.py` | `predict_loader()` keeps `model(images)` as logits. |
| Softmax probability | Re-image summary maps CNN/training probability interpretation to pp.20-22; `docs/evaluation_prediction_plan.md` | `predict_loader()` applies `torch.softmax(logits, dim=1)` only during evaluation. |
| Threshold | `docs/evaluation_prediction_plan.md`; 50% threshold from local Re-image summary pp.20-22 mapping | `prob_up >= 0.5` when `tie_break_class: 1`. |
| Tie rule | Implementation convention because the paper does not separately report exact `0.5` behavior | `evaluation.threshold` and `evaluation.tie_break_class` in config. |
| Averaging | `docs/evaluation_prediction_plan.md` | `average_seed_predictions()` averages softmax probabilities, not logits. |
| Prediction metadata | Stage 1 data-loading and label plans | Prediction CSV preserves `Date`, `StockID`, `MarketCap`, target returns, labels, logits/probabilities, and correctness. |
| Metrics | `docs/evaluation_prediction_plan.md` | `compute_classification_metrics()` and `compute_correlation_metrics()`. |

Validation:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --split validation --average-seed-predictions 42`

1-I7 conclusion:
- Seed-level and averaged prediction export is now implemented.
- Classification and prediction-return correlation metrics are now implemented.
- Smoke outputs under `outputs/` are not reproduction results and are excluded
  from GitHub.
- Portfolio/decile H-L metrics remain deferred until the final report
  convention is rechecked.
- Grad-CAM sample selection can now consume prediction CSVs in `1-I8`.

## 1-I7R Code Annotation/Readability Pass

Updated on:
- 2026-05-01

Output:
- `checklist_results/1-I7R_code_annotation.md`

Source/policy mapping:

| Topic | Source | Stage 1 code action |
| --- | --- | --- |
| Code readability | Root `PLAN.md`, code-writing principles | Added a permanent rule that all code should include detailed explanatory comments. |
| Korean code explanations | Root `PLAN.md`, user review request | Explanatory comments/docstrings are now written primarily in Korean. |
| Tensor shape comments | Root `PLAN.md`, user review request | Added comments/docstrings for image tensors `(1,64,60)`, DataLoader batches `(B,1,64,60)`, CNN feature maps, logits `(B,2)`, and probability outputs. |
| Leakage comments | Root `PLAN.md`, Stage 1 leakage rule | Added comments where metadata/returns are preserved but not fed into the CNN. |
| Data movement comments | User review request | Added comments describing how values move from config to loader, loader to dataset, dataset to model, model to loss, and model to prediction CSV. |

1-I7R conclusion:
- The code behavior is unchanged.
- The code now carries more reader-facing explanations for how each stage works.
- Reader-facing comments/docstrings are now primarily Korean.

## 1-8 Grad-CAM Detail Plan

Checked on:
- 2026-04-30

Output:
- `docs/gradcam_plan.md`

Source mapping:

| Topic | Source | Stage 1 decision |
| --- | --- | --- |
| Figure layout | Re-image Figure 13; local Re-image summary maps interpretation material to pp.41-49; user-provided Figure 13 screenshot | Reproduce original image row plus CNN-layer Grad-CAM rows for 2019 `Up` and `Down` predictions. |
| Grad-CAM target score | Grad-CAM original pp.4-6 as summarized in `../자료조사/Grad-CAM요약.md` | Use pre-softmax target-class logit, not softmax probability. |
| Grad-CAM formula | Grad-CAM original pp.4-6 as summarized in `../자료조사/Grad-CAM요약.md` | Use activation gradients, spatial-average channel weights, `ReLU(sum alpha A)`, and bilinear upsampling. |
| Layer choice caution | Grad-CAM original appendix/pp.12-21 as summarized in `../자료조사/Grad-CAM요약.md` | Report that layer choice affects class-discriminative behavior; show all three I20 conv layers. |
| Target layers | `docs/baseline_cnn_implementation_plan.md`; `lich99/Stock_CNN`-style Stage 1 model | Hook `model.layer1[0]`, `model.layer2[0]`, and `model.layer3[0]`. |
| Prediction source | `docs/evaluation_prediction_plan.md` | Select samples from per-seed or averaged prediction CSVs after training. |

Grad-CAM decisions:

| Topic | Decision |
| --- | --- |
| Primary experiment | `stage1_i20_r20`. |
| Extended experiments | `stage1_i20_r5`, `stage1_i20_r20`, `stage1_i20_r60`. |
| Primary year | `2019`. |
| Samples | 10 `Up` predictions and 10 `Down` predictions per figure. |
| Target class | Predicted class by default. |
| Heatmap normalization | Per-heatmap min-max normalization; implementation choice because the Re-image figure does not specify exact normalization. |
| Colormap | Blue/cyan-style heatmap; implementation choice pending visual comparison. |
| Full paper-style ensemble heatmap | Select samples from averaged predictions; compute seed-level Grad-CAM and average normalized heatmaps across seed checkpoints. |
| Correctness filter | Do not require correct predictions by default; the figure explains received classifications. |

1-8 conclusion:
- Stage 1 Grad-CAM is defined as plain predicted-class Grad-CAM, not raw feature
  maps and not Guided Grad-CAM.
- Code implementation remains deferred to `1-I8`.
- Before adding final source comments in code, visually recheck the PDF pages
  because local automatic PDF text extraction is unavailable.

## 1-9 Stage 1 Report Plan

Checked on:
- 2026-04-30

Output:
- `docs/stage1_report_plan.md`

Report source mapping:

| Topic | Source | Stage 1 report use |
| --- | --- | --- |
| Report scope | `../PLAN.md`; Stage 0 data audit | Report Stage 1 as public I20 full-spec reproduction, not full paper-wide reproduction. |
| Data limitations | `../stage0_data_check/docs/monthly20_data_check.md`; `docs/label_construction_plan.md` | State that available rendered images support `I20/R5`, `I20/R20`, and `I20/R60` only. |
| Split and normalization | `docs/split_normalization_plan.md`; Re-image summary CNN/training pp.12-21 | Report 1993-2000 train/validation, 2001-2019 test, 70/30 split, train-only pixel standardization. |
| Model structure | `docs/baseline_cnn_implementation_plan.md`; `Stock_CNN` commit `415e2acf...` | Report GitHub-style I20 architecture and the dilation mismatch. |
| Training reproducibility | `docs/training_loop_plan.md`; `docs/kaggle_runner_plan.md` | Report Kaggle `full_paper_style` as primary reproduction and local smoke tests as non-reproduction. |
| Classification and correlation | `docs/evaluation_prediction_plan.md`; Re-image Table 2 in U.S. experiments pp.21-33 | Prepare classification, majority-baseline, and correlation tables. |
| Portfolio outputs | `docs/evaluation_prediction_plan.md`; Re-image U.S. portfolio results pp.21-33 | Prepare stock cross-sectional decile/H-L tables when convention is matched. |
| Grad-CAM | `docs/gradcam_plan.md`; Re-image Figure 13; Grad-CAM pp.4-6 | Include Figure 13-style Grad-CAM and interpret it as a class-discriminative heatmap. |

Required report file:
- `reports/stage1_reproduction_report.md`

Required report tables:
- `reports/tables/stage1_dataset_summary.csv`
- `reports/tables/stage1_split_summary.csv`
- `reports/tables/stage1_training_summary.csv`
- `reports/tables/stage1_classification_metrics.csv`
- `reports/tables/stage1_majority_baseline_comparison.csv`
- `reports/tables/stage1_correlation_metrics.csv`
- `reports/tables/stage1_portfolio_metrics.csv`
- `reports/tables/stage1_paper_comparison.csv`
- `reports/tables/stage1_gradcam_samples.csv`
- `reports/tables/stage1_artifact_manifest.csv`

1-9 conclusion:
- The report must compare only like-for-like paper cells.
- `I20/R20` and `I20/R60` classification paper values are available from the
  local summary; `I20/R5` classification must be checked in the PDF before
  reporting a paper value.
- H-L portfolio comparisons are allowed only after decile, value-weighting, and
  annualization conventions are checked.
- Actual report generation remains deferred to `1-I12`.

## 1-I0 Implementation Readiness Review

Checked on:
- 2026-04-30

Output:
- `docs/implementation_readiness_review.md`

Readiness result:
- Ready to start `1-I1. Shared code/config scaffold implementation`.
- Readiness applies to the public `monthly_20d` I20 full-spec reproduction path:
  `I20/R5`, `I20/R20`, and `I20/R60`.

Confirmed:

| Item | Result |
| --- | --- |
| Planning gates | `1-0` through `1-9` complete. |
| Local data | 27 image `.dat` files and 27 label `.feather` files found. |
| Local smoke-test dependencies | `torch`, `numpy`, `pandas`, `pyarrow`, `sklearn`, `matplotlib`, and `yaml` importable. |
| Configs | `configs/env_local.yaml` and `configs/env_kaggle.yaml` exist. |
| Code state | `src/`, `scripts/`, and `notebooks/` contain README placeholders only. |

Readiness caveats:
- Root `PLAN.md` includes raw OHLC image generator/MA/volume items, while the
  immediate implementation uses already-rendered public I20 `.dat` files. This
  is acceptable for the first public-data reproduction pass, but raw image
  generator work must be added as a separate gate if required before claiming a
  paper-wide pipeline.
- The workspace is not currently a Git repository. Use a code snapshot or Git
  commit before Kaggle full runs to satisfy manifest provenance.
- Kaggle full runs require uploading/attaching `monthly_20d` as the
  `reimage-monthly-20d` dataset or updating the Kaggle config path.

1-I0 conclusion:
- No blocker for `1-I1`.
- Do not proceed beyond `1-I1` until that implementation gate is completed and
  reported.

## 한국어

상태:
- 초기 source map을 만들었습니다.
- 2026-04-30에 1-1 근거와 제한사항 재확인을 완료했습니다.
- 2026-04-30에 1-2 data loading 세부계획을 완료했습니다.
- 2026-04-30에 1-3 label construction 세부계획을 완료했습니다.
- 2026-04-30에 1-4 split and normalization 세부계획을 완료했습니다.
- 2026-04-30에 1-5 baseline CNN 구현 세부계획을 완료했습니다.
- 2026-04-30에 1-6 training loop 세부계획을 완료했습니다.
- 2026-04-30에 `docs/stage1_checklist.md`에 1단계 implementation/execution gate를 추가했습니다.
- 2026-04-30에 1-6K Kaggle runner와 environment config 세부계획을 완료했습니다.
- 2026-04-30에 1-7 evaluation과 prediction-output 세부계획을 완료했습니다.
- 2026-04-30에 1-8 Grad-CAM 세부계획을 완료했습니다.
- 2026-04-30에 1-9 report plan을 완료했습니다.
- 2026-04-30에 1-I0 implementation readiness review를 완료했습니다.
- 구현 파일 reference는 실제 코드를 작성한 뒤에만 추가합니다.

1단계 gate 구조:
- Planning/design gate: `1-0`부터 `1-9`.
- Implementation/execution gate: `1-I0`부터 `1-I12`.
- implementation, smoke test, Kaggle full run, prediction output, metrics,
  Grad-CAM figure가 생기기 전에는 1단계 재현 완료로 보지 않습니다.

## 전체 진행 기준 자료

| 자료 | 경로 / URL | 1단계에서 쓰는 방식 |
| --- | --- | --- |
| 루트 계획 | `../PLAN.md` | 단계 순서, GitHub 우선 모델 규칙, Grad-CAM 규칙, 한 단계씩 진행 원칙을 정의합니다. |
| 0단계 데이터 감사 | `../stage0_data_check/docs/monthly20_data_check.md` | 가능한 데이터 범위를 정의합니다: public I20 full-spec image와 `I20/R5`, `I20/R20`, `I20/R60`. |
| 0단계 근거 감사 | `../stage0_data_check/docs/source_reference_check.md` | Re-image 논문 근거, GitHub commit, 논문/GitHub mismatch를 정의합니다. |

## 논문 자료

| 항목 | 자료 | 1단계에서 쓰는 방식 |
| --- | --- | --- |
| Re-image pipeline | `../자료조사/Re-image 요약.md` | 이미지 생성, CNN 구조, 학습 설정, split, 해석 관련 근거입니다. |
| Re-image 원문 PDF | `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | 구현 전 Figure 7, Figure 13, paper detail을 눈으로 확인합니다. |
| Grad-CAM 방법론 | `../자료조사/Grad-CAM요약.md` | Grad-CAM 수식, logit target, activation/gradient heatmap 해석 근거입니다. |
| Grad-CAM 원문 PDF | `../자료조사/Grad-CAM.pdf` | Grad-CAM 구현 전 방법론을 다시 확인합니다. |

## GitHub 자료

| 항목 | 자료 | 1단계에서 쓰는 방식 |
| --- | --- | --- |
| Stock CNN reference repo | `https://github.com/lich99/Stock_CNN` | 1단계 I20 model core 기준입니다. |
| 확인한 commit | `415e2acf2a5013afca67e383acd3edc61fced840` | model code comment에 남길 commit입니다. |
| 확인한 model file | `models/baseline.py` | I20 CNN layer 구조와 forward behavior 기준입니다. |
| 로컬 저장본 | `../stage0_data_check/references/stock_cnn_baseline_415e2ac.py.txt` | 구현 전 offline reference로 사용합니다. |

## 0단계에서 이어받은 1단계 결정

| 결정 | 현재 값 | 이유 |
| --- | --- | --- |
| 직접 가능한 stock 실험 | `I20/R5`, `I20/R20`, `I20/R60` | 현재 로컬 데이터는 20-day image만 있지만 5/20/60 future-return label이 있습니다. |
| Image spec | Full `OHLC + MA + Volume` | 현재 `.dat` 파일은 이미 `20_ma`, `has_vb`가 포함된 렌더링 이미지입니다. |
| Tensor convention | `(batch, channel, height=64, width=60)` | 로컬 `.dat` 파일이 `(N, 64, 60)`으로 reshape됩니다. |
| Label rule | future return `> 0`이면 `1`, 아니면 `0` | Re-image binary classification setup입니다. |
| Loss interface | logits에 대한 `CrossEntropyLoss` | GitHub model은 logits를 반환하고 softmax를 주석 처리합니다. |
| I20 model core | 사용자가 바꾸지 않는 한 `Stock_CNN/models/baseline.py`를 따름 | 사용자가 GitHub 핵심 구현을 따르라고 지시했습니다. |
| Grad-CAM 상태 | 학습 후 필수 | Re-image Figure 13 스타일 해석 산출물이 1단계에 포함됩니다. |

## 계속 보여야 하는 mismatch

| Mismatch | 현재 처리 |
| --- | --- |
| 논문 요약은 first-layer dilation을 강조하지만, GitHub는 I20 dilation을 세 conv layer 모두에 적용합니다. | 사용자가 바꾸지 않는 한 I20 core는 GitHub를 따르고, 코드 주석에 차이를 남깁니다. |
| 논문은 softmax probability를 설명하지만, GitHub는 logits를 반환합니다. | logits로 학습하고 prediction/evaluation 저장 시에만 softmax를 계산합니다. |
| 논문/table의 dimension 표기는 width/height 순서가 섞일 수 있습니다. | `(batch, 1, height=64, width=60)`으로 명시합니다. |
| 현재 데이터는 논문의 모든 window/spec ablation을 지원하지 않습니다. | 1단계는 full paper-wide reproduction이 아니라 public I20 full-spec reproduction이라고 보고합니다. |

## 1-1 근거와 제한사항 재확인

확인 일자:
- 2026-04-30

확인한 필수 자료:

| 자료 | 확인 결과 |
| --- | --- |
| `../PLAN.md` | 한 단계씩 진행, BTC 이전 1단계 재현, GitHub 우선 model core, Grad-CAM 필수 산출 원칙을 재확인했습니다. |
| `../stage0_data_check/docs/monthly20_data_check.md` | 현재 직접 가능한 데이터 범위가 public `monthly_20d` full-spec image뿐임을 재확인했습니다. |
| `../stage0_data_check/docs/source_reference_check.md` | `Stock_CNN` commit과 I20 논문/GitHub mismatch log를 재확인했습니다. |
| `../자료조사/Re-image 요약.md` | image construction, 20-day image size, CNN/training setup, split, interpretation 요구사항을 재확인했습니다. |
| `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | 로컬 파일 존재를 확인했습니다. 특정 page/figure에 묶인 코드 주석을 쓸 때는 PDF viewer로 해당 page를 다시 눈으로 확인해야 합니다. |
| `../자료조사/Grad-CAM요약.md` | Grad-CAM이 target class score/logit, activation gradient, spatial-average channel weight, ReLU heatmap, bilinear upsampling을 사용한다는 점을 재확인했습니다. |
| `../자료조사/Grad-CAM.pdf` | 로컬 파일 존재를 확인했습니다. Grad-CAM 코딩 전 방법론 세부는 로컬 요약을 기준으로 다시 확인합니다. |

확정된 1단계 제한사항:

| 제한사항 | 1-1 결과 |
| --- | --- |
| 직접 실험 | 여전히 `I20/R5`, `I20/R20`, `I20/R60`만 직접 가능합니다. |
| 이미지 소스 | 여전히 `monthly_20d`의 저자 제공 rendered `.dat` image입니다. |
| Image spec | 여전히 full `OHLC + MA + Volume`입니다. 현재 `.dat`에서 A/B/C/D ablation은 복구할 수 없습니다. |
| Tensor convention | 여전히 `(batch, 1, height=64, width=60)`입니다. |
| Label | horizon별 NaN 제거 후 selected future return `> 0`이면 `1`입니다. |
| Model core | 사용자가 바꾸지 않는 한 I20은 `lich99/Stock_CNN/models/baseline.py`를 따릅니다. |
| Dilation mismatch | 계속 표시해야 합니다: 논문 요약은 first-layer dilation을 강조하지만 GitHub는 I20 세 conv layer 모두에 dilation을 씁니다. |
| Loss/softmax mismatch | 계속 표시해야 합니다: 논문은 softmax probability를 설명하지만 GitHub는 logits를 반환합니다. |
| Grad-CAM | 학습 후 필수입니다. raw feature map이 아니라 class-discriminative heatmap입니다. |

1-1 결론:
- 1단계는 여전히 저자 공개 I20 full-spec pipeline의 public-data reproduction입니다.
- 다음 체크리스트인 `1-2. Data loading 세부계획`을 먼저 계획/확인하기 전에는 구현을 시작하지 않습니다.

## 1-2 Data Loading 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/data_loading_plan.md`

Data loading 결정:

| 항목 | 결정 |
| --- | --- |
| Data root | `../테스트/Test/img_data/monthly_20d` |
| File discovery | 1993년부터 2019년까지 deterministic year pair를 사용합니다. |
| Image pattern | `20d_month_has_vb_20_ma_{year}_images.dat` |
| Label pattern | `20d_month_has_vb_20_ma_{year}_labels_w_delay.feather` |
| Image dtype | `.dat`는 `uint8`로 읽습니다. |
| Pixel scale | model tensor에서는 `float32`로 변환하고 `255.0`으로 나눕니다. |
| Image shape | sample `(1, 64, 60)`, batch `(batch_size, 1, 64, 60)`. |
| Label reader | `pandas.read_feather`; 로컬에서 `pyarrow` 사용 가능합니다. |
| Row alignment | 같은 연도와 같은 local row index가 하나의 sample입니다. |
| Memory strategy | image file 총량이 약 `7.9G`이므로 기본은 memory-mapped lazy loading입니다. |
| Shuffling | horizon filtering, split assignment, normalization 결정 전에는 base data를 shuffle하지 않습니다. |
| 이후 결정 | horizon filtering은 1-3, split/normalization은 1-4, DataLoader 성능 설정은 1-6으로 넘깁니다. |

1-2 결론:
- Data loading은 strict per-year row-alignment validation을 가진 shard-aware lazy image loading으로 구현해야 합니다.
- 아직 data-loading 코드는 작성하지 않았습니다.

## 1-3 Label Construction 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/label_construction_plan.md`

Label 결정:

| 항목 | 결정 |
| --- | --- |
| Main targets | `Ret_5d`, `Ret_20d`, `Ret_60d`. |
| Experiments | `stage1_i20_r5`, `stage1_i20_r20`, `stage1_i20_r60`. |
| Label rule | selected future return `> 0`이면 `label = 1`, 아니면 `0`. |
| Zero returns | 정확히 0인 return은 class `0`입니다. Up은 strictly positive return입니다. |
| NaN handling | 선택한 target return이 missing인 row는 horizon별로 따로 제거합니다. |
| Model input | image tensor만 사용합니다. return column과 metadata는 input feature가 아닙니다. |
| Required metadata | `Date`, `StockID`, `MarketCap`, return columns, `EWMA_vol`, `year`, `local_row`, `target_return_name`, `target_return`, `label`을 보존합니다. |
| `Ret_month` | metadata 또는 optional future experiment로 보존합니다. 고정 1단계 target은 아닙니다. |
| Leakage rule | future return은 label/evaluation target일 뿐 model input으로 쓰지 않습니다. |

유효 row 수:

| Target | Valid rows | Valid row 기준 positive rate |
| --- | ---: | ---: |
| `Ret_5d` | 2,190,724 | 50.48% |
| `Ret_20d` | 2,180,610 | 51.23% |
| `Ret_60d` | 2,151,213 | 53.17% |

1-3 결론:
- 1단계는 같은 I20 image row에서 horizon별 filtered dataset 3개를 만듭니다.
- split과 normalization은 `1-4`로 넘깁니다.

## 1-4 Split and Normalization 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/split_normalization_plan.md`
- `data_inventory/stage1_horizon_counts_by_year.csv`
- `data_inventory/stage1_horizon_counts_by_period.csv`

Split 결정:

| 항목 | 결정 |
| --- | --- |
| Train/validation pool | 1993-2000. |
| Test period | 2001-2019. |
| Train/validation split | horizon별 NaN filtering 이후 deterministic 70/30 random split. |
| Split seed | 논문 요약에 seed가 없으므로 재현용으로 `42`를 명시합니다. |
| Split unit | `(year, local_row)`로 식별되는 sample row. |
| Stratification | 논문 요약이 random 70/30만 말하므로 기본값은 `false`입니다. |
| Test leakage | test row는 training, early stopping, normalization-stat fitting에 쓰지 않습니다. |

Normalization 결정:

| 항목 | 결정 |
| --- | --- |
| Pixel scaling | `uint8 / 255.0`으로 `[0, 1]` 변환. |
| Standardization | training subset에서만 scalar pixel mean/std를 fit합니다. |
| 적용 | train mean/std를 train, validation, test 모두에 적용합니다. |
| Scope | `Ret_5d`, `Ret_20d`, `Ret_60d`별 filtered train row가 다르므로 horizon별로 저장합니다. |
| 저장 metadata | horizon별 `normalization.json`, `split_summary.json`을 저장합니다. |

1-4 결론:
- 1단계는 논문 요약의 split인 1993-2000 train/validation, 2001-2019 test를 따릅니다.
- seed는 논문에 없으므로 명시적 재현 선택입니다.
- 이제 `1-5`에서 model implementation 계획으로 넘어갈 수 있지만, 아직 model code는 작성하지 않았습니다.

## 1-5 Baseline CNN 구현 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/baseline_cnn_implementation_plan.md`

기준 구현:

| 항목 | 결정 |
| --- | --- |
| Repository | `https://github.com/lich99/Stock_CNN` |
| Commit | `415e2acf2a5013afca67e383acd3edc61fced840` |
| File | `models/baseline.py` |
| Local copy | `../stage0_data_check/references/stock_cnn_baseline_415e2ac.py.txt` |

모델 결정:

| 항목 | 결정 |
| --- | --- |
| 예정 파일 | `src/models/stock_cnn.py` |
| 예정 class | `StockCNNI20` |
| 직접 범위 | public `I20/R5`, `I20/R20`, `I20/R60` 재현. |
| 입력 convention | `(batch_size, 1, height=64, width=60)`. |
| GitHub compatibility | `layer1`, `layer2`, `layer3`, `fc1`, logits-returning forward behavior를 유지합니다. |
| Softmax | `forward` 안에서 적용하지 않고 evaluation/prediction output에서만 적용합니다. |
| 목표 parameter count | `708,866`. |

I20 architecture 구현 고정:

| Block | Layer values |
| --- | --- |
| `layer1` | `Conv2d(1,64,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`, BN, LeakyReLU `0.01`, MaxPool `(2,1)` |
| `layer2` | `Conv2d(64,128,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`, BN, LeakyReLU `0.01`, MaxPool `(2,1)` |
| `layer3` | `Conv2d(128,256,kernel=(5,3),stride=(3,1),dilation=(2,1),padding=(12,1))`, BN, LeakyReLU `0.01`, MaxPool `(2,1)` |
| `fc1` | Dropout `0.5`, Linear `46080 -> 2` |

Shape contract:

| Step | Shape |
| --- | --- |
| Input | `(batch_size, 1, 64, 60)` |
| After `layer1` | `(batch_size, 64, 13, 60)` |
| After `layer2` | `(batch_size, 128, 5, 60)` |
| After `layer3` | `(batch_size, 256, 3, 60)` |
| Flatten | `(batch_size, 46080)` |
| Logits | `(batch_size, 2)` |

Grad-CAM 준비:
- model `forward`에서는 Grad-CAM을 계산하지 않습니다.
- 이후 `1-8`에서 convolution module에 hook을 걸 수 있게 구조를 유지합니다.
- 예정 Grad-CAM target names:
  - `layer1_conv`: `model.layer1[0]`
  - `layer2_conv`: `model.layer2[0]`
  - `layer3_conv`: `model.layer3[0]`

1-5 결론:
- 1단계 baseline model 구현은 GitHub I20 core를 중심으로 작성하면 됩니다.
- 논문/GitHub dilation mismatch는 계속 문서화합니다.
- training behavior는 아직 `1-6`으로 넘기며, model code는 아직 작성하지 않았습니다.

## 1-6 Training Loop 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/training_loop_plan.md`

논문/근거 학습 설정:

| 항목 | 결정 |
| --- | --- |
| Loss | logits에 대한 `CrossEntropyLoss`. |
| Optimizer | Adam. |
| Learning rate | `1e-5`. |
| Batch size | `128`. |
| Initialization | Xavier; 구현 variant는 `xavier_uniform_`으로 고정. |
| Dropout | `0.5`, model classifier에서 이미 고정. |
| Early stopping | validation loss, patience `2`, `min_delta=0.0`. |
| Train/test split | `1-4` 기준: 1993-2000 train/validation, 2001-2019 test. |
| Normalization | `1-4` 기준: horizon별 train-only pixel mean/std. |
| 독립 학습 | horizon마다 5회 학습 후 test `prob_up` 평균. |

논문 미보고 항목에 대한 재현 선택:

| 항목 | 결정 |
| --- | --- |
| Split seed | `1-4`에서 이미 `42`로 고정. |
| Run seeds | `[42, 43, 44, 45, 46]`. |
| Epoch cap | 논문에 exact epoch가 없으므로 safety cap `100`. |
| Weight decay | 논문에 없으므로 `0.0`. |
| LR schedule | 논문에 없으므로 기본 사용 안 함. |
| Mixed precision | 기본 비활성화. |
| Gradient clipping | 기본 비활성화. |
| Device | `auto`; full run target은 Kaggle CUDA GPU. |
| Determinism | Python, NumPy, PyTorch CPU/CUDA seed를 고정하고 가능한 경우 deterministic cuDNN 사용. |

Checkpoint/logging 결정:

| 항목 | 결정 |
| --- | --- |
| Best checkpoint | `outputs/checkpoints/{experiment_name}/seed_{run_seed}/best.pt`. |
| Last checkpoint | `outputs/checkpoints/{experiment_name}/seed_{run_seed}/last.pt`. |
| Best 기준 | 가장 낮은 validation loss. |
| Test checkpoint | `best.pt` 사용. |
| Train history | `outputs/metrics/{experiment_name}/seed_{run_seed}/train_history.csv`. |
| Train metadata | `outputs/metrics/{experiment_name}/seed_{run_seed}/train_metadata.json`. |

1-6 결론:
- 1단계 full reproduction은 horizon마다 `StockCNNI20`을 5개 seed로 독립 학습하고 probability를 평균해야 합니다.
- local smoke test는 1개 seed와 tiny subset만 사용하며 reproduction result로 보고하지 않습니다.
- Kaggle path/performance 세부사항은 `1-6K`로 넘기며, training code는 아직 작성하지 않았습니다.

## 1-6K Kaggle Runner와 Environment Config 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/kaggle_runner_plan.md`
- `configs/env_kaggle.yaml`
- `configs/env_local.yaml`
- `notebooks/README.md`

환경 결정:

| 항목 | 결정 |
| --- | --- |
| Full run environment | Kaggle Notebook. |
| Codebase rule | 하나의 공유 `src/`; Kaggle 전용/local 전용 코드베이스를 만들지 않음. |
| Kaggle data input | `monthly_20d`를 Kaggle Dataset으로 upload/attach. |
| 권장 dataset 이름 | `reimage-monthly-20d`. |
| Kaggle data root | `/kaggle/input/reimage-monthly-20d/monthly_20d`. |
| Kaggle output root | `/kaggle/working/stage1_reimage_reproduction/outputs`. |
| Local 역할 | 기본적으로 smoke test only. |
| Local data root | `/Users/jaehyeonjeong/Desktop/논문/테스트/Test/img_data/monthly_20d`. |

Kaggle runtime 기본값:

| 항목 | 결정 |
| --- | --- |
| Device | full run은 `cuda`. |
| CUDA unavailable behavior | full run에서는 명확히 실패. |
| DataLoader workers | 보수적 기본값 `num_workers: 2`. |
| Pin memory | Kaggle/CUDA에서 `true`. |
| Mixed precision | 논문에 AMP 보고가 없으므로 `false`. |
| Internet | 권장 code snapshot mode에서는 필요 없음. |

Run modes:

| Mode | Seeds | 용도 |
| --- | --- | --- |
| `smoke` | `[42]` | shape/path/checkpoint smoke test only. |
| `full_single_seed` | `[42]` | 첫 full baseline check. |
| `full_paper_style` | `[42, 43, 44, 45, 46]` | 논문식 5-run averaged reproduction. |

Manifest requirement:
- 모든 Kaggle run은 `outputs/run_manifests/run_manifest.json`을 저장해야 합니다.
- manifest에는 dataset/code version, config files, seeds, package versions, CUDA/GPU 정보, source references를 기록합니다.

1-6K 결론:
- 이제 Stage 1 구현은 local/Kaggle path를 hardcoding하지 않고 environment config를 기준으로 작성하면 됩니다.
- 실제 runner notebook과 training command는 구현 시작 시점으로 넘깁니다.
- evaluation schema는 `1-7`, Grad-CAM 실행은 `1-8`로 계속 넘깁니다.

## 1-7 Evaluation과 Prediction-output 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/evaluation_prediction_plan.md`

Prediction 결정:

| 항목 | 결정 |
| --- | --- |
| Probability | `softmax(logits, dim=1)`; 논문 근거: Re-image 요약이 softmax Up probability를 CNN/training pp.20-22로 매핑합니다. |
| Up probability | class `1`의 `prob_up`. |
| Prediction threshold | 50% threshold; 논문 근거: Re-image 요약이 threshold를 CNN/training pp.20-22로 매핑합니다. 구현은 `prob_up >= 0.5`를 사용합니다. |
| Tie rule | 정확히 `0.5`면 class `1`; 논문은 tie behavior를 보고하지 않으므로 implementation choice입니다. |
| 5-run averaging | logits가 아니라 probability를 평균. |
| Seed별 file | `outputs/predictions/{experiment_name}/seed_{run_seed}/test_predictions.csv`. |
| Averaged file | `outputs/predictions/{experiment_name}/averaged/test_predictions.csv`. |

Metric 결정:

| 항목 | 결정 |
| --- | --- |
| Core classification | accuracy, precision, recall, F1, ROC AUC, average precision, Brier, log loss, confusion matrix. |
| Class balance | positive rate, negative rate, predicted positive rate. |
| Baseline comparison | majority-class accuracy와 accuracy minus majority-class accuracy. |
| Return diagnostics | prediction signal과 target return의 global/date-wise Pearson/Spearman correlation. |
| Paper-style output | stock cross-sectional Stage 1 evaluation으로 decile ranking과 H-L portfolio metrics 계획. |

1-7 결론:
- prediction file은 `Date`, `StockID`, `MarketCap`, target returns, labels, logits/probabilities, correctness를 보존해야 합니다.
- 1단계 evaluation은 classification metric과 stock cross-sectional ranking output을 모두 포함합니다.
- 실제 구현은 `1-I7`, Grad-CAM sample selection은 `1-8`로 넘깁니다.

## 1-I7 Evaluation과 Prediction-output 구현

구현 일자:
- 2026-05-01

산출물:
- `docs/evaluation_prediction_implementation.md`
- `src/stage1_reimage/evaluation/prediction.py`
- `scripts/evaluate_stage1_predictions.py`

구현 근거 mapping:

| 항목 | 근거 | 코드 위치 |
| --- | --- | --- |
| Model output interface | `lich99/Stock_CNN/models/baseline.py`, commit `415e2acf2a5013afca67e383acd3edc61fced840`; local model implementation `src/stage1_reimage/models/stock_cnn.py` | `predict_loader()`는 `model(images)`를 logits로 유지합니다. |
| Softmax probability | Re-image 요약은 CNN/training probability 해석을 pp.20-22로 매핑; `docs/evaluation_prediction_plan.md` | `predict_loader()`에서 evaluation 시에만 `torch.softmax(logits, dim=1)` 적용. |
| Threshold | `docs/evaluation_prediction_plan.md`; local Re-image 요약 pp.20-22 mapping의 50% threshold | `tie_break_class: 1`일 때 `prob_up >= 0.5`. |
| Tie rule | 논문이 정확히 `0.5`인 경우를 별도 보고하지 않으므로 implementation convention | config의 `evaluation.threshold`, `evaluation.tie_break_class`. |
| Averaging | `docs/evaluation_prediction_plan.md` | `average_seed_predictions()`는 logits가 아니라 softmax probability를 평균합니다. |
| Prediction metadata | Stage 1 data-loading/label plan | Prediction CSV는 `Date`, `StockID`, `MarketCap`, target returns, labels, logits/probabilities, correctness를 보존합니다. |
| Metrics | `docs/evaluation_prediction_plan.md` | `compute_classification_metrics()`, `compute_correlation_metrics()`. |

검증:
- `python -m compileall src scripts`
- `python scripts/check_scaffold.py --config configs/env_local.yaml`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --max-rows 4`
- `python scripts/evaluate_stage1_predictions.py --config configs/env_local.yaml --horizon stage1_i20_r20 --split validation --average-seed-predictions 42`

1-I7 결론:
- seed별 prediction export와 averaged prediction export를 구현했습니다.
- classification metric과 prediction-return correlation metric을 구현했습니다.
- `outputs/` 아래 smoke output은 reproduction result가 아니며 GitHub에서 제외합니다.
- portfolio/decile H-L metric은 최종 report convention 재확인 전까지 보류합니다.
- 이제 `1-I8` Grad-CAM sample selection이 prediction CSV를 사용할 수 있습니다.

## 1-I7R 코드 주석/가독성 보강

수정 일자:
- 2026-05-01

산출물:
- `checklist_results/1-I7R_code_annotation.md`

근거/policy mapping:

| 항목 | 근거 | Stage 1 코드 조치 |
| --- | --- | --- |
| 코드 가독성 | Root `PLAN.md`, 코드 작성 원칙 | 모든 코드에 자세한 설명 주석을 남긴다는 고정 규칙을 추가. |
| 한국어 코드 설명 | Root `PLAN.md`, 사용자 review 요청 | 설명 주석/docstring은 기본적으로 한국어로 작성하도록 정리. |
| Tensor shape 주석 | Root `PLAN.md`, 사용자 review 요청 | image tensor `(1,64,60)`, DataLoader batch `(B,1,64,60)`, CNN feature map, logits `(B,2)`, probability output 설명 추가. |
| Leakage 주석 | Root `PLAN.md`, Stage 1 leakage rule | metadata/return은 보존하지만 CNN input으로 넣지 않는 지점에 설명 추가. |
| Data movement 주석 | 사용자 review 요청 | config -> loader -> dataset -> model -> loss -> prediction CSV로 값이 이동하는 방식 설명 추가. |

1-I7R 결론:
- 코드 동작은 바뀌지 않았습니다.
- 각 단계가 어떻게 작동하는지 읽을 수 있도록 코드 안 설명을 보강했습니다.
- 읽는 사람을 위한 주석과 docstring은 기본적으로 한국어로 정리했습니다.

## 1-8 Grad-CAM 세부계획

확인 일자:
- 2026-04-30

산출물:
- `docs/gradcam_plan.md`

근거 mapping:

| 항목 | 근거 | 1단계 결정 |
| --- | --- | --- |
| Figure layout | Re-image Figure 13; 로컬 Re-image 요약은 해석 자료를 pp.41-49로 매핑; 사용자가 제공한 Figure 13 이미지 | 2019년 `Up`/`Down` prediction에 대해 original image row와 CNN-layer Grad-CAM row를 재현. |
| Grad-CAM target score | `../자료조사/Grad-CAM요약.md`가 정리한 Grad-CAM 원전 pp.4-6 | softmax probability가 아니라 pre-softmax target-class logit 사용. |
| Grad-CAM formula | `../자료조사/Grad-CAM요약.md`가 정리한 Grad-CAM 원전 pp.4-6 | activation gradient, spatial-average channel weight, `ReLU(sum alpha A)`, bilinear upsampling 사용. |
| Layer 선택 주의 | `../자료조사/Grad-CAM요약.md`가 정리한 Grad-CAM 원전 appendix/pp.12-21 | layer 선택이 class-discriminative behavior에 영향을 주므로 I20 세 conv layer를 모두 표시. |
| Target layer | `docs/baseline_cnn_implementation_plan.md`; `lich99/Stock_CNN`식 Stage 1 model | `model.layer1[0]`, `model.layer2[0]`, `model.layer3[0]` hook. |
| Prediction source | `docs/evaluation_prediction_plan.md` | 학습 이후 per-seed 또는 averaged prediction CSV에서 sample 선택. |

Grad-CAM 결정:

| 항목 | 결정 |
| --- | --- |
| Primary experiment | `stage1_i20_r20`. |
| Extended experiments | `stage1_i20_r5`, `stage1_i20_r20`, `stage1_i20_r60`. |
| Primary year | `2019`. |
| Samples | figure마다 `Up` prediction 10개와 `Down` prediction 10개. |
| Target class | 기본값은 predicted class. |
| Heatmap normalization | heatmap별 min-max normalization; Re-image figure가 정확한 normalization을 명시하지 않으므로 구현상 선택. |
| Colormap | blue/cyan 스타일 heatmap; 시각 비교 전까지 구현상 선택. |
| Full paper-style ensemble heatmap | averaged prediction에서 sample 선택; seed별 Grad-CAM 계산 후 normalized heatmap을 평균. |
| Correctness filter | 기본적으로 correct prediction만 요구하지 않음. Figure는 받은 classification을 설명하는 그림입니다. |

1-8 결론:
- 1단계 Grad-CAM은 raw feature map도 Guided Grad-CAM도 아닌, predicted-class
  plain Grad-CAM으로 정의합니다.
- 실제 코드는 `1-I8`에서 구현합니다.
- 최종 코드 주석을 넣기 전에는 로컬 자동 PDF text extraction이 불가능하므로
  PDF 페이지를 눈으로 다시 확인합니다.

## 1-9 1단계 Report Plan

확인 일자:
- 2026-04-30

산출물:
- `docs/stage1_report_plan.md`

보고서 근거 mapping:

| 항목 | 근거 | 1단계 보고서 사용 |
| --- | --- | --- |
| 보고 범위 | `../PLAN.md`; 0단계 데이터 감사 | 1단계를 public I20 full-spec reproduction으로 보고하고, full paper-wide reproduction으로 쓰지 않음. |
| 데이터 제한 | `../stage0_data_check/docs/monthly20_data_check.md`; `docs/label_construction_plan.md` | 사용 가능한 rendered image가 `I20/R5`, `I20/R20`, `I20/R60`만 직접 지원한다고 명시. |
| Split and normalization | `docs/split_normalization_plan.md`; Re-image summary CNN/training pp.12-21 | 1993-2000 train/validation, 2001-2019 test, 70/30 split, train-only pixel standardization 보고. |
| Model structure | `docs/baseline_cnn_implementation_plan.md`; `Stock_CNN` commit `415e2acf...` | GitHub식 I20 architecture와 dilation mismatch 보고. |
| Training reproducibility | `docs/training_loop_plan.md`; `docs/kaggle_runner_plan.md` | Kaggle `full_paper_style`을 primary reproduction으로 두고 local smoke test는 reproduction이 아니라고 명시. |
| Classification and correlation | `docs/evaluation_prediction_plan.md`; Re-image Table 2, U.S. experiments pp.21-33 | classification, majority-baseline, correlation table 생성. |
| Portfolio outputs | `docs/evaluation_prediction_plan.md`; Re-image U.S. portfolio results pp.21-33 | convention이 맞는 경우 stock cross-sectional decile/H-L table 생성. |
| Grad-CAM | `docs/gradcam_plan.md`; Re-image Figure 13; Grad-CAM pp.4-6 | Figure 13 스타일 Grad-CAM을 포함하고 class-discriminative heatmap으로 해석. |

필수 보고서 파일:
- `reports/stage1_reproduction_report.md`

필수 report tables:
- `reports/tables/stage1_dataset_summary.csv`
- `reports/tables/stage1_split_summary.csv`
- `reports/tables/stage1_training_summary.csv`
- `reports/tables/stage1_classification_metrics.csv`
- `reports/tables/stage1_majority_baseline_comparison.csv`
- `reports/tables/stage1_correlation_metrics.csv`
- `reports/tables/stage1_portfolio_metrics.csv`
- `reports/tables/stage1_paper_comparison.csv`
- `reports/tables/stage1_gradcam_samples.csv`
- `reports/tables/stage1_artifact_manifest.csv`

1-9 결론:
- 보고서는 같은 조건의 paper cell끼리만 비교해야 합니다.
- `I20/R20`, `I20/R60` classification paper value는 로컬 요약에서 확인됐지만,
  `I20/R5` classification paper value는 보고 전 PDF에서 확인해야 합니다.
- H-L portfolio 비교는 decile, value-weighting, annualization convention을
  확인한 뒤에만 합니다.
- 실제 report 생성은 `1-I12`로 넘깁니다.

## 1-I0 Implementation Readiness Review

확인 일자:
- 2026-04-30

산출물:
- `docs/implementation_readiness_review.md`

Readiness 판정:
- `1-I1. 공통 code/config scaffold 구현`을 시작할 준비가 됐습니다.
- 이 readiness는 public `monthly_20d` I20 full-spec reproduction 경로,
  즉 `I20/R5`, `I20/R20`, `I20/R60`에 적용됩니다.

확인 결과:

| 항목 | 결과 |
| --- | --- |
| Planning gate | `1-0`부터 `1-9`까지 완료. |
| Local data | image `.dat` 27개와 label `.feather` 27개 확인. |
| Local smoke-test dependency | `torch`, `numpy`, `pandas`, `pyarrow`, `sklearn`, `matplotlib`, `yaml` import 가능. |
| Config | `configs/env_local.yaml`, `configs/env_kaggle.yaml` 존재. |
| Code 상태 | `src/`, `scripts/`, `notebooks/`에는 README placeholder만 있음. |

Readiness 제한사항:
- 루트 `PLAN.md`에는 raw OHLC image generator/MA/volume 항목이 있지만,
  즉시 구현은 이미 렌더링된 public I20 `.dat` 파일을 사용합니다. 첫 public-data
  reproduction에는 허용되지만, paper-wide pipeline이라고 주장하려면 raw image
  generator 작업을 별도 gate로 추가해야 합니다.
- 현재 workspace는 Git repository가 아닙니다. Kaggle full run 전에는 code
  snapshot 또는 Git commit으로 manifest provenance를 만족시켜야 합니다.
- Kaggle full run에는 `monthly_20d`를 `reimage-monthly-20d` dataset으로
  upload/attach하거나 Kaggle config path를 수정해야 합니다.

1-I0 결론:
- `1-I1`을 막는 blocker는 없습니다.
- `1-I1` 구현과 보고 전에는 다음 구현 gate로 넘어가지 않습니다.
