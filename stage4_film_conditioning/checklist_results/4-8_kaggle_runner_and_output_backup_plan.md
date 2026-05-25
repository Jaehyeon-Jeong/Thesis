# 4-8 Kaggle Runner And Output Backup Plan

## English

Status: complete for the planning phase.

This item fixes the Stage 4 Kaggle execution contract before implementation.
The goal is to avoid the Stage 1 failure mode where training finished but the
usable result package was incomplete because predictions, metrics, or Grad-CAM
were not preserved.

## Fixed Execution Scope

Primary Stage 4 run:
- Image/model baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
- Context window: matched `60` days.
- Context features:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
- Ablations:
  - `concat`
  - `gating`
  - `film_gamma`
  - `film_full`
- First full sanity run: four ablations, seed `42`.
- Later robustness run: the same four ablations with seeds
  `42, 43, 44, 45, 46`.

News context is not included in this first runner. It remains a separate
second-phase `4-N` runner after source/date/leakage rules are locked.

## Kaggle Input Contract

The Stage 4 Kaggle notebook should attach:
- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
  or the equivalent uploaded folder/zip.
- BTC OHLCV source:
  `/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv`
  or auto-detected BTC CSV under `/kaggle/input`.
- Fear & Greed source:
  the Kaggle dataset
  `ashishpatel8736/historical-and-fear-greed-index-datasets`, or an explicitly
  supplied CSV path after audit.

The first 4-A/B/C/D runner must fail early if:
- the Stage 4 code snapshot is missing required scripts;
- the BTC CSV cannot be found;
- the F&G file cannot be found or does not overlap the BTC test period;
- output directories are not writable.

## Kaggle Working Paths

```python
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning")
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage4_saved_outputs")
```

If IO becomes a bottleneck, small CSV inputs may be copied to `/tmp`, but this
must not change the dataset contents or split logic.

## Required Runner Stages

The one-cell Kaggle runner should execute the following stages in this order:

1. Copy or extract the Stage 4 code snapshot into `PROJECT_ROOT`.
2. Assert required Stage 4 scripts exist.
3. Patch `configs/env_kaggle.yaml` with Kaggle paths and runtime settings.
4. Audit BTC and F&G sources.
5. Build context features and the train-only context scaler.
6. Train the selected ablation model.
7. Evaluate classification predictions.
8. Evaluate trading metrics.
9. Generate Grad-CAM and context/gate/gamma/beta exports.
10. Run an output receipt check.
11. Summarize available Stage 4 results.

Expected implementation scripts:
- `scripts/audit_stage4_context_sources.py`
- `scripts/build_stage4_context_features.py`
- `scripts/run_stage4_context_model.py`
- `scripts/evaluate_stage4_predictions.py`
- `scripts/evaluate_stage4_trading.py`
- `scripts/generate_stage4_gradcam_context.py`
- `scripts/check_stage4_outputs.py`
- `scripts/summarize_stage4_results.py`

These names are the implementation target for 4-I1 through 4-I13; this 4-8
document does not claim the scripts already exist.

## Backup Phases

The runner must create backup zips outside `PROJECT_ROOT` after each major
stage:

- `after_context_build`
- `after_train`
- `after_prediction_eval`
- `after_trading_eval`
- `after_gradcam`
- `after_output_check`
- `after_summary`

Backup root:

```text
/kaggle/working/stage4_saved_outputs
```

Archive naming:

```text
<experiment>_seed<seed>_<phase>_<timestamp>_outputs.zip
```

Receipt naming:

```text
<experiment>_seed<seed>_<phase>_<timestamp>_receipt.json
```

Example experiment names:

```text
stage4_i60_ohlc_ma_vb_r20_concat_ctx60
stage4_i60_ohlc_ma_vb_r20_gating_ctx60
stage4_i60_ohlc_ma_vb_r20_film_gamma_ctx60
stage4_i60_ohlc_ma_vb_r20_film_full_ctx60
```

Each receipt must record:
- experiment name;
- ablation name;
- image window, image spec, return horizon, context window;
- seed;
- code snapshot path;
- BTC and F&G source file paths;
- context feature list;
- expected output paths;
- archive path and size;
- creation timestamp.

## Completion Rule

An experiment is complete only when the output check passes. A checkpoint alone
is not enough.

The output checker must verify:
- context feature table;
- context feature audit;
- context scaler/statistics JSON;
- best checkpoint;
- last checkpoint;
- train history;
- train metadata;
- test predictions;
- classification metrics;
- trading metrics;
- Grad-CAM figure;
- Grad-CAM selected sample metadata;
- selected context values;
- modulation exports for applicable models:
  - gate values for `gating`;
  - gamma values for `film_gamma`;
  - gamma and beta values for `film_full`;
- run manifest;
- explanation/export manifest.

## Resume Rule

`SKIP_COMPLETED=True` should mean:
- skip only when output check passes;
- if checkpoint exists but predictions are missing, run evaluation, trading,
  Grad-CAM, and output check;
- if context features and context scaler already pass audit, reuse them;
- if one ablation fails, keep completed outputs and continue when
  `CONTINUE_ON_ERROR=True`.

## Runtime Settings

Primary Stage 4 runs should start with the strict Stage 2-style settings:

```text
BATCH_SIZE = 128
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
```

Speed diagnostics may use mixed precision or larger batch size, but those runs
must be labeled as diagnostics and not mixed with the strict comparison table.

## Kaggle Notebook Targets

Implementation should later provide these notebook one-cell wrappers:

- `notebooks/kaggle_stage4_single_ablation_one_cell.md`
- `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`
- `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`
- `notebooks/kaggle_stage4_results_viewer_one_cell.md`

## First Run Order

Recommended execution order:

1. Smoke: `concat`, seed `42`, tiny rows, 2 Grad-CAM samples per predicted
   class.
2. Full sanity: four ablations, seed `42`, strict settings.
3. Five-seed robustness: four ablations, seeds `42, 43, 44, 45, 46`.
4. News-context runner only after the numeric context ablation is stable.

## 한국어

상태: 계획 단계 완료.

이 항목은 Stage 4 구현 전에 Kaggle 실행 계약을 고정합니다. 목적은 Stage 1에서
겪었던 문제, 즉 학습은 끝났지만 prediction, metric, Grad-CAM이 보존되지 않아
결과 패키지가 불완전해지는 상황을 반복하지 않는 것입니다.

## 고정 실행 범위

Primary Stage 4 run:
- Image/model baseline: Stage 2 `I60/R20/ohlc_ma_vb`.
- Context window: matched `60` days.
- Context features:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
- Ablation:
  - `concat`
  - `gating`
  - `film_gamma`
  - `film_full`
- 첫 full sanity run: 네 ablation, seed `42`.
- 이후 robustness run: 같은 네 ablation을 seed `42, 43, 44, 45, 46`으로 실행.

뉴스 context는 첫 4-A/B/C/D runner에 넣지 않습니다. source/date/leakage 규칙이
고정된 뒤 별도 second-phase `4-N` runner로 둡니다.

## Kaggle input 계약

Stage 4 Kaggle notebook은 다음 input을 attach해야 합니다.
- Stage 4 code snapshot:
  `/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning`
  또는 동일한 uploaded folder/zip.
- BTC OHLCV source:
  `/kaggle/input/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024/btc_1d_data_2018_to_2025.csv`
  또는 `/kaggle/input` 아래에서 auto-detect되는 BTC CSV.
- Fear & Greed source:
  Kaggle dataset
  `ashishpatel8736/historical-and-fear-greed-index-datasets`, 또는 audit 후
  명시된 CSV path.

첫 4-A/B/C/D runner는 다음 상황에서 초기에 실패해야 합니다.
- Stage 4 code snapshot에 필요한 script가 없음;
- BTC CSV를 찾을 수 없음;
- F&G file을 찾을 수 없거나 BTC test period와 겹치지 않음;
- output directory에 쓸 수 없음.

## Kaggle working path

```python
CODE_INPUT = Path("/kaggle/input/datasets/moskow/stage4/stage4_film_conditioning")
PROJECT_ROOT = Path("/kaggle/working/stage4_film_conditioning")
DATA_ROOT = Path("/kaggle/input")
BACKUP_ROOT = Path("/kaggle/working/stage4_saved_outputs")
```

IO가 병목이면 작은 CSV input은 `/tmp`로 복사할 수 있습니다. 다만 dataset 내용이나
split logic은 절대 바꾸지 않습니다.

## 필수 runner 단계

Kaggle one-cell runner는 다음 순서로 실행해야 합니다.

1. Stage 4 code snapshot을 `PROJECT_ROOT`로 복사/압축해제.
2. 필요한 Stage 4 script 존재 여부 확인.
3. `configs/env_kaggle.yaml`에 Kaggle path와 runtime setting patch.
4. BTC와 F&G source audit.
5. Context feature와 train-only context scaler 생성.
6. 선택한 ablation model 학습.
7. Classification prediction 평가.
8. Trading metric 평가.
9. Grad-CAM과 context/gate/gamma/beta export 생성.
10. Output receipt check 실행.
11. 사용 가능한 Stage 4 result summary 생성.

구현 대상 script 이름:
- `scripts/audit_stage4_context_sources.py`
- `scripts/build_stage4_context_features.py`
- `scripts/run_stage4_context_model.py`
- `scripts/evaluate_stage4_predictions.py`
- `scripts/evaluate_stage4_trading.py`
- `scripts/generate_stage4_gradcam_context.py`
- `scripts/check_stage4_outputs.py`
- `scripts/summarize_stage4_results.py`

이 이름들은 4-I1부터 4-I13까지의 구현 목표입니다. 이 4-8 문서는 해당 script가
이미 존재한다고 주장하지 않습니다.

## Backup phase

Runner는 각 주요 단계 뒤에 `PROJECT_ROOT` 밖에 backup zip을 만들어야 합니다.

- `after_context_build`
- `after_train`
- `after_prediction_eval`
- `after_trading_eval`
- `after_gradcam`
- `after_output_check`
- `after_summary`

Backup root:

```text
/kaggle/working/stage4_saved_outputs
```

Archive 이름:

```text
<experiment>_seed<seed>_<phase>_<timestamp>_outputs.zip
```

Receipt 이름:

```text
<experiment>_seed<seed>_<phase>_<timestamp>_receipt.json
```

예시 experiment name:

```text
stage4_i60_ohlc_ma_vb_r20_concat_ctx60
stage4_i60_ohlc_ma_vb_r20_gating_ctx60
stage4_i60_ohlc_ma_vb_r20_film_gamma_ctx60
stage4_i60_ohlc_ma_vb_r20_film_full_ctx60
```

각 receipt에는 다음을 기록합니다.
- experiment name;
- ablation name;
- image window, image spec, return horizon, context window;
- seed;
- code snapshot path;
- BTC/F&G source file path;
- context feature list;
- expected output paths;
- archive path와 size;
- 생성 timestamp.

## 완료 판정 규칙

Experiment는 output check가 통과해야만 완료입니다. Checkpoint만 있는 것은 완료가
아닙니다.

Output checker는 다음을 확인해야 합니다.
- context feature table;
- context feature audit;
- context scaler/statistics JSON;
- best checkpoint;
- last checkpoint;
- train history;
- train metadata;
- test predictions;
- classification metrics;
- trading metrics;
- Grad-CAM figure;
- Grad-CAM selected sample metadata;
- selected context values;
- 해당 model의 modulation export:
  - `gating`: gate values;
  - `film_gamma`: gamma values;
  - `film_full`: gamma and beta values;
- run manifest;
- explanation/export manifest.

## Resume 규칙

`SKIP_COMPLETED=True`의 의미:
- output check가 통과한 경우만 skip;
- checkpoint는 있지만 prediction이 없으면 evaluation, trading, Grad-CAM,
  output check를 다시 실행;
- context feature와 context scaler가 audit을 통과하면 재사용;
- 하나의 ablation이 실패해도 `CONTINUE_ON_ERROR=True`면 완료된 output을 보존하고
  다음 설정으로 진행.

## Runtime setting

Primary Stage 4 run은 Stage 2 strict setting으로 시작합니다.

```text
BATCH_SIZE = 128
MIXED_PRECISION = False
DATA_PARALLEL = False
FAST_CUDNN = False
```

Mixed precision이나 더 큰 batch size는 speed diagnostic으로만 사용할 수 있습니다.
그런 run은 strict comparison table과 섞지 않고 diagnostic이라고 표시해야 합니다.

## Kaggle notebook 목표

구현 단계에서 다음 one-cell wrapper를 제공합니다.

- `notebooks/kaggle_stage4_single_ablation_one_cell.md`
- `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`
- `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`
- `notebooks/kaggle_stage4_results_viewer_one_cell.md`

## 첫 실행 순서

권장 실행 순서:

1. Smoke: `concat`, seed `42`, tiny rows, predicted class별 Grad-CAM sample 2개.
2. Full sanity: 네 ablation, seed `42`, strict setting.
3. Five-seed robustness: 네 ablation, seed `42, 43, 44, 45, 46`.
4. News-context runner는 numeric context ablation이 안정화된 뒤 실행.
