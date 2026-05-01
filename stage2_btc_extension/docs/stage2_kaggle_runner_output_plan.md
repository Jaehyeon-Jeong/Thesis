# Stage 2 Kaggle Runner and Output Plan

## English

Status: planning complete for checklist 2-8. Implementation happens later in
`2-I9` and `2-I10` after the Stage 2 scripts exist.

Purpose:
- Keep Stage 2 execution consistent with the Stage 1 Kaggle policy.
- Use one Kaggle cell as a wrapper, while the actual implementation remains in
  repo `src/` and `scripts/`.
- Define output paths before writing the runner code.

Source basis:
- Root `PLAN.md`, execution environment principle.
- Stage 1 standard runner:
  `stage1_reimage_reproduction/notebooks/kaggle_stage1_single_horizon_one_cell.md`.
- Stage 2 audit runner:
  `stage2_btc_extension/notebooks/kaggle_stage2_btc_ohlcv_audit_one_cell.md`.

## Runner Policy

Stage 2 uses the same execution pattern as Stage 1:
1. attach a code snapshot dataset to Kaggle.
2. attach the BTC OHLCV Kaggle dataset.
3. copy or extract the code snapshot into `/kaggle/working`.
4. auto-detect or explicitly set the BTC daily CSV.
5. patch config paths and runtime options.
6. run one experiment tuple at a time.
7. write outputs under `/kaggle/working/stage2_btc_extension`.

Concrete Kaggle input setup:
- Code input: attach a dataset or zip containing the `stage2_btc_extension`
  folder. The folder must include `configs/`, `src/`, `scripts/`, and
  `notebooks/`.
- Data input: attach either the public Kaggle BTC dataset
  `novandraanugrah/bitcoin-historical-datasets-2018-2024` or a private uploaded
  dataset containing `btc_1d_data_2018_to_2025.csv`.
- MA input: no separate MA dataset is needed. The Stage 2 code computes
  5/20/60-day SMA from the BTC `Close` column for the `ohlc_ma` and
  `ohlc_ma_vb` image specs.
- First Kaggle cell: list `/kaggle/input` and set `CODE_INPUT` to the actual
  attached code path. Keep `SOURCE_FILE = ""` unless BTC CSV auto-detection
  fails.
- Output preservation: write backup zips outside `PROJECT_ROOT`, under
  `/kaggle/working/stage2_saved_outputs/`, after each long-running stage. This
  prevents losing a completed model run when the next run recreates the project
  folder.

Default experiment tuple:
- `image_window = 20`
- `image_spec = ohlc_ma_vb`
- `return_horizon = 20`
- `run_seed = 42`
- `batch_size = 128`
- `split = test`
- quick Grad-CAM samples: `2` per class

Strict baseline default:
- `batch_size = 128`
- `mixed_precision = false`
- `data_parallel = false`
- `fast_cudnn = false`

Speed options:
- Mixed precision can be enabled later as a speed diagnostic.
- DataParallel is not the default because Stage 2 BTC data is small.
- Any speed option must be recorded in `run_manifest.json`.

## Planned Command Flow

The final one-cell runner will call these scripts after they are implemented:

```text
scripts/audit_btc_ohlcv.py
        ↓
scripts/run_stage2_btc_baseline.py
        ↓
scripts/evaluate_stage2_predictions.py
        ↓
scripts/generate_stage2_gradcam.py
        ↓
scripts/check_stage2_outputs.py
```

The runner should execute one tuple at a time. Full grid execution is a loop
over tuples, not one huge all-in-one process.

## Output Contract

Kaggle working outputs:

```text
outputs/stage2/{experiment}/
  checkpoints/seed_{seed}/best.pt
  predictions_seed_{seed}.csv
  predictions_averaged.csv
  classification_metrics.json
  trading_metrics.json
  trading_daily_returns.csv
  calibration_bins.csv
  run_manifest.json
  figures/
    gradcam/{split}/figure13_style.png
```

Small report copies:

```text
reports/tables/stage2_classification_metrics.csv
reports/tables/stage2_trading_metrics.csv
reports/tables/stage2_calibration_bins.csv
reports/figures/gradcam/
```

GitHub upload rule:
- planning docs, configs, source code, small summary CSVs, and small selected
  figures can be published.
- checkpoints, large prediction CSVs, raw data, and full output folders are not
  committed to GitHub.

## Full Stage 2 Grid

Default full grid:
- image windows: `I5`, `I20`, `I60`
- return horizons: `R5`, `R20`, `R60`
- image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seeds:
  - first pass: `42`
  - paper-style robustness pass: `42, 43, 44, 45, 46`

Priority order:
1. `I20/R20/ohlc_ma_vb`, seed `42`
2. `I5/R5/ohlc_ma_vb`, seed `42`
3. `I60/R60/ohlc_ma_vb`, seed `42`
4. remaining horizons for `ohlc_ma_vb`
5. image-spec ablation for all windows/horizons
6. multi-seed robustness

## Planned One-cell Interface

The planned Kaggle one-cell document is:
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md`

It is an interface draft until `2-I5` through `2-I8` create the underlying
scripts.

## 한국어

상태: checklist 2-8 계획 완료. 실제 구현은 Stage 2 script가 생긴 뒤 `2-I9`,
`2-I10`에서 합니다.

목적:
- Stage 2 실행 방식을 Stage 1 Kaggle 정책과 일관되게 유지합니다.
- Kaggle에서는 한 셀 wrapper를 사용하지만, 실제 구현은 repo의 `src/`와 `scripts/`
  안에 둡니다.
- runner code를 쓰기 전에 output path를 고정합니다.

근거:
- Root `PLAN.md`, execution environment principle.
- Stage 1 표준 runner:
  `stage1_reimage_reproduction/notebooks/kaggle_stage1_single_horizon_one_cell.md`.
- Stage 2 audit runner:
  `stage2_btc_extension/notebooks/kaggle_stage2_btc_ohlcv_audit_one_cell.md`.

## Runner Policy

Stage 2는 Stage 1과 같은 실행 패턴을 사용합니다.
1. Kaggle에 code snapshot dataset을 attach합니다.
2. BTC OHLCV Kaggle dataset을 attach합니다.
3. code snapshot을 `/kaggle/working`으로 복사하거나 압축 해제합니다.
4. BTC daily CSV를 자동 탐색하거나 명시적으로 지정합니다.
5. config path와 runtime option을 patch합니다.
6. experiment tuple 하나씩 실행합니다.
7. output은 `/kaggle/working/stage2_btc_extension` 아래에 저장합니다.

구체적인 Kaggle input 설정:
- Code input: `stage2_btc_extension` 폴더를 포함한 dataset 또는 zip을 attach합니다.
  이 폴더 안에는 `configs/`, `src/`, `scripts/`, `notebooks/`가 있어야 합니다.
- Data input: public Kaggle BTC dataset
  `novandraanugrah/bitcoin-historical-datasets-2018-2024`를 attach하거나, 로컬의
  `btc_1d_data_2018_to_2025.csv`를 private Kaggle dataset으로 업로드해서 attach합니다.
- MA input: 별도 MA dataset은 필요 없습니다. Stage 2 코드는 `ohlc_ma`,
  `ohlc_ma_vb` image spec에서 BTC `Close` column으로 5/20/60-day SMA를
  직접 계산합니다.
- 첫 Kaggle cell: `/kaggle/input`을 출력해서 실제 attach path를 확인하고,
  `CODE_INPUT`만 그 경로로 맞춥니다. BTC CSV 자동 탐색이 실패할 때만
  `SOURCE_FILE`을 정확한 CSV path로 바꿉니다.
- Output 보존: 긴 실행 단계가 끝날 때마다 `PROJECT_ROOT` 밖의
  `/kaggle/working/stage2_saved_outputs/`에 backup zip을 저장합니다. 다음 run에서
  project folder를 새로 만들더라도 완료된 model run 결과가 사라지지 않게 하기
  위한 장치입니다.

기본 experiment tuple:
- `image_window = 20`
- `image_spec = ohlc_ma_vb`
- `return_horizon = 20`
- `run_seed = 42`
- `batch_size = 128`
- `split = test`
- quick Grad-CAM sample: class당 `2`

Strict baseline 기본값:
- `batch_size = 128`
- `mixed_precision = false`
- `data_parallel = false`
- `fast_cudnn = false`

속도 옵션:
- Mixed precision은 나중에 speed diagnostic으로 켤 수 있습니다.
- Stage 2 BTC data는 작으므로 DataParallel은 기본값이 아닙니다.
- 속도 옵션을 쓰면 반드시 `run_manifest.json`에 기록합니다.

## Planned Command Flow

최종 one-cell runner는 아래 script들이 구현된 뒤 이 순서로 호출합니다.

```text
scripts/audit_btc_ohlcv.py
        ↓
scripts/run_stage2_btc_baseline.py
        ↓
scripts/evaluate_stage2_predictions.py
        ↓
scripts/generate_stage2_gradcam.py
        ↓
scripts/check_stage2_outputs.py
```

Runner는 experiment tuple 하나씩 실행합니다. full grid는 하나의 거대한 all-in-one
process가 아니라 tuple loop입니다.

## Output Contract

Kaggle working output:

```text
outputs/stage2/{experiment}/
  checkpoints/seed_{seed}/best.pt
  predictions_seed_{seed}.csv
  predictions_averaged.csv
  classification_metrics.json
  trading_metrics.json
  trading_daily_returns.csv
  calibration_bins.csv
  run_manifest.json
  figures/
    gradcam/{split}/figure13_style.png
```

작은 report copy:

```text
reports/tables/stage2_classification_metrics.csv
reports/tables/stage2_trading_metrics.csv
reports/tables/stage2_calibration_bins.csv
reports/figures/gradcam/
```

GitHub upload rule:
- planning docs, configs, source code, 작은 summary CSV, 작은 selected figure는 publish할
  수 있습니다.
- checkpoint, 대용량 prediction CSV, raw data, 전체 output folder는 GitHub에 commit하지
  않습니다.

## Full Stage 2 Grid

기본 full grid:
- image windows: `I5`, `I20`, `I60`
- return horizons: `R5`, `R20`, `R60`
- image specs: `ohlc`, `ohlc_vb`, `ohlc_ma`, `ohlc_ma_vb`
- seeds:
  - first pass: `42`
  - paper-style robustness pass: `42, 43, 44, 45, 46`

우선순위:
1. `I20/R20/ohlc_ma_vb`, seed `42`
2. `I5/R5/ohlc_ma_vb`, seed `42`
3. `I60/R60/ohlc_ma_vb`, seed `42`
4. `ohlc_ma_vb`의 나머지 horizon
5. 모든 window/horizon에 대한 image-spec ablation
6. multi-seed robustness

## Planned One-cell Interface

예정된 Kaggle one-cell 문서:
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md`

이 파일은 `2-I5`부터 `2-I8`까지 underlying script가 만들어지기 전까지는 interface
draft입니다.
