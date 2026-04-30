# Stage 1-I10 Kaggle Single-seed Runbook

## English

Purpose:
- Run the first full Stage 1 Re-image reproduction baseline on Kaggle with
  seed `42`.
- This is the first non-smoke execution gate. It should produce checkpoints,
  prediction CSVs, metric JSONs, a run manifest, and Figure 13-style Grad-CAM
  outputs.

Scope:
- Public data currently supports only the provided `I20 + MA + Volume` image
  shard.
- Therefore this single-seed run covers:
  - `stage1_i20_r5`
  - `stage1_i20_r20`
  - `stage1_i20_r60`

Kaggle input requirement:

```text
/kaggle/input/reimage-monthly-20d/monthly_20d/
  20d_month_has_vb_20_ma_1993_images.dat
  20d_month_has_vb_20_ma_1993_labels_w_delay.feather
  ...
  20d_month_has_vb_20_ma_2019_images.dat
  20d_month_has_vb_20_ma_2019_labels_w_delay.feather
```

Recommended Kaggle notebook cells:

Preferred code input method for a private GitHub repo:
- Upload the current `stage1_reimage_reproduction/` folder as a Kaggle Dataset
  or Kaggle Notebook input code snapshot.
- Attach that code snapshot to the Kaggle Notebook.
- Copy it into `/kaggle/working` before running.

```bash
cd /kaggle/working
cp -R /kaggle/input/thesis-stage1-code/stage1_reimage_reproduction /kaggle/working/stage1_reimage_reproduction
cd /kaggle/working/stage1_reimage_reproduction
```

If using live GitHub clone instead, the private repo requires a Kaggle Secret or
another authenticated method. Do not hard-code a token in the notebook.

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
```

The wrapper runs the following sequence:

```bash
python scripts/check_data_loading.py --config configs/env_kaggle.yaml --sample-indices 0 -1
python scripts/run_stage1_baseline.py --config configs/env_kaggle.yaml --run-mode full_single_seed --run-seeds 42
python scripts/evaluate_stage1_predictions.py --config configs/env_kaggle.yaml --horizon <horizon> --run-seed 42 --split test
python scripts/generate_stage1_gradcam.py --config configs/env_kaggle.yaml --horizon <horizon> --run-seed 42 --split test --year 2019 --samples-per-class 10 --write-report-copy
python scripts/check_stage1_single_seed_outputs.py --config configs/env_kaggle.yaml --run-seed 42 --split test --gradcam-year 2019
```

Expected output root:

```text
/kaggle/working/stage1_reimage_reproduction/outputs/
```

Key expected outputs:
- `outputs/run_manifests/run_manifest.json`
- `outputs/checkpoints/<horizon>/seed_42/best.pt`
- `outputs/checkpoints/<horizon>/seed_42/last.pt`
- `outputs/predictions/<horizon>/seed_42/test_predictions.csv`
- `outputs/metrics/<horizon>/seed_42/test_metrics.json`
- `outputs/metrics/<horizon>/seed_42/test_correlation_metrics.json`
- `outputs/figures/gradcam/<horizon>/seed_42/test/figure13_style_2019_test.png`

Interpretation note:
- Outputs from this gate are single-seed diagnostics, not yet paper-style
  5-run averaged reproduction results.
- The final paper-style reproduction remains Stage `1-I11`.

## 한국어

목적:
- Kaggle에서 seed `42`로 Stage 1 Re-image baseline 첫 full run을 실행합니다.
- 이 항목은 smoke test가 아니라 첫 non-smoke 실행 gate입니다.
- checkpoint, prediction CSV, metric JSON, run manifest, Figure 13-style
  Grad-CAM output을 만들어야 합니다.

범위:
- 현재 공개 데이터는 `I20 + MA + Volume` image shard만 직접 제공합니다.
- 따라서 single-seed run은 아래 세 horizon을 대상으로 합니다.
  - `stage1_i20_r5`
  - `stage1_i20_r20`
  - `stage1_i20_r60`

Kaggle input requirement:

```text
/kaggle/input/reimage-monthly-20d/monthly_20d/
  20d_month_has_vb_20_ma_1993_images.dat
  20d_month_has_vb_20_ma_1993_labels_w_delay.feather
  ...
  20d_month_has_vb_20_ma_2019_images.dat
  20d_month_has_vb_20_ma_2019_labels_w_delay.feather
```

권장 Kaggle Notebook cell:

Private GitHub repo 기준의 권장 code input 방식:
- 현재 `stage1_reimage_reproduction/` 폴더를 Kaggle Dataset 또는 Kaggle
  Notebook input code snapshot으로 업로드합니다.
- Kaggle Notebook에 그 code snapshot을 attach합니다.
- 실행 전에 `/kaggle/working`으로 복사합니다.

```bash
cd /kaggle/working
cp -R /kaggle/input/thesis-stage1-code/stage1_reimage_reproduction /kaggle/working/stage1_reimage_reproduction
cd /kaggle/working/stage1_reimage_reproduction
```

Live GitHub clone을 쓰려면 private repo 접근 권한 때문에 Kaggle Secret이나 다른
인증 방식이 필요합니다. Notebook에 token을 직접 적지 않습니다.

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
```

Wrapper는 아래 순서로 실행됩니다.

```bash
python scripts/check_data_loading.py --config configs/env_kaggle.yaml --sample-indices 0 -1
python scripts/run_stage1_baseline.py --config configs/env_kaggle.yaml --run-mode full_single_seed --run-seeds 42
python scripts/evaluate_stage1_predictions.py --config configs/env_kaggle.yaml --horizon <horizon> --run-seed 42 --split test
python scripts/generate_stage1_gradcam.py --config configs/env_kaggle.yaml --horizon <horizon> --run-seed 42 --split test --year 2019 --samples-per-class 10 --write-report-copy
python scripts/check_stage1_single_seed_outputs.py --config configs/env_kaggle.yaml --run-seed 42 --split test --gradcam-year 2019
```

예상 output root:

```text
/kaggle/working/stage1_reimage_reproduction/outputs/
```

주요 예상 산출물:
- `outputs/run_manifests/run_manifest.json`
- `outputs/checkpoints/<horizon>/seed_42/best.pt`
- `outputs/checkpoints/<horizon>/seed_42/last.pt`
- `outputs/predictions/<horizon>/seed_42/test_predictions.csv`
- `outputs/metrics/<horizon>/seed_42/test_metrics.json`
- `outputs/metrics/<horizon>/seed_42/test_correlation_metrics.json`
- `outputs/figures/gradcam/<horizon>/seed_42/test/figure13_style_2019_test.png`

해석상 주의:
- 이 gate의 결과는 single-seed diagnostic입니다.
- 논문식 5회 독립 재학습 평균 재현 결과는 아직 아니며, 그 항목은 Stage `1-I11`입니다.
