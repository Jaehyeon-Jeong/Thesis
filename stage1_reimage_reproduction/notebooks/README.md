# Stage 1 Notebooks

The Kaggle runner skeleton is now available as:

```text
kaggle_stage1_runner.md
```

The single-horizon one-cell Kaggle runner is available as:

```text
kaggle_stage1_single_horizon_one_cell.md
```

The executable runner is:

```text
../scripts/run_stage1_baseline.py
```

Purpose:
- Load `configs/env_kaggle.yaml`.
- Validate the attached `monthly_20d` Kaggle Dataset.
- Run Stage 1 training/evaluation commands after implementation.
- Save checkpoints, metrics, predictions, figures, and run manifests under
  `/kaggle/working/stage1_reimage_reproduction/outputs`.

한국어:
- Kaggle runner skeleton은 `kaggle_stage1_runner.md`에 있습니다.
- horizon 하나를 한 셀로 실행하는 Kaggle runner는
  `kaggle_stage1_single_horizon_one_cell.md`에 있습니다.
- 실제 실행은 `../scripts/run_stage1_baseline.py`를 사용합니다.
