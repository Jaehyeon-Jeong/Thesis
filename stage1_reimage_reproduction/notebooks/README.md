# Stage 1 Notebooks

The canonical Kaggle execution interface is:

```text
kaggle_stage1_single_horizon_one_cell.md
```

It is a copy-paste notebook cell that:
- copies the Stage 1 code snapshot into `/kaggle/working`
- copies the public stock image data into `/tmp`
- patches `configs/env_kaggle.yaml`
- runs one horizon through training, test evaluation, and quick Grad-CAM

The cell calls these modular scripts:

```text
../scripts/check_data_loading.py
../scripts/run_stage1_baseline.py
../scripts/evaluate_stage1_predictions.py
../scripts/generate_stage1_gradcam.py
```

Purpose:
- Keep Kaggle usage simple while preserving the modular thesis codebase.
- Save checkpoints, metrics, predictions, figures, and run manifests under
  `/kaggle/working/stage1_reimage_reproduction/outputs`.

한국어:
- Stage 1의 공식 Kaggle 실행 interface는
  `kaggle_stage1_single_horizon_one_cell.md`입니다.
- 이 파일의 cell을 Kaggle에 복붙해서 horizon 하나를
  training, test evaluation, quick Grad-CAM까지 실행합니다.
- 구현을 중복하지 않고 `../scripts/`의 modular script들을 호출합니다.
