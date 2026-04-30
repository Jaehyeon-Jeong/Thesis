# Kaggle Stage 1 Runner Skeleton

## English

This is the Kaggle runner skeleton for Stage 1. The executable runner is the
shared Python script:

```bash
python scripts/run_stage1_baseline.py --config configs/env_kaggle.yaml --run-mode full_single_seed
```

Expected Kaggle input layout:

```text
/kaggle/input/reimage-monthly-20d/monthly_20d/
  20d_month_has_vb_20_ma_1993_images.dat
  20d_month_has_vb_20_ma_1993_labels_w_delay.feather
  ...
  20d_month_has_vb_20_ma_2019_images.dat
  20d_month_has_vb_20_ma_2019_labels_w_delay.feather
```

Recommended Kaggle notebook cells:

```python
from pathlib import Path
assert Path("/kaggle/input/reimage-monthly-20d/monthly_20d").exists()
```

```bash
python scripts/check_data_loading.py --config configs/env_kaggle.yaml --sample-indices 0 -1
```

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed
```

For the final paper-style run after evaluation output is implemented:

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_paper_style
```

Outputs are written under:

```text
/kaggle/working/stage1_reimage_reproduction/outputs/
```

## 한국어

이 문서는 1단계 Kaggle runner skeleton입니다. 실제 실행 runner는 공통 Python
script입니다.

```bash
python scripts/run_stage1_baseline.py --config configs/env_kaggle.yaml --run-mode full_single_seed
```

Kaggle input layout은 아래를 기대합니다.

```text
/kaggle/input/reimage-monthly-20d/monthly_20d/
  20d_month_has_vb_20_ma_1993_images.dat
  20d_month_has_vb_20_ma_1993_labels_w_delay.feather
  ...
  20d_month_has_vb_20_ma_2019_images.dat
  20d_month_has_vb_20_ma_2019_labels_w_delay.feather
```

권장 Kaggle notebook cell:

```python
from pathlib import Path
assert Path("/kaggle/input/reimage-monthly-20d/monthly_20d").exists()
```

```bash
python scripts/check_data_loading.py --config configs/env_kaggle.yaml --sample-indices 0 -1
```

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed
```

evaluation output 구현 이후 최종 paper-style run은 아래처럼 실행합니다.

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_paper_style
```

출력은 아래에 저장됩니다.

```text
/kaggle/working/stage1_reimage_reproduction/outputs/
```
