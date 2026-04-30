# Kaggle Stage 1 Runner Skeleton

## English

This is the Kaggle runner skeleton for Stage 1. The executable runner is the
shared Python script:

```bash
python scripts/run_stage1_baseline.py --config configs/env_kaggle.yaml --run-mode full_single_seed
```

For the current `1-I10` gate, use the wrapper below so that training,
evaluation, Grad-CAM, and output checks are not separated by hand:

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
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

Preferred private-repo code input:
- Attach a Kaggle Dataset/code snapshot that contains
  `stage1_reimage_reproduction/`.

```bash
cd /kaggle/working
cp -R /kaggle/input/thesis-stage1-code/stage1_reimage_reproduction /kaggle/working/stage1_reimage_reproduction
cd /kaggle/working/stage1_reimage_reproduction
```

For live clone from the private GitHub repo, use Kaggle Secrets or another
authenticated method. Do not hard-code tokens.

```python
from pathlib import Path
assert Path("/kaggle/input/reimage-monthly-20d/monthly_20d").exists()
```

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
```

Manual command sequence, if debugging one step at a time:

```bash
python scripts/check_data_loading.py --config configs/env_kaggle.yaml --sample-indices 0 -1
```

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed
```

Export predictions and metrics after training:

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test
```

Generate Figure 13-style Grad-CAM after test predictions exist:

```bash
python scripts/generate_stage1_gradcam.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test \
  --year 2019 \
  --samples-per-class 10 \
  --write-report-copy
```

For the final paper-style run:

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_paper_style
```

Then export each seed prediction and average probabilities:

```bash
for seed in 42 43 44 45 46; do
  python scripts/evaluate_stage1_predictions.py \
    --config configs/env_kaggle.yaml \
    --horizon stage1_i20_r20 \
    --run-seed "$seed" \
    --split test
done

python scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --split test \
  --average-seed-predictions 42 43 44 45 46
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

현재 `1-I10` gate에서는 학습, 평가, Grad-CAM, output check를 손으로 나누지
않도록 아래 wrapper를 사용합니다.

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
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

Private repo 기준 권장 code input:
- `stage1_reimage_reproduction/`가 들어 있는 Kaggle Dataset/code snapshot을
  attach합니다.

```bash
cd /kaggle/working
cp -R /kaggle/input/thesis-stage1-code/stage1_reimage_reproduction /kaggle/working/stage1_reimage_reproduction
cd /kaggle/working/stage1_reimage_reproduction
```

Private GitHub repo에서 live clone하려면 Kaggle Secrets나 다른 인증 방식이
필요합니다. token을 notebook에 직접 적지 않습니다.

```python
from pathlib import Path
assert Path("/kaggle/input/reimage-monthly-20d/monthly_20d").exists()
```

```bash
bash scripts/run_stage1_kaggle_single_seed.sh
```

한 단계씩 debug할 때의 manual command sequence:

```bash
python scripts/check_data_loading.py --config configs/env_kaggle.yaml --sample-indices 0 -1
```

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_single_seed
```

학습 이후 prediction과 metric을 export합니다.

```bash
python scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test
```

test prediction이 생긴 뒤 Figure 13-style Grad-CAM을 생성합니다.

```bash
python scripts/generate_stage1_gradcam.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --run-seed 42 \
  --split test \
  --year 2019 \
  --samples-per-class 10 \
  --write-report-copy
```

최종 paper-style run은 아래처럼 실행합니다.

```bash
python scripts/run_stage1_baseline.py \
  --config configs/env_kaggle.yaml \
  --run-mode full_paper_style
```

그 다음 seed별 prediction을 export하고 probability를 평균합니다.

```bash
for seed in 42 43 44 45 46; do
  python scripts/evaluate_stage1_predictions.py \
    --config configs/env_kaggle.yaml \
    --horizon stage1_i20_r20 \
    --run-seed "$seed" \
    --split test
done

python scripts/evaluate_stage1_predictions.py \
  --config configs/env_kaggle.yaml \
  --horizon stage1_i20_r20 \
  --split test \
  --average-seed-predictions 42 43 44 45 46
```

출력은 아래에 저장됩니다.

```text
/kaggle/working/stage1_reimage_reproduction/outputs/
```
