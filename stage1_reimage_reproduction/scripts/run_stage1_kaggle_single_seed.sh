#!/usr/bin/env bash
# 1-I10 Kaggle full single-seed 실행 wrapper.
#
# 사용 위치:
#   Kaggle Notebook에서 `/kaggle/working/stage1_reimage_reproduction`를 현재
#   작업 폴더로 둔 뒤 실행한다.
#
# 이 script가 하는 일:
#   1. Kaggle input data path가 config와 맞는지 확인한다.
#   2. seed 42 단일 full training을 3개 horizon에 대해 실행한다.
#   3. 각 horizon의 test prediction CSV와 metric JSON을 export한다.
#   4. 각 horizon의 2019년 Figure 13-style Grad-CAM figure를 만든다.
#   5. 마지막에 산출물 존재 여부를 검사한다.
#
# 주의:
#   이 script는 로컬 smoke test용이 아니라 Kaggle full run용이다.
#   로컬에서 실행하면 `env_kaggle.yaml`의 `/kaggle/...` path 때문에 실패하는 것이 정상이다.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${STAGE1_CONFIG_PATH:-configs/env_kaggle.yaml}"
RUN_SEED="${STAGE1_RUN_SEED:-42}"
EVAL_SPLIT="${STAGE1_EVAL_SPLIT:-test}"
GRADCAM_YEAR="${STAGE1_GRADCAM_YEAR:-2019}"
GRADCAM_SAMPLES_PER_CLASS="${STAGE1_GRADCAM_SAMPLES_PER_CLASS:-10}"

# Bash array로 horizon 순서를 명시한다. 공개 monthly_20d는 I20 image만 제공하므로
# single-seed full run은 I20/R5, I20/R20, I20/R60 세 horizon을 실행한다.
HORIZONS=(
  "stage1_i20_r5"
  "stage1_i20_r20"
  "stage1_i20_r60"
)

LOG_DIR="outputs/kaggle_logs/1-I10_seed_${RUN_SEED}"
mkdir -p "${LOG_DIR}"

echo "[1-I10] project_root=${PROJECT_ROOT}"
echo "[1-I10] config=${CONFIG_PATH}"
echo "[1-I10] seed=${RUN_SEED}"
echo "[1-I10] split=${EVAL_SPLIT}"
echo "[1-I10] gradcam_year=${GRADCAM_YEAR}"

# 1. 데이터 경로와 첫/마지막 sample shape를 먼저 확인한다.
python scripts/check_data_loading.py \
  --config "${CONFIG_PATH}" \
  --sample-indices 0 -1 \
  | tee "${LOG_DIR}/check_data_loading.json"

# 2. full_single_seed training.
#    `run_stage1_baseline.py`는 config의 `full_single_seed_run_seeds: [42]`를 읽지만,
#    여기서는 실행 로그에서 seed가 분명히 보이도록 CLI에서도 seed를 한 번 더 고정한다.
python scripts/run_stage1_baseline.py \
  --config "${CONFIG_PATH}" \
  --run-mode full_single_seed \
  --run-seeds "${RUN_SEED}" \
  | tee "${LOG_DIR}/run_stage1_baseline.json"

# 3. 학습된 best checkpoint로 horizon별 test prediction/metric을 저장한다.
for horizon in "${HORIZONS[@]}"; do
  echo "[1-I10] evaluating ${horizon}"
  python scripts/evaluate_stage1_predictions.py \
    --config "${CONFIG_PATH}" \
    --horizon "${horizon}" \
    --run-seed "${RUN_SEED}" \
    --split "${EVAL_SPLIT}" \
    | tee "${LOG_DIR}/evaluate_${horizon}_${EVAL_SPLIT}.json"
done

# 4. Figure 13-style Grad-CAM.
#    최종 논문 그림과 같은 2019년 test sample 기준으로 Up/Down 예측 각각 10개를 고른다.
for horizon in "${HORIZONS[@]}"; do
  echo "[1-I10] generating Grad-CAM ${horizon}"
  python scripts/generate_stage1_gradcam.py \
    --config "${CONFIG_PATH}" \
    --horizon "${horizon}" \
    --run-seed "${RUN_SEED}" \
    --split "${EVAL_SPLIT}" \
    --year "${GRADCAM_YEAR}" \
    --samples-per-class "${GRADCAM_SAMPLES_PER_CLASS}" \
    --write-report-copy \
    | tee "${LOG_DIR}/gradcam_${horizon}_${EVAL_SPLIT}_${GRADCAM_YEAR}.json"
done

# 5. 빠진 산출물이 있으면 여기서 실패시킨다.
python scripts/check_stage1_single_seed_outputs.py \
  --config "${CONFIG_PATH}" \
  --run-seed "${RUN_SEED}" \
  --split "${EVAL_SPLIT}" \
  --gradcam-year "${GRADCAM_YEAR}" \
  | tee "${LOG_DIR}/check_single_seed_outputs.json"

echo "[1-I10] completed"
