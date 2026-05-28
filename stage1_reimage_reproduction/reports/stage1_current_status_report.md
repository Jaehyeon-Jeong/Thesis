# Stage 1 Current Kaggle Output Status

## English

Status date: 2026-05-28

Stage 1 now has seed-42 full-data fast diagnostic outputs for all three public
I20 horizons. The implementation follows the Stage 1 Re-image reproduction
pipeline and uses the same public `monthly_20d` images for `I20/R5`, `I20/R20`,
and `I20/R60`.

Run setting:
- Run mode: `full_single_seed`
- Seed: `42`
- Batch size: `1024`
- Mixed precision: `True`
- DataParallel: `True`
- fast cuDNN: `True`
- Train/validation image preload: `True`
- Split: train/validation years `1993-2000`, test years `2001-2019`
- Grad-CAM preview: test split, year `2019`, `2` samples per predicted class

This is a full-data fast diagnostic, not the final strict paper-style
reproduction. Strict reproduction remains later work with batch size `128` and
five independent seeds/runs.

| Experiment | Current artifact status | Split | Samples | Accuracy | Majority accuracy | Accuracy - majority | ROC-AUC | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| `I20/R5` | Full seed-42 fast diagnostic | test | 1,399,933 | 0.5273 | 0.5078 | +0.0195 | 0.5373 | Output check passed; checkpoint, test predictions, metrics, correlation metrics, and Grad-CAM are present in Kaggle backup. |
| `I20/R20` | Full seed-42 fast diagnostic | test | 1,393,845 | 0.5285 | 0.5222 | +0.0063 | 0.5339 | Output check passed; checkpoint, test predictions, metrics, correlation metrics, and Grad-CAM are present in Kaggle backup. |
| `I20/R60` | Full seed-42 fast diagnostic | test | 1,376,215 | 0.5312 | 0.5408 | -0.0096 | 0.5298 | Earlier seed-42 full test artifact is archived; same fast diagnostic family. |

Important interpretation:
- The Stage 1 experiment pipeline is implemented and has now been exercised
  end-to-end for all three public I20 horizons.
- `I20/R5` and `I20/R20` beat the majority-class baseline in this seed-42 fast
  diagnostic. `I20/R60` has positive ROC-AUC signal but does not beat the
  majority-class baseline by accuracy.
- These results should be reported as seed-42 fast diagnostics because the run
  used batch size `1024`, mixed precision, DataParallel, and fast cuDNN.
- The current Grad-CAM previews use `2` predicted-up and `2` predicted-down
  examples. Final Figure-13-style output should use `10` up and `10` down.

Current result table:
- [Stage 1 seed-42 current status CSV](tables/stage1_seed42_current_status.csv)

Current `I20/R60` Grad-CAM preview:

![Stage 1 I20/R60 Grad-CAM preview](figures/gradcam/stage1_i20_r60_seed42_test_2019_figure13_style.png)

Note:
- The newly generated `I20/R5` and `I20/R20` Grad-CAM figures are preserved in
  the Kaggle output backup zip. They are not tracked in GitHub yet because the
  large output package was not committed.

## 한국어

상태 기준일: 2026-05-28

Stage 1은 이제 public I20 horizon 3개 전체에 대해 seed-42 full-data fast
diagnostic output을 확보했습니다. 구현은 Stage 1 Re-image reproduction
pipeline을 따르며, 동일한 public `monthly_20d` 이미지를 사용해 `I20/R5`,
`I20/R20`, `I20/R60`을 실행했습니다.

실행 설정:
- Run mode: `full_single_seed`
- Seed: `42`
- Batch size: `1024`
- Mixed precision: `True`
- DataParallel: `True`
- fast cuDNN: `True`
- Train/validation image preload: `True`
- Split: train/validation years `1993-2000`, test years `2001-2019`
- Grad-CAM preview: test split, year `2019`, predicted class당 sample `2`개

이 결과는 full-data fast diagnostic이며 최종 strict paper-style reproduction은
아닙니다. Strict reproduction은 batch size `128`과 five independent seeds/runs로
나중에 수행합니다.

| 실험 | 현재 artifact 상태 | Split | Sample 수 | Accuracy | Majority accuracy | Accuracy - majority | ROC-AUC | 비고 |
|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| `I20/R5` | Full seed-42 fast diagnostic | test | 1,399,933 | 0.5273 | 0.5078 | +0.0195 | 0.5373 | Output check 통과. checkpoint, test prediction, metric, correlation metric, Grad-CAM이 Kaggle backup에 있습니다. |
| `I20/R20` | Full seed-42 fast diagnostic | test | 1,393,845 | 0.5285 | 0.5222 | +0.0063 | 0.5339 | Output check 통과. checkpoint, test prediction, metric, correlation metric, Grad-CAM이 Kaggle backup에 있습니다. |
| `I20/R60` | Full seed-42 fast diagnostic | test | 1,376,215 | 0.5312 | 0.5408 | -0.0096 | 0.5298 | 이전에 보존한 seed-42 full test artifact이며 같은 fast diagnostic 계열입니다. |

중요 해석:
- Stage 1 실험 파이프라인은 구현 완료됐고, public I20 horizon 3개 전체에서
  end-to-end로 실행됐습니다.
- 이번 seed-42 fast diagnostic에서 `I20/R5`와 `I20/R20`은 majority-class
  baseline을 넘었습니다. `I20/R60`은 ROC-AUC signal은 있으나 accuracy 기준으로는
  majority-class baseline보다 낮습니다.
- 이 결과는 batch size `1024`, mixed precision, DataParallel, fast cuDNN을 쓴
  seed-42 fast diagnostic으로 보고해야 합니다.
- 현재 Grad-CAM preview는 predicted-up 2개, predicted-down 2개입니다. 최종
  Figure 13 스타일 output은 up 10개, down 10개로 다시 만들어야 합니다.

현재 결과표:
- [Stage 1 seed-42 current status CSV](tables/stage1_seed42_current_status.csv)

현재 `I20/R60` Grad-CAM preview:

![Stage 1 I20/R60 Grad-CAM preview](figures/gradcam/stage1_i20_r60_seed42_test_2019_figure13_style.png)

참고:
- 새로 생성된 `I20/R5`, `I20/R20` Grad-CAM 그림은 Kaggle output backup zip에
  보존되어 있습니다. 대용량 output package를 GitHub에 commit하지 않았기 때문에
  아직 GitHub tracked figure로는 올리지 않았습니다.
