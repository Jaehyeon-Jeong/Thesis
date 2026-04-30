# 1-I8 Grad-CAM Implementation Result

## English

Status:
- Completed on 2026-05-01.

Summary:
- Implemented Grad-CAM for the Stage 1 I20 CNN baseline.
- Added a command-line script that reads prediction CSVs, selects Up/Down
  examples, loads the trained checkpoint, computes layer-wise Grad-CAM, and
  writes Figure 13-style outputs.

Files:
- `src/stage1_reimage/interpretability/__init__.py`
- `src/stage1_reimage/interpretability/gradcam.py`
- `scripts/generate_stage1_gradcam.py`
- `docs/gradcam_implementation.md`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

Validation:
- `python -m compileall src scripts`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`

Result:
- Smoke Grad-CAM output was generated successfully.
- The local figure is explicitly marked as a smoke output, not a reproduction
  figure.

## 한국어

상태:
- 2026-05-01 완료.

요약:
- Stage 1 I20 CNN baseline용 Grad-CAM을 구현했습니다.
- prediction CSV를 읽고, Up/Down 예시를 고르고, 학습된 checkpoint를 load하고,
  layer별 Grad-CAM을 계산해 Figure 13-style output을 저장하는 command-line script를
  추가했습니다.

파일:
- `src/stage1_reimage/interpretability/__init__.py`
- `src/stage1_reimage/interpretability/gradcam.py`
- `scripts/generate_stage1_gradcam.py`
- `docs/gradcam_implementation.md`
- `reports/figures/gradcam/stage1_i20_r20_seed_42_validation_1993_figure13_style.png`

검증:
- `python -m compileall src scripts`
- `python scripts/generate_stage1_gradcam.py --config configs/env_local.yaml --horizon stage1_i20_r20 --run-seed 42 --split validation --year 1993 --samples-per-class 1 --write-report-copy`

결과:
- smoke Grad-CAM output이 정상 생성되었습니다.
- local figure는 재현 figure가 아니라 smoke output임을 명시했습니다.
