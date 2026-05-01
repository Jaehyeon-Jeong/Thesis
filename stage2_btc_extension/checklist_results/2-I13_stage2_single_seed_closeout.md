# 2-I13. Stage 2 Single-Seed Closeout

## English

Status: Stage 2 single-seed package closed for now.

Closed scope:
- Stage 2 BTC baseline grid with seed `42`.
- `36` experiments:
  `3` image windows x `3` return horizons x `4` image specifications.
- README result narrative updated with:
  - four BTC image specifications and sample images,
  - three return horizons,
  - window/horizon/spec comparison tables,
  - best/promising/caution result tiers,
  - Grad-CAM preview and Figure-13-style follow-up cell.

Main single-seed finding:
- Best configuration: `I60 / R20 / ohlc_ma_vb`.
- Test accuracy: `0.6031`.
- Test ROC-AUC: `0.6169`.
- Accuracy minus majority-class baseline: `+0.0618`.

Risk statement:
- No severe train-validation overfitting is visible in the best run at the best
  epoch.
- However, the result still has small-sample and model-selection risk because
  it is the best result selected from `36` single-seed configurations.
- Final stability claims require five-seed reruns, especially for the `I60/R20`
  leading group.

Deferred follow-ups:
- Five-seed grid or focused five-seed rerun for the leading configurations.
- Re-Imagining Figure-13-style BTC Grad-CAM with `10` predicted-up and `10`
  predicted-down images. The Kaggle generation cell is already prepared in
  `notebooks/kaggle_stage2_best_gradcam_10_one_cell.md`.

## 한국어

상태: Stage 2 single-seed 결과 패키지는 현재 기준으로 마무리합니다.

마무리한 범위:
- seed `42` 기준 Stage 2 BTC baseline grid.
- 총 `36`개 실험:
  `3` image window x `3` return horizon x `4` image specification.
- README 결과 설명을 업데이트했습니다.
  - BTC image specification 4가지와 sample image,
  - return horizon 3가지,
  - window/horizon/spec별 비교표,
  - best/promising/caution 결과 구분,
  - Grad-CAM preview와 Figure 13 스타일 후속 생성 cell.

single-seed 핵심 결과:
- Best configuration: `I60 / R20 / ohlc_ma_vb`.
- Test accuracy: `0.6031`.
- Test ROC-AUC: `0.6169`.
- Majority-class baseline 대비 accuracy 개선: `+0.0618`.

위험/제한 해석:
- best run의 best epoch 기준으로 심한 train-validation overfitting은 보이지 않습니다.
- 하지만 `36`개 single-seed configuration 중 가장 좋은 결과를 고른 것이므로,
  small-sample risk와 model-selection risk는 남아 있습니다.
- 최종 안정성 주장은 특히 `I60/R20` leading group에 대한 5-seed rerun 이후에만
  해야 합니다.

후속으로 남긴 것:
- leading configuration에 대한 5-seed grid 또는 focused 5-seed rerun.
- `Predicted Up` 10개와 `Predicted Down` 10개를 포함하는 Re-Imagining Figure 13
  스타일 BTC Grad-CAM. 생성용 Kaggle cell은
  `notebooks/kaggle_stage2_best_gradcam_10_one_cell.md`에 준비되어 있습니다.
