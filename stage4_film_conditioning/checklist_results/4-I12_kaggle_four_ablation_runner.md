# 4-I12. Kaggle Four-Ablation Single-Config Runner

## English

Status: ready for Kaggle execution.

Added file:
- `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`

This item prepares the real Stage 4 Kaggle runner for the fixed first full
configuration:

```text
image window: I60
return horizon: R20
image spec: ohlc_ma_vb
context window: 60
seed: 42
context methods: concat, gating, film_gamma, film_full
```

The Kaggle cell runs the following sequence:

```text
copy Stage 4 code
copy Stage 2 dependency code
patch configs/env_kaggle.yaml
audit BTC and Fear & Greed sources
build context features
for each context method:
    train
    prediction evaluation
    trading evaluation
    Grad-CAM plus context/gate/gamma/beta export
    output receipt check
    backup zip
write compact summary table
final backup zip
```

Output preservation:
- Backup root: `/kaggle/working/stage4_saved_outputs`
- The runner writes backup zips after:
  - `after_context_build`
  - `after_train`
  - `after_prediction_eval`
  - `after_trading_eval`
  - `after_gradcam`
  - `after_output_check`
  - `after_summary`

Completion rule:
- A model is complete only if `scripts/check_stage4_outputs.py` passes.
- A checkpoint alone is not enough.
- The cell supports `SKIP_COMPLETED=True`, so interrupted runs can resume by
  rerunning the same cell.

Important:
- This file is a Kaggle execution cell, not a completed experiment result.
- After the user runs it in Kaggle and shares the summary/zip outputs, this
  checklist item should be marked complete with the real metrics.

Validation performed locally:
- The Python block in the notebook Markdown compiles successfully.

## 한국어

상태: Kaggle 실행 준비 완료.

추가 파일:
- `notebooks/kaggle_stage4_four_ablation_single_seed_one_cell.md`

이 항목은 Stage 4의 첫 실제 Kaggle full configuration runner를 준비합니다.

```text
image window: I60
return horizon: R20
image spec: ohlc_ma_vb
context window: 60
seed: 42
context methods: concat, gating, film_gamma, film_full
```

Kaggle cell 실행 순서:

```text
Stage 4 code copy
Stage 2 dependency code copy
configs/env_kaggle.yaml patch
BTC / Fear & Greed source audit
context feature build
각 context method마다:
    train
    prediction evaluation
    trading evaluation
    Grad-CAM + context/gate/gamma/beta export
    output receipt check
    backup zip
compact summary table 저장
final backup zip
```

Output 보존:
- Backup root: `/kaggle/working/stage4_saved_outputs`
- Runner는 아래 단계 후 zip backup을 저장합니다.
  - `after_context_build`
  - `after_train`
  - `after_prediction_eval`
  - `after_trading_eval`
  - `after_gradcam`
  - `after_output_check`
  - `after_summary`

완료 기준:
- `scripts/check_stage4_outputs.py`가 통과해야 해당 model이 완료입니다.
- checkpoint만 있는 상태는 완료가 아닙니다.
- `SKIP_COMPLETED=True`를 지원하므로 중간에 끊겨도 같은 cell을 다시 실행해
  이어갈 수 있습니다.

중요:
- 이 파일은 Kaggle 실행 cell이며, 아직 완료된 실험 결과가 아닙니다.
- 사용자가 Kaggle에서 실행한 뒤 summary/zip output을 공유하면 실제 metric으로
  이 checklist item을 완료 처리해야 합니다.

로컬 검증:
- Notebook Markdown 안의 Python block이 compile 통과했습니다.
