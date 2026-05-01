# 2-I10 Kaggle Full BTC Baseline Runner

## English

Status: ready for user Kaggle run

Updated runner:
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md`

How to run in Kaggle:
- Attach the Stage 2 code snapshot dataset.
- Attach the BTC OHLCV dataset.
- Paste the Python cell from `notebooks/kaggle_stage2_btc_baseline_one_cell.md`.
- Set:
  - `IMAGE_WINDOW = 20`
  - `IMAGE_SPEC = "ohlc_ma_vb"`
  - `RETURN_HORIZON = 20`
  - `SMOKE_TEST = False`
- Run one tuple at a time.

For quick Kaggle verification:
- Set `SMOKE_TEST = True`.

Full-run expected output root:
- `/kaggle/working/stage2_btc_extension/outputs/stage2`

Important:
- This checklist item creates the Kaggle runner. The actual full Kaggle result
  is not available until the user runs it in Kaggle and returns the output.

## 한국어

상태: 사용자가 Kaggle에서 실행할 수 있는 상태

업데이트한 runner:
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md`

Kaggle 실행 방법:
- Stage 2 code snapshot dataset을 attach합니다.
- BTC OHLCV dataset을 attach합니다.
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md` 안의 Python cell을 붙여넣습니다.
- 다음처럼 둡니다.
  - `IMAGE_WINDOW = 20`
  - `IMAGE_SPEC = "ohlc_ma_vb"`
  - `RETURN_HORIZON = 20`
  - `SMOKE_TEST = False`
- experiment tuple 하나씩 실행합니다.

Kaggle에서 빠르게 확인하려면:
- `SMOKE_TEST = True`로 바꿉니다.

Full-run 예상 output root:
- `/kaggle/working/stage2_btc_extension/outputs/stage2`

중요:
- 이 체크리스트는 Kaggle runner를 작성한 것입니다. 실제 full Kaggle 결과는 사용자가
  Kaggle에서 실행하고 output을 가져온 뒤 확정합니다.
