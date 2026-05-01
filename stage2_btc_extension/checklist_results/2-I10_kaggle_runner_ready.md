# 2-I10 Kaggle Full BTC Baseline Runner

## English

Status: ready for user Kaggle run

Updated runner:
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md`

How to run in Kaggle:
- Create a new Kaggle Notebook and turn on GPU.
- Attach the Stage 2 code snapshot dataset. It must contain the
  `stage2_btc_extension` folder with `configs/`, `src/`, `scripts/`, and
  `notebooks/`.
- Attach the BTC OHLCV data in one of two ways:
  - public Kaggle dataset:
    `novandraanugrah/bitcoin-historical-datasets-2018-2024`
  - or a private uploaded dataset containing `btc_1d_data_2018_to_2025.csv`
- Do not upload a separate MA file. Stage 2 computes MA from the BTC `Close`
  column inside the code.
- Paste the input discovery cell and then the Python cell from
  `notebooks/kaggle_stage2_btc_baseline_one_cell.md`.
- Set or verify:
  - `CODE_INPUT` points to the attached Stage 2 code folder/zip shown under
    `/kaggle/input`
  - `DATA_ROOT = Path("/kaggle/input")`
  - `SOURCE_FILE = ""` unless auto-detection fails
  - `IMAGE_WINDOW = 20`
  - `IMAGE_SPEC = "ohlc_ma_vb"`
  - `RETURN_HORIZON = 20`
  - `SMOKE_TEST = False`
- Run one tuple at a time.

For quick Kaggle verification:
- Set `SMOKE_TEST = True`.

First-run path check:

```python
from pathlib import Path

for p in Path("/kaggle/input").glob("*"):
    print("\nINPUT:", p)
    for child in list(p.glob("*"))[:15]:
        print(" ", child)
```

Full-run expected output root:
- `/kaggle/working/stage2_btc_extension/outputs/stage2`

Automatic backup:
- The runner now writes backup zips to `/kaggle/working/stage2_saved_outputs/`
  after training, prediction evaluation, trading evaluation, Grad-CAM, and
  output check.
- These backups are outside `PROJECT_ROOT`, so they survive the next run that
  recreates `/kaggle/working/stage2_btc_extension`.

Important:
- This checklist item creates the Kaggle runner. The actual full Kaggle result
  is not available until the user runs it in Kaggle and returns the output.

## 한국어

상태: 사용자가 Kaggle에서 실행할 수 있는 상태

업데이트한 runner:
- `notebooks/kaggle_stage2_btc_baseline_one_cell.md`

Kaggle 실행 방법:
- Kaggle에서 새 Notebook을 만들고 GPU를 켭니다.
- Stage 2 code snapshot dataset을 attach합니다. 이 dataset에는
  `stage2_btc_extension` 폴더와 그 안의 `configs/`, `src/`, `scripts/`,
  `notebooks/`가 있어야 합니다.
- BTC OHLCV data는 둘 중 하나로 attach합니다.
  - public Kaggle dataset:
    `novandraanugrah/bitcoin-historical-datasets-2018-2024`
  - 또는 로컬의 `btc_1d_data_2018_to_2025.csv`를 private Kaggle dataset으로
    업로드해서 attach
- MA 파일은 따로 업로드하지 않습니다. Stage 2 코드는 BTC `Close` column으로 MA를
  직접 계산합니다.
- input discovery cell을 먼저 붙여넣고, 그다음
  `notebooks/kaggle_stage2_btc_baseline_one_cell.md` 안의 Python cell을 붙여넣습니다.
- 다음을 설정하거나 확인합니다.
  - `CODE_INPUT`: `/kaggle/input` 아래에 attach된 Stage 2 code folder/zip 경로
  - `DATA_ROOT = Path("/kaggle/input")`
  - `SOURCE_FILE = ""`: 자동 탐색. 실패할 때만 정확한 CSV path로 변경
  - `IMAGE_WINDOW = 20`
  - `IMAGE_SPEC = "ohlc_ma_vb"`
  - `RETURN_HORIZON = 20`
  - `SMOKE_TEST = False`
- experiment tuple 하나씩 실행합니다.

Kaggle에서 빠르게 확인하려면:
- `SMOKE_TEST = True`로 바꿉니다.

처음 실행할 path 확인 cell:

```python
from pathlib import Path

for p in Path("/kaggle/input").glob("*"):
    print("\nINPUT:", p)
    for child in list(p.glob("*"))[:15]:
        print(" ", child)
```

Full-run 예상 output root:
- `/kaggle/working/stage2_btc_extension/outputs/stage2`

자동 backup:
- runner는 training, prediction evaluation, trading evaluation, Grad-CAM,
  output check 직후 `/kaggle/working/stage2_saved_outputs/`에 backup zip을
  저장합니다.
- 이 backup은 `PROJECT_ROOT` 밖에 있으므로, 다음 run에서
  `/kaggle/working/stage2_btc_extension`을 새로 만들어도 남아 있습니다.

중요:
- 이 체크리스트는 Kaggle runner를 작성한 것입니다. 실제 full Kaggle 결과는 사용자가
  Kaggle에서 실행하고 output을 가져온 뒤 확정합니다.
