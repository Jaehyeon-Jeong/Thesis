# 4-I1 Shared Stage 4 Config/Code Scaffold

## English

Status: complete.

Purpose:
- create the shared Stage 4 config/code foundation before implementing context
  features and models;
- make all future Stage 4 scripts import both Stage 4 `src` and Stage 2 `src`;
- keep Stage 2 BTC data/image/split/evaluation helpers as the fixed dependency.

Created files:
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`
- `src/stage4_film/__init__.py`
- `src/stage4_film/config.py`
- `src/stage4_film/paths.py`
- `src/stage4_film/runtime.py`
- `src/stage4_film/seed.py`
- `scripts/_stage4_script_utils.py`
- `scripts/check_stage4_scaffold.py`

Config decisions:
- Primary experiment stays fixed to `I60/R20/ohlc_ma_vb`.
- Primary context window is `60`.
- Main context methods are:
  - `concat`
  - `gating`
  - `film_gamma`
  - `film_full`
- Primary context vector remains the 8 matched-window features:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
- Stage 2 dependency is explicit through `stage2_dependency.src_path`.
- Raw F&G data is referenced locally but not tracked by GitHub.

Scaffold check:

```bash
python -m py_compile \
  src/stage4_film/__init__.py \
  src/stage4_film/config.py \
  src/stage4_film/paths.py \
  src/stage4_film/runtime.py \
  src/stage4_film/seed.py \
  scripts/_stage4_script_utils.py \
  scripts/check_stage4_scaffold.py

python scripts/check_stage4_scaffold.py --config configs/env_local.yaml
```

Result:
- status: `ok`
- device: `cpu`
- BTC source found:
  `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- F&G source found:
  `/Users/jaehyeonjeong/Desktop/논문/stage4_film_conditioning/F&G_data/fear_greed_index.csv`
- Stage 2 source dependency found:
  `/Users/jaehyeonjeong/Desktop/논문/stage2_btc_extension/src`

Generated experiment names:
- `stage4_concat_i60_ohlc_ma_vb_r20_c60`
- `stage4_gating_i60_ohlc_ma_vb_r20_c60`
- `stage4_film_gamma_i60_ohlc_ma_vb_r20_c60`
- `stage4_film_full_i60_ohlc_ma_vb_r20_c60`

Next:
- proceed to `4-I2`: structured context source audit and feature builder.

## 한국어

상태: 완료.

목적:
- context feature와 model 구현 전에 Stage 4 공통 config/code 기반을 만듭니다.
- 이후 모든 Stage 4 script가 Stage 4 `src`와 Stage 2 `src`를 함께 import하게 합니다.
- Stage 2 BTC data/image/split/evaluation helper를 고정 dependency로 유지합니다.

생성한 파일:
- `configs/env_local.yaml`
- `configs/env_kaggle.yaml`
- `src/stage4_film/__init__.py`
- `src/stage4_film/config.py`
- `src/stage4_film/paths.py`
- `src/stage4_film/runtime.py`
- `src/stage4_film/seed.py`
- `scripts/_stage4_script_utils.py`
- `scripts/check_stage4_scaffold.py`

Config 결정:
- Primary experiment는 `I60/R20/ohlc_ma_vb`로 유지합니다.
- Primary context window는 `60`입니다.
- Main context method:
  - `concat`
  - `gating`
  - `film_gamma`
  - `film_full`
- Primary context vector는 matched-window 8개 feature로 유지합니다:
  `fg_value`, `fg_mean_60`, `fg_delta_60`, `fg_std_60`,
  `bb_percent_b_60`, `bb_bandwidth_60`, `mfi_60`, `rv_60`.
- Stage 2 dependency는 `stage2_dependency.src_path`로 명시합니다.
- Raw F&G data는 로컬에서 참조하지만 GitHub에는 track하지 않습니다.

Scaffold check:

```bash
python -m py_compile \
  src/stage4_film/__init__.py \
  src/stage4_film/config.py \
  src/stage4_film/paths.py \
  src/stage4_film/runtime.py \
  src/stage4_film/seed.py \
  scripts/_stage4_script_utils.py \
  scripts/check_stage4_scaffold.py

python scripts/check_stage4_scaffold.py --config configs/env_local.yaml
```

결과:
- status: `ok`
- device: `cpu`
- BTC source 확인:
  `/Users/jaehyeonjeong/Desktop/논문/데이터셋/BTC _OHLCV/btc_1d_data_2018_to_2025.csv`
- F&G source 확인:
  `/Users/jaehyeonjeong/Desktop/논문/stage4_film_conditioning/F&G_data/fear_greed_index.csv`
- Stage 2 source dependency 확인:
  `/Users/jaehyeonjeong/Desktop/논문/stage2_btc_extension/src`

생성된 experiment name:
- `stage4_concat_i60_ohlc_ma_vb_r20_c60`
- `stage4_gating_i60_ohlc_ma_vb_r20_c60`
- `stage4_film_gamma_i60_ohlc_ma_vb_r20_c60`
- `stage4_film_full_i60_ohlc_ma_vb_r20_c60`

다음:
- `4-I2`: structured context source audit and feature builder로 진행합니다.
