# 4-I13. Kaggle Five-Seed Four-Ablation Runner

## English

Status: ready for Kaggle execution, not yet complete.

Added file:
- `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`

Purpose:
- Test whether the `4-I12` seed-42 pattern is stable across random seeds.
- Keep the same selected Stage 2 baseline configuration:
  `I60/R20/ohlc_ma_vb`.
- Compare the same four Stage 4 context methods:
  `concat`, `gating`, `film_gamma`, and `film_full`.

Run grid:

```text
image window: I60
return horizon: R20
image spec: ohlc_ma_vb
context window: 60
seeds: 42, 43, 44, 45, 46
methods: concat, gating, film_gamma, film_full
total runs: 20
```

Output:
- Seed-level table:
  `reports/tables/stage4_four_ablation_five_seed_seed_results.csv`
- Mean/std table:
  `reports/tables/stage4_four_ablation_five_seed_mean_std_results.csv`
- Run summary:
  `reports/tables/stage4_four_ablation_five_seed_run_summary.json`
- Backup zips:
  disabled by default for the five-seed runner to avoid filling
  `/kaggle/working`.

Disk-space policy:
- `SAVE_BACKUP_ZIPS=False` by default.
- `DELETE_EXISTING_BACKUP_ZIPS_ON_START=True` by default.
- If a run stops with `No space left on device`, set
  `RESUME_EXISTING_PROJECT=True` and rerun the same cell. It will keep existing
  `/kaggle/working/stage4_film_conditioning` outputs, remove old backup zips,
  skip completed method/seed runs, and continue.

Completion rule:
- This item is complete only after the Kaggle five-seed cell runs and the
  output checker passes for the required 20 method/seed combinations.
- `SKIP_COMPLETED=True` is enabled, so interrupted runs can be resumed.
- The output checker uses `MIN_PREDICTIONS=1000`, so old smoke-test artifacts
  cannot be mistaken for completed full runs.

Validation:
- The Python block in the notebook Markdown compiles locally.
- The result reader now flattens nested trading metrics from `long_flat` and
  `long_short` correctly.

## 한국어

상태: Kaggle 실행 준비 완료, 아직 완료는 아님.

추가 파일:
- `notebooks/kaggle_stage4_four_ablation_five_seed_one_cell.md`

목적:
- `4-I12` seed 42 결과 패턴이 random seed에 대해 안정적인지 확인합니다.
- Stage 2 selected baseline configuration은 그대로 고정합니다:
  `I60/R20/ohlc_ma_vb`.
- Stage 4 네 context method를 동일하게 비교합니다:
  `concat`, `gating`, `film_gamma`, `film_full`.

실행 grid:

```text
image window: I60
return horizon: R20
image spec: ohlc_ma_vb
context window: 60
seeds: 42, 43, 44, 45, 46
methods: concat, gating, film_gamma, film_full
total runs: 20
```

Output:
- Seed-level table:
  `reports/tables/stage4_four_ablation_five_seed_seed_results.csv`
- Mean/std table:
  `reports/tables/stage4_four_ablation_five_seed_mean_std_results.csv`
- Run summary:
  `reports/tables/stage4_four_ablation_five_seed_run_summary.json`
- Backup zip:
  five-seed runner에서는 `/kaggle/working` 디스크가 차지 않도록 기본 비활성화.

Disk-space policy:
- `SAVE_BACKUP_ZIPS=False`가 기본값입니다.
- `DELETE_EXISTING_BACKUP_ZIPS_ON_START=True`가 기본값입니다.
- `No space left on device`로 중단되면 `RESUME_EXISTING_PROJECT=True`로 바꾸고
  같은 cell을 다시 실행합니다. 기존
  `/kaggle/working/stage4_film_conditioning` output은 유지하고, 오래된 backup
  zip은 삭제하며, 완료된 method/seed run은 skip하고 이어서 실행합니다.

완료 기준:
- Kaggle five-seed cell이 끝나고 20개 method/seed 조합의 output checker가
  통과해야 완료입니다.
- `SKIP_COMPLETED=True`라서 중간에 끊겨도 이어서 실행할 수 있습니다.
- output checker는 `MIN_PREDICTIONS=1000`을 사용하므로 과거 smoke-test output을
  full run 완료로 오인하지 않습니다.

검증:
- Notebook Markdown 안의 Python block compile을 로컬에서 통과했습니다.
- 결과 reader는 이제 `long_flat`, `long_short` nested trading metric을 제대로
  flatten합니다.
