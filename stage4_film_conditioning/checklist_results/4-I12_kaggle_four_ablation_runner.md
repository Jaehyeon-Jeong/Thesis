# 4-I12. Kaggle Four-Ablation Single-Config Run

## English

Status: complete for seed `42`.

Kaggle run:

```text
image window: I60
return horizon: R20
image spec: ohlc_ma_vb
context window: 60
seed: 42
context methods: concat, gating, film_gamma, film_full
```

Completion evidence:
- The four methods all returned `status = ok` in the received Kaggle summary.
- Each method produced 1,441 test predictions.
- The Stage 4 output checker passed for the run artifacts.
- The compact result table is tracked at
  `reports/tables/stage4_four_ablation_seed42_run_summary.csv`.

Seed-42 classification results:

| Method | Accuracy | Majority | Accuracy - majority | ROC-AUC | F1 | Brier |
|:---|---:|---:|---:|---:|---:|---:|
| concat | 0.489244 | 0.541291 | -0.052047 | 0.500816 | 0.459618 | 0.250146 |
| gating | 0.552394 | 0.541291 | 0.011103 | 0.555933 | 0.621701 | 0.287193 |
| film_gamma | 0.459403 | 0.541291 | -0.081888 | 0.526125 | 0.002561 | 0.251857 |
| film_full | 0.584316 | 0.541291 | 0.043026 | 0.596811 | 0.680533 | 0.278363 |

Interpretation:
- `concat` is below the majority baseline, so simple side-information fusion is
  not enough in this seed.
- `gating` is modestly above majority and provides a useful non-FiLM modulation
  baseline.
- `film_gamma` has acceptable ROC-AUC but near-zero F1, indicating a degenerate
  decision-threshold behavior in this seed.
- `film_full` is the best Stage 4 method in seed 42. It beats the majority
  baseline by about `+4.30` percentage points and has the best ROC-AUC.

Stage 2 comparison:
- Stage 2 selected five-seed baseline for `I60/R20/ohlc_ma_vb`:
  accuracy mean `0.579320`, ROC-AUC mean `0.584862`.
- Stage 4 `film_full` seed 42:
  accuracy `0.584316`, ROC-AUC `0.596811`.
- Stage 2 same-configuration seed 42:
  accuracy `0.603053`, ROC-AUC `0.616950`.

Conclusion:
- `4-I12` can be marked complete.
- The result is promising for full FiLM, but it is not enough to claim a robust
  improvement over Stage 2 because the same seed Stage 2 result is still higher.
- The next required check is `4-I13`: run the same four ablations for seeds
  `42, 43, 44, 45, 46`.

Note:
- The received compact summary confirms that trading metric files exist, but
  the first viewer table did not flatten the nested trading JSON values. The
  Kaggle notebooks have been patched so later summaries read `long_flat` and
  `long_short` trading metrics correctly.

## 한국어

상태: seed `42` 기준 완료.

Kaggle 실행 조건:

```text
image window: I60
return horizon: R20
image spec: ohlc_ma_vb
context window: 60
seed: 42
context methods: concat, gating, film_gamma, film_full
```

완료 근거:
- 받은 Kaggle summary에서 네 방법 모두 `status = ok`입니다.
- 각 방법은 test prediction 1,441개를 생성했습니다.
- Stage 4 output checker가 run artifact를 통과했습니다.
- compact result table은
  `reports/tables/stage4_four_ablation_seed42_run_summary.csv`에 보존했습니다.

Seed-42 classification 결과:

| Method | Accuracy | Majority | Accuracy - majority | ROC-AUC | F1 | Brier |
|:---|---:|---:|---:|---:|---:|---:|
| concat | 0.489244 | 0.541291 | -0.052047 | 0.500816 | 0.459618 | 0.250146 |
| gating | 0.552394 | 0.541291 | 0.011103 | 0.555933 | 0.621701 | 0.287193 |
| film_gamma | 0.459403 | 0.541291 | -0.081888 | 0.526125 | 0.002561 | 0.251857 |
| film_full | 0.584316 | 0.541291 | 0.043026 | 0.596811 | 0.680533 | 0.278363 |

해석:
- `concat`은 majority baseline보다 낮으므로, 단순히 context embedding을 붙이는
  방식만으로는 충분하지 않았습니다.
- `gating`은 majority보다 조금 높아서 non-FiLM modulation 기준선으로 의미가
  있습니다.
- `film_gamma`는 ROC-AUC는 완전히 무너지지 않았지만 F1이 거의 0이라, 이 seed에서
  decision-threshold 쪽이 비정상적으로 쏠린 결과입니다.
- `film_full`이 seed 42의 Stage 4 네 방법 중 가장 좋습니다. majority 대비 약
  `+4.30`%p 높고 ROC-AUC도 가장 높습니다.

Stage 2 비교:
- Stage 2 selected five-seed baseline `I60/R20/ohlc_ma_vb`:
  accuracy mean `0.579320`, ROC-AUC mean `0.584862`.
- Stage 4 `film_full` seed 42:
  accuracy `0.584316`, ROC-AUC `0.596811`.
- Stage 2 같은 configuration의 seed 42:
  accuracy `0.603053`, ROC-AUC `0.616950`.

결론:
- `4-I12`는 완료 처리해도 됩니다.
- full FiLM은 promising하지만, 같은 seed의 Stage 2보다 높지는 않으므로 아직 robust
  improvement라고 주장하면 안 됩니다.
- 다음 필수 확인은 `4-I13`: 같은 네 ablation을 seeds `42, 43, 44, 45, 46`으로
  다시 돌리는 것입니다.

메모:
- 받은 compact summary에는 trading metric 파일 존재 여부는 확인되지만, 첫 viewer
  table이 nested trading JSON 값을 flatten하지 못해서 trading 값이 비어 있었습니다.
  이후 Kaggle notebook은 `long_flat`, `long_short` nested trading metric을 제대로
  읽도록 패치했습니다.
